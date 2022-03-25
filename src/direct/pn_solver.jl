getinputinds(fun::RD.AbstractFunction, n, m) = getinputinds(RD.functioninputs(fun), n, m)
getinputinds(::RD.StateControl, n, m) = 1:n+m
getinputinds(::RD.StateOnly, n, m) = 1:n
getinputinds(::RD.ControlOnly, n, m) = n .+ (1:m)

"""
    ProjectedNewtonSolver

Projected Newton Solver
Direct method developed by the REx Lab at Stanford University
Achieves machine-level constraint satisfaction by projecting onto the feasible subspace.
    It can also take a full Newton step by solving the KKT system.
This solver is to be used exlusively for solutions that are close to the optimal solution.
    It is intended to be used as a "solution polishing" method for augmented Lagrangian methods.
"""
struct ProjectedNewtonSolver2{L<:DiscreteDynamics,O<:AbstractObjective,Nx,Nu,T,F} <: ConstrainedSolver{T}
    # Problem Info
    model::L
    obj::O
    x0::Vector{T}
    conset::PNConstraintSet{T}
    opts::SolverOptions{T}
    stats::SolverStats{T}
    funsig::F

    # Data
    _data::Vector{T}  # storage for primal and dual variables
    Zdata::VectorView{T,Int}  
    Z̄data::VectorView{T,Int}
    dY::Vector{T}                 # KKT solution vector
    Ydata::VectorView{T,Int}  
    Atop::SparseMatrixCSC{T,Int}  # top Np rows of KKT matrix
    colptr::Vector{Int}
    rowval::Vector{Int}
    nzval::Vector{T}
    d::Vector{T}               # constraints
    b::Vector{T}               # rhs vector
    active::BitVector

    # Trajectories
    Z::SampledTrajectory{Nx,Nu,T,KnotPoint{Nx,Nu,VectorView{T,Int},T}}
    Z̄::SampledTrajectory{Nx,Nu,T,KnotPoint{Nx,Nu,VectorView{T,Int},T}}

    # Cost function
    hess::Vector{Matrix{T}}
    hessdiag::Vector{Diagonal{T,SubArray{T,1,Matrix{T}, Tuple{Vector{CartesianIndex{2}}}, false}}}
    grad::Vector{VectorView{T,Int}}

    # Dynamics
    ∇f::Matrix{Matrix{T}}  # dynamics Jacobians
    f::Vector{Vector{T}}           # dynamics values 
    e::Vector{VectorView{T,Int}}   # dynamics residuals
    Iinit::Diagonal{T,Vector{T}}
    Ireg::Diagonal{T,Vector{T}}    # primal regularization

    # Indices
    ix::Vector{UnitRange{Int}}
    iu::Vector{UnitRange{Int}}
    iz::Vector{UnitRange{Int}}
    blocks::SparseBlocks
    hessblocks::Vector{SparseBlockIndex}
    ∇fblocks::Matrix{SparseBlockIndex}
    qdldl::Cqdldl.QDLDLSolver{T,Int}
end

function ProjectedNewtonSolver2(prob::Problem{T}, opts::SolverOptions=SolverOptions{T}(), 
                                stats::SolverStats=SolverStats(parent=solvername(ProjectedNewtonSolver2)); 
                                use_static::Val{USE_STATIC}=usestaticdefault(get_model(prob)[1]),
                                kwargs...) where {T,USE_STATIC}
    nx,nu,N = RD.dims(prob)
    Nc = sum(num_constraints(prob.constraints))  # stage constraints
    @assert length(nx) == N
    @assert length(nu) == N
    Np = sum(nx) + sum(nu)
    Nd = sum(nx) + Nc       # number of duals (no constraints)

    set_options!(opts; kwargs...)
    if stats.parent == solvername(ProjectedNewtonSolver2) 
        reset!(stats, opts.iterations)
    end

    nn = [[0;0] reshape(cumsum(vec([nx'; nu'])), 2, :)]
    ix = map(2:N+1) do k
        nn[2,k-1] + 1 : nn[1,k]
    end
    iu = map(2:N) do k
        nn[1,k] + 1 : nn[2,k]
    end
    iz = map(2:N+1) do k
        nn[2,k-1] + 1 : nn[2,k]
    end

    _data = zeros(2Np + Nd + 2nu[1])  # leave extra room for terminal control
    Zdata = view(_data, 1:Np)
    Z̄data = view(_data, Np .+ (1:Np)) 
    Ydata = view(_data, 2Np + 2nu[1] .+ (1:Nd)) 
    Atop = spzeros(T, Np, Np + Nd)
    d = zeros(T, Nd)
    b = zeros(T, Np + Nd)  # allocate maximum size
    active = trues(Np + Nd)
    Ireg = Diagonal(fill(-one(T), Nd))

    Z = SampledTrajectory([KnotPoint{nx[k],nu[k]}(view(_data, iz[k]), prob.Z[k].t, prob.Z[k].dt) for k = 1:N])
    Z[end].dt = Inf 
    Z̄ = SampledTrajectory([KnotPoint{nx[k],nu[k]}(view(_data, Np .+ iz[k]), prob.Z[k].t, prob.Z[k].dt) for k = 1:N])
    Z̄[end].dt = Inf 
    dY = zeros(T, Np + Nd)

    # Create sparse blocks and initialize cost Hessian sparsity
    blocks = SparseBlocks(Np, Np + Nd)
    isdiag = prob.obj[1] isa TO.QuadraticCostFunction ? TO.is_diag(prob.obj[1]) : false
    for i in iz
        addblock!(blocks, i, i, isdiag)
    end
    id = Np .+ (1:Nd)  # dual indices

    # Create the constraint set
    # NOTE: this initializes the sparsity structure in Atop
    conset = PNConstraintSet(prob.constraints, Z̄, opts, Atop, d, view(active, id), blocks)

    # Initialize the storage for the full sparse KKT matrix
    #   These will get dynamically resized depending on the active set
    nnz_Atop = nnz(Atop)
    nnz_A = nnz_Atop + Nd  # add nnz for the dual regularization 
    colptr = zeros(Int, Np + Nd + 1)
    rowval = zeros(Int, nnz_A)
    nzval = zeros(T, nnz_A)

    # Storage for the dynamics expansion
    ∇f = [j == 1 ? zeros(T, nx[k], nx[k]+nu[k]) : zeros(T, nx[k], nx[k]) for k = 1:N-1, j = 1:2]
    f = [zeros(T,nx[k]) for k = 1:N-1]
    e = [view(d, conset.cinds[end][k+1]) for k = 1:N-1]

    # Cache the sparse view blocks for the dynamics Jacobians
    izk(k,j) = j == 1 ? iz1(k) : iz2(k)
    iz1(k) = iz[k]    #(k-1) * (n + m) .+ (1:n+m)
    iz2(k) = ix[k+1]  #(k) * (n + m) .+ (1:n)
    ci(k) = conset.cinds[end][k+1] .+ Np
    ∇fblocks = [
        begin blocks[izk(k,j), ci(k)] end for k = 1:N-1, j = 1:2
    ] 
    ∇fblocks = vcat(
        [blocks[ix[1], ci(0)] blocks[ix[1], ci(0)]], 
        ∇fblocks
    )

    # Storage for the cost expansion
    hess = [zeros(T, length(i), length(i)) for i in iz]
    grad = [view(b,i) for i in iz]
    hessdiag = map(hess) do H
        inds = [CartesianIndex(i,i) for i = 1:size(H,1)]
        Diagonal(view(H, inds))
    end
    hessblocks = map(iz) do i  # cache for the Hessian block views
        blocks[i, i]
    end

    Iinit = -Diagonal(ones(nx[1]))
    Ireg = Diagonal(ones(nx[1]+nu[1]))  # TODO: allow varying sizes
    qdldl = QDLDLSolver{T}(Np + Nd, nnz_A, 2*nnz_A)

    funsig = USE_STATIC ? RD.StaticReturn() : RD.InPlace()

    ProjectedNewtonSolver2(prob.model[1], prob.obj, Vector(prob.x0), conset, opts, stats, funsig,
        _data, Zdata, Z̄data, dY, Ydata, Atop, colptr, rowval, nzval, d, b, active,
        Z, Z̄, hess, hessdiag, grad, ∇f, f, e, Iinit, Ireg, ix, iu, iz, blocks, hessblocks, 
        ∇fblocks, qdldl,
    )
end

# Getters
RD.dims(pn::ProjectedNewtonSolver2{<:Any,<:Any,n,m}) where {n,m} = n,m,length(pn.ix)
TO.get_objective(pn::ProjectedNewtonSolver2) = pn.obj
TO.get_model(pn::ProjectedNewtonSolver2) = pn.model
TO.get_trajectory(pn::ProjectedNewtonSolver2) = pn.Z
iterations(pn::ProjectedNewtonSolver2) = pn.stats.iterations_pn
solvername(::Type{<:ProjectedNewtonSolver2}) = :ProjectedNewton
num_primals(pn::ProjectedNewtonSolver2) = length(pn.Zdata)
num_duals(pn::ProjectedNewtonSolver2) = length(pn.Ydata)
TO.get_constraints(pn::ProjectedNewtonSolver2) = pn.conset

function primalregularization!(pn::ProjectedNewtonSolver2)
    ρ_primal = pn.opts.ρ_primal
    Ireg = pn.Ireg
    Ireg.diag .= ρ_primal
    for block in pn.hessblocks 
        pn.Atop[block] .+= Ireg
    end
end

function cost_hessian!(pn::ProjectedNewtonSolver2, Z::SampledTrajectory=pn.Z)
    obj = pn.obj
    for k in eachindex(Z)
        RD.hessian!(obj.diffmethod[k], obj.cost[k], pn.hess[k], Z[k])

        # Copy to Sparse Array
        hessblock = pn.hessblocks[k]
        if hessblock.block.isdiag
            pn.Atop[hessblock] .= pn.hessdiag[k]
        else
            pn.Atop[hessblock] .= UpperTriangular(pn.hess[k])
        end
    end
end

function cost_gradient!(pn::ProjectedNewtonSolver2, Z::SampledTrajectory=pn.Z)
    obj = pn.obj
    for k in eachindex(Z)
        RD.gradient!(obj.diffmethod[k], obj.cost[k], pn.grad[k], Z[k])
    end
end

function dynamics_expansion!(pn::ProjectedNewtonSolver2{<:Any,<:Any,Nx,Nu}, 
        Z::SampledTrajectory=pn.Z̄, ∇f=pn.∇f) where {Nx,Nu}
    diff = pn.opts.dynamics_diffmethod
    model = pn.model
    n,_,N = RD.dims(pn)
    for k = 1:N - 1
        RD.jacobian!(pn.funsig, diff, model, ∇f[k,1], pn.f[k], Z[k])

        # Copy to Sparse Array
        ∇fview1 = pn.Atop[pn.∇fblocks[k+1,1]]'
        ∇fview1 .= ∇f[k,1]

        ∇fblock2 = pn.∇fblocks[k+1,2]
        pn.Atop[∇fblock2] .= pn.Iinit  # assumes explicit
    end
end

function dynamics_error!(pn::ProjectedNewtonSolver2, Z::SampledTrajectory=pn.Z̄)
    model = pn.model
    N = length(Z)
    for k = 1:N - 1
        RD.evaluate!(pn.funsig, model, pn.f[k], Z[k])
        RD.state_diff!(model, pn.e[k], pn.f[k], RD.state(Z[k+1]))
    end
end

function evaluate_constraints!(pn::ProjectedNewtonSolver2, Z::SampledTrajectory=pn.Z̄)
    ix = pn.ix
    n = RD.dims(pn)[1]

    # Initial condition
    pn.d[ix[1]] .= pn.x0 .- RD.state(Z[1])

    # Dynamics 
    dynamics_error!(pn, Z)

    # Stage constraints
    evaluate_constraints!(pn.conset, Z)
end

function constraint_jacobians!(pn::ProjectedNewtonSolver2, Z::SampledTrajectory=pn.Z̄)
    ix = pn.ix
    n = RD.state_dim(pn.model)

    # Initial condition
    # TODO: skip this since it's constant
    x0block = pn.∇fblocks[1]
    pn.Atop[x0block] .= pn.Iinit

    # Dynamics
    dynamics_expansion!(pn, Z)

    # Stage constraints
    constraint_jacobians!(pn.conset, Z)
end

function max_violation(pn::ProjectedNewtonSolver2, Z::SampledTrajectory=pn.Z̄)
    evaluate_constraints!(pn, Z)
    update_active_set!(pn)
    max_violation(pn, nothing)
end

function norm_violation(pn::ProjectedNewtonSolver2, p=1)
    return norm(pn.d[pn.active], p)
end

function max_violation(pn::ProjectedNewtonSolver2, Z::Nothing)
    Np = num_primals(pn)
    c_max = zero(eltype(pn.d))
    for i in eachindex(pn.d)
        if pn.active[i + Np]
            c_max = max(c_max, abs(pn.d[i]))
        end
    end
    return c_max
end


function update_active_set!(pn::ProjectedNewtonSolver2)
    update_active_set!(pn.conset)
    return nothing
end

function active_constraints(pn::ProjectedNewtonSolver2)
    return sparse(pn.D)[pn.active, :], pn.d[pn.active]
end