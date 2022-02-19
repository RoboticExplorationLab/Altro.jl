
"""
$(TYPEDEF)
Projected Newton Solver
Direct method developed by the REx Lab at Stanford University
Achieves machine-level constraint satisfaction by projecting onto the feasible subspace.
    It can also take a full Newton step by solving the KKT system.
This solver is to be used exlusively for solutions that are close to the optimal solution.
    It is intended to be used as a "solution polishing" method for augmented Lagrangian methods.
"""
struct ProjectedNewtonSolver2{L<:DiscreteDynamics,O<:AbstractObjective,Nx,Nu,T} <: ConstrainedSolver{T}
    # Problem Info
    model::L
    obj::O
    x0::Vector{T}
    conset::PNConstraintSet{T}
    opts::SolverOptions{T}
    stats::SolverStats{T}

    # Data
    Zdata::Vector{T}  # storage for primal variables
    Z̄data::Vector{T}
    Ydata::Vector{T}  # storage for dual variables
    A::SparseMatrixCSC{T,Int}  # KKT matrix
    b::Vector{T}               # KKT vector
    H::SparseView{T,Int}
    g::VectorView{T,Int}
    D::SparseView{T,Int}
    d::VectorView{T,Int}
    active::BitVector

    # Trajectories
    Z::Traj{Nx,Nu,T,KnotPoint{Nx,Nu,VectorView{T,Int},T}}
    Z̄::Traj{Nx,Nu,T,KnotPoint{Nx,Nu,VectorView{T,Int},T}}

    # Cost function
    hess::Vector{SparseView{T,Int}}
    grad::Vector{VectorView{T,Int}}

    # Dynamics
    ∇f::Matrix{SparseView{T,Int}}  # dynamics Jacobians
    f::Vector{Vector{T}}           # dynamics values 
    e::Vector{VectorView{T,Int}}   # dynamics residuals

    # Indices
    ix::Vector{UnitRange{Int}}
    iu::Vector{UnitRange{Int}}
    iz::Vector{UnitRange{Int}}
    id::Vector{UnitRange{Int}}
end

function ProjectedNewtonSolver2(prob::Problem{T}, opts::SolverOptions=SolverOptions{T}(), 
                                stats::SolverStats=SolverStats(parent=solvername(ProjectedNewtonSolver2)); kwargs...) where T
    n,m,N = RD.dims(prob)
    Nc = sum(num_constraints(prob.constraints))  # stage constraints
    Np = N*n + (N-1)*m  # number of primals
    Nd = N*n + Nc       # number of duals (no constraints)

    set_options!(opts; kwargs...)

    ix = [(1:n) .+ (k-1)*(n+m) for k = 1:N]
    iu = [n .+ (1:m) .+ (k-1)*(n+m) for k = 1:N-1]
    iz = push!([(1:n+m) .+ (k-1)*(n+m) for k = 1:N-1], ix[end])

    Zdata = zeros(T, Np)
    Z̄data = zeros(T, Np)
    Ydata = zeros(T, Nd)
    A = spzeros(T, Np + Nd, Np + Nd)
    b = zeros(T, Np + Nd)
    H = view(A, 1:Np, 1:Np) 
    g = view(b, 1:Np) 
    D = view(A, Np .+ (1:Nd), 1:Np) 
    d = view(b, Np .+ (1:Nd)) 
    active = trues(Nd)

    conset = PNConstraintSet(prob.constraints, D, d, active)

    Z = Traj([KnotPoint{n,m}(view(Zdata, iz[k]), prob.Z[k].t, prob.Z[k].dt) for k = 1:N])
    Z[end].dt = 0
    Z̄ = Traj([KnotPoint{n,m}(view(Z̄data, iz[k]), prob.Z[k].t, prob.Z[k].dt) for k = 1:N])
    Z̄[end].dt = 0

    hess = [view(A,i,i) for i in iz]
    grad = [view(b,i) for i in iz]

    id = [Np + n + (k-1)*n .+ (1:n) for k = 1:N-1]
    ∇f = [view(D, conset.cinds[end][k+1], j ? iz[k] : ix[k+1]) for k = 1:N-1, j in (true, false)] 
    f = [zeros(T,n) for k = 1:N-1]
    e = [view(d, conset.cinds[end][k+1]) for k = 1:N-1]

    ProjectedNewtonSolver2(prob.model, prob.obj, Vector(prob.x0), conset, opts, stats, Zdata, Z̄data, Ydata,
        A, b, H, g, D, d, active, Z, Z̄, hess, grad, ∇f, f, e, ix, iu, iz, id)

    # ProjectedNewtonSolver2(prob.model, prob.obj, prob.x0, opts, stats, Zdata, Ydata, H, g, 
    #                        hess, grad, Z, ∇f, f, e, ix, iu, iz ,id)
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

function cost_hessian!(pn::ProjectedNewtonSolver2, Z::AbstractTrajectory=pn.Z)
    obj = pn.obj
    for k in eachindex(Z)
        RD.hessian!(obj.diffmethod[k], obj.cost[k], pn.hess[k], Z[k])
    end
end

function cost_gradient!(pn::ProjectedNewtonSolver2, Z::AbstractTrajectory=pn.Z)
    obj = pn.obj
    for k in eachindex(Z)
        RD.gradient!(obj.diffmethod[k], obj.cost[k], pn.grad[k], Z[k])
    end
end

function dynamics_expansion!(pn::ProjectedNewtonSolver2, Z::AbstractTrajectory=pn.Z)
    sig = pn.opts.dynamics_funsig
    diff = pn.opts.dynamics_diffmethod
    model = pn.model
    n,_,N = RD.dims(pn)
    for k = 1:N - 1
        RD.jacobian!(sig, diff, model, pn.∇f[k,1], pn.f[k], Z[k])
        pn.∇f[k,2] .= -I(n)
    end
end

function dynamics_error!(pn::ProjectedNewtonSolver2, Z::AbstractTrajectory=pn.Z)
    sig = pn.opts.dynamics_funsig
    model = pn.model
    N = length(Z)
    for k = 1:N - 1
        RD.evaluate!(sig, model, pn.f[k], Z[k])
        RD.state_diff!(model, pn.e[k], pn.f[k], RD.state(Z[k+1]))
    end
end

function evaluate_constraints!(pn::ProjectedNewtonSolver2, Z::AbstractTrajectory=pn.Z)
    ix = pn.ix
    n = RD.dims(pn)[1]

    # Initial condition
    pn.b[ix[end] .+ n] .= RD.state(Z[1]) .- pn.x0

    # Dynamics 
    dynamics_error!(pn, Z)

    # Stage constraints
    evaluate_constraints!(pn.conset, Z)
end

function constraint_jacobians!(pn::ProjectedNewtonSolver2, Z::AbstractTrajectory=pn.Z)
    ix = pn.ix
    n = RD.state_dim(pn.model)

    # Initial condition
    # TODO: skip this since it's constant
    pn.A[ix[end] .+ n, ix[1]] .= I(n)

    # Dynamics
    dynamics_expansion!(pn, Z)

    # Stage constraints
    constraint_jacobians!(pn.conset, Z)
end

function TO.max_violation(pn::ProjectedNewtonSolver2, Z::Traj=pn.Z)
    evaluate_constraints!(pn, Z)
    update_active_set!(pn)
    max_violation(pn, nothing)
end

function TO.max_violation(pn::ProjectedNewtonSolver2, Z::Nothing)
    c_max = zero(eltype(pn.d))
    for i in eachindex(pn.d)
        if pn.active[i]
            c_max = max(c_max, abs(pn.d[i]))
        end
    end
    return c_max
    # norm(view(pn.d, pn.active), Inf)  # TODO: update this for inequality constraints
end


function update_active_set!(pn::ProjectedNewtonSolver2)
    for con in pn.conset
        update_active_set!(con, pn.opts.active_set_tolerance_pn)
    end
    return nothing
end

function active_constraints(pn::ProjectedNewtonSolver2)
    return pn.D[pn.active, :], pn.d[pn.active]
end