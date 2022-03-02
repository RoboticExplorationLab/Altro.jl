struct PNConstraintSet{T}
    constraints::Vector{PNConstraint{T}}
    cinds::Vector{Vector{UnitRange{Int}}}
    c_max::Vector{T}
    Zref::Ref{SampledTrajectory}
end

function PNConstraintSet(cons::TO.ConstraintList, Z::SampledTrajectory, 
                         opts::SolverOptions, A, d, a, blocks)
    # NOTE: blocks needs to already have all of the blocks for the objective Hessian added
    n,m = cons.n, cons.m
    ncon = length(cons)
    N = length(cons.p)
    cinds = [[1:0 for j in eachindex(inds)] for inds in cons.inds]
    push!(cinds, [1:0 for k = 1:N])  # dynamics and initial condition constraints
    T = eltype(A)

    # Sort by knot point
    idx = 0
    for k = 1:N
        # Initial condition
        if k == 1
            cinds[end][1] = 1:n
            idx += n
        end
        # Stage constraint
        for (i,con) in enumerate(cons)
            inds = cons.inds[i]
            p = RD.output_dim(con)
            if insorted(k, inds) 
                j = k - inds[1] + 1
                cinds[i][j] = idx .+ (1:p)
                idx += p
            end
        end
        # Dynamics constraint
        if k < N
            cinds[end][k+1] = idx .+ (1:n)
            idx += n
        end
    end

    cblocks = [[(0:0, 0:0) for j in cons.inds[i]] for i = 1:ncon]

    # Add Jacobian (transpose) blocks
    Np = RD.num_vars(Z)
    addblock!(blocks, 1:n, cinds[end][1] .+ Np, true)
    for i = 1:ncon
        ii = getinputinds(cons[i], n, m)
        for (j,k) in enumerate(cons.inds[i])
            iz = (k-1) * (n + m) .+ ii
            ic = cinds[i][j] .+ Np
            addblock!(blocks, iz, ic)
            cblocks[i][j] = (iz, ic)
        end
    end
    for k = 1:N-1
        iz1 = (k-1) * (n + m) .+ (1:n+m)
        iz2 = (k) * (n + m) .+ (1:n)
        ic = cinds[end][k+1] .+ Np
        addblock!(blocks, iz1, ic)
        addblock!(blocks, iz2, ic, true)  # assumes explicit integrators
    end

    # Initialize the sparsity structure in the KKT Jacobian
    initialize!(blocks, A)

    constraints = map(1:ncon) do i
        views = [A[blocks[block...]] for block in cblocks[i]]

        PNConstraint(Z, cons[i], cons.inds[i], d, a, views, sig=cons.sigs[i], diffmethod=cons.diffs[i], solveropts=opts)
    end
    constraints = convert(Vector{PNConstraint{T}}, constraints)
    c_max = zeros(T, ncon)
    Zref = Ref{SampledTrajectory}(Z)
    PNConstraintSet(constraints, cinds, c_max, Zref)
end

# Indexing and Iteration
@inline Base.length(conset::PNConstraintSet) = length(conset.constraints)
@inline Base.getindex(conset::PNConstraintSet, i::Integer) = conset.constraints[i]
Base.firstindex(::PNConstraintSet) = 1
Base.lastindex(conset::PNConstraintSet) = length(conset.constraints)

function Base.iterate(conset::PNConstraintSet) 
    isempty(conset.constraints) ? nothing : (conset.constraints[1], 1)
end

function Base.iterate(conset::PNConstraintSet, state::Int) 
    state >= length(conset) ? nothing : (conset.constraints[state+1], state+1)
end

Base.IteratorSize(::PNConstraintSet) = Base.HasLength()
Base.IteratorEltype(::PNConstraintSet) = Base.HasEltype()
Base.eltype(::PNConstraintSet{T}) where T = PNConstraint{T}

# Methods
function settraj!(conset::PNConstraintSet, Z)
    for pncon in conset.constraints
        settraj!(pncon, Z)
    end
end

function evaluate_constraints!(conset::PNConstraintSet, Z)
    if Z !== conset.Zref[]
        println("Updating Z")
        settraj!(conset, Z)
    end
    for pncon in conset.constraints
        evaluate_constraints!(pncon)
    end
end

function constraint_jacobians!(conset::PNConstraintSet, Z)
    if Z !== conset.Zref[]
        settraj!(conset, Z)
    end
    for pncon in conset.constraints
        constraint_jacobians!(pncon)
    end
end

function update_active_set!(conset::PNConstraintSet)
    for pncon in conset.constraints 
        update_active_set!(pncon) 
    end
end

function reset!(conset::PNConstraintSet)
    for pncon in conset.constraints 
        reset!(pncon) 
    end
end