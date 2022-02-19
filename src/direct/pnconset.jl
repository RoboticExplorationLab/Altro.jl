struct PNConstraintSet{T}
    constraints::Vector{PNConstraint{T}}
    cinds::Vector{Vector{UnitRange{Int}}}
    c_max::Vector{T}
end

function PNConstraintSet(cons::TO.ConstraintList, D, d, a)
    n,m = cons.n, cons.m
    ncon = length(cons)
    N = length(cons.p)
    cinds = [[1:0 for j in eachindex(inds)] for inds in cons.inds]
    push!(cinds, [1:0 for k = 1:N])  # dynamics and initial condition constraints
    T = eltype(D)

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

    constraints = map(1:ncon) do i
        PNConstraint(n, m, N, cons[i], cons.inds[i], D, d, a, cinds[i], sig=cons.sigs[i], diffmethod=cons.diffs[i])
    end
    constraints = convert(Vector{PNConstraint{T}}, constraints)
    c_max = zeros(T, ncon)
    PNConstraintSet(constraints, cinds, c_max)
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
function evaluate_constraints!(conset::PNConstraintSet, Z)
    for alcon in conset.constraints
        evaluate_constraint!(alcon, Z)
    end
end

function constraint_jacobians!(conset::PNConstraintSet, Z)
    for alcon in conset.constraints
        constraint_jacobian!(alcon, Z)
    end
end