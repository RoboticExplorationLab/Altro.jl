struct ALConstraintSet2{T}
    constraints::Vector{ALConstraint{T}}
    c_max::Vector{T}
    μ_max::Vector{T}
end

function ALConstraintSet2{T}(cons::TO.ConstraintList) where T
    n,m = cons.n, cons.m
    ncon = length(cons)
    constraints = map(1:ncon) do i
        ALConstraint{T}(n, m, cons[i], cons.inds[i], sig=cons.sigs[i], diffmethod=cons.diffs[i])
    end
    constraints = convert(Vector{ALConstraint{T}}, constraints)

    c_max = zeros(ncon)
    μ_max = zeros(ncon)

    ALConstraintSet2{T}(constraints, c_max, μ_max)
end

# Indexing and Iteration
@inline Base.length(conset::ALConstraintSet2) = length(conset.constraints)
@inline Base.getindex(conset::ALConstraintSet2, i::Integer) = conset.constraints[i]
Base.firstindex(::ALConstraintSet2) = 1
Base.lastindex(conset::ALConstraintSet2) = length(conset.constraints)

function Base.iterate(conset::ALConstraintSet2) 
    isempty(conset.constraints) ? nothing : (conset.constraints[1], 1)
end

function Base.iterate(conset::ALConstraintSet2, state::Int) 
    state >= length(conset) ? nothing : (conset.constraints[state+1], state+1)
end

Base.IteratorSize(::ALConstraintSet2) = Base.HasLength()
Base.IteratorEltype(::ALConstraintSet2) = Base.HasEltype()
Base.eltype(::ALConstraintSet2{T}) where T = ALConstraint{T}

# Methods
function evaluate_constraints!(conset::ALConstraintSet2, Z)
    for alcon in conset.constraints
        evaluate_constraint!(alcon, Z)
    end
end

function constraint_jacobians!(conset::ALConstraintSet2, Z)
    for alcon in conset.constraints
        constraint_jacobian!(alcon, Z)
    end
end

function alcost(conset::ALConstraintSet2{T}) where T
    J = zero(T)
    for alcon in conset.constraints
        J += alcost(alcon)
    end
    return J
end

for method in (:algrad!, :alhess!, :dualupdate!, :penaltyupdate!, :normviolation!, 
        :max_penalty, :reset_duals!, :reset_penalties!, :add_alcost_expansion!)
    @eval function $method(conset::ALConstraintSet2, args...)
        for alcon in conset.constraints
            $method(alcon, args...)
        end
    end
end

function max_violation(conset::ALConstraintSet2)
    for i = 1:length(conset) 
        conset.c_max[i] = max_violation(conset.constraints[i])
    end
    return maximum(conset.c_max)
end

function max_penalty(conset::ALConstraintSet2)
    for i = 1:length(conset) 
        conset.μ_max[i] = max_penalty(conset.constraints[i])
    end
    return maximum(conset.μ_max)
end

function reset!(conset::ALConstraintSet2, opts::SolverOptions)
    for con in conset.constraints
        setparams!(con, opts)
        reset_duals!(con)
        reset_penalties!(con)
    end
end