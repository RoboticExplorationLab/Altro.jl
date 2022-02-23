struct ALConstraintSet2{T}
    constraints::Vector{ALConstraint{T}}
    c_max::Vector{T}
    μ_max::Vector{T}
    Zref::Ref{AbstractTrajectory}
    Eref::Ref{CostExpansion2}
end

function ALConstraintSet2{T}() where T
    constraints = ALConstraint{T}[]
    c_max = T[]
    μ_max = T[]
    Z = Traj(KnotPoint{Any,Any,T,Vector{T}}[]) 
    Zref = Ref{RD.AbstractTrajectory}(Z)
    E = CostExpansion2{T}(0,0,0)
    Eref = Ref{CostExpansion2}(E)

    ALConstraintSet2{T}(constraints, c_max, μ_max, Zref, Eref)
end

function initialize!(conset::ALConstraintSet2{T}, cons::TO.ConstraintList, 
                     Z::AbstractTrajectory, opts::SolverOptions,
                     costs, E=CostExpansion2{T}(RD.dims(Z)...)) where T
    n,m = cons.n, cons.m
    @assert RD.state_dim(Z) == n
    @assert RD.control_dim(Z) == m
    @assert length(Z) == length(cons.p)
    ncon = length(cons)
    for i = 1:ncon
        alcon = ALConstraint{T}(Z, cons[i], cons.inds[i], costs, E, sig=cons.sigs[i], 
                                diffmethod=cons.diffs[i], solveropts=opts)
        push!(conset.constraints, alcon)
    end
    resize!(conset.c_max, length(Z))
    resize!(conset.μ_max, ncon)
    conset.c_max .= 0
    conset.μ_max .= 0
    conset.Zref[] = Z
    conset.Eref[] = E
    return
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
for method in (:evaluate_constraints!, :constraint_jacobians!) 
    @eval function $method(conset::ALConstraintSet2)
        for alcon in conset.constraints
            $method(alcon)
        end
    end
    @eval function $method(conset::ALConstraintSet2, Z::AbstractTrajectory)
        # Hack to avoid an allocation from using a function barrier w/ more than 1 arg
        # Store a pointer to the trajectory in each conval
        # Only update the trajectory if it's different than the current one
        if Z !== conset.Zref[]
            settraj!(conset, Z)  # this allocates
        end
        $method(conset)  # this does not, but something like  
                         # `evaluate_constraints!(conset, Z)` does
    end
end

function alcost(conset::ALConstraintSet2{T}) where T
    for alcon in conset.constraints
        alcost(alcon)
    end
end

for method in (:algrad!, :alhess!, :dualupdate!, :penaltyupdate!, 
        :max_penalty, :reset_duals!, :reset_penalties!)
    @eval function $method(conset::ALConstraintSet2, args...)
        for alcon in conset.constraints
            $method(alcon, args...)
        end
    end
end

function add_alcost_expansion!(conset::ALConstraintSet2, E::CostExpansion2)
    for alcon in conset.constraints
        add_alcost_expansion!(alcon)
    end
end

function settraj!(conset::ALConstraintSet2, Z::AbstractTrajectory)
    conset.Zref[] = Z
    for alcon in conset.constraints
        settraj!(alcon, Z)
    end
end

function normviolation!(conset::ALConstraintSet2, p=2)
    isempty(conset) && return 0.0
    conset.c_max .= 0
    for i = 1:length(conset) 
        normviolation!(conset.constraints[i], p, conset.c_max)
    end
    return norm(conset.c_max, p)
end

# Need to duplicate due to allocation
function max_violation(conset::ALConstraintSet2)
    isempty(conset) && return 0.0
    conset.c_max .= 0
    for i = 1:length(conset) 
        normviolation!(conset.constraints[i], Inf, conset.c_max)
    end
    return norm(conset.c_max, Inf)
end

function max_penalty(conset::ALConstraintSet2)
    for i = 1:length(conset) 
        conset.μ_max[i] = max_penalty(conset.constraints[i])
    end
    return maximum(conset.μ_max)
end

function reset!(conset::ALConstraintSet2)
    for con in conset.constraints
        resetparams!(con)
        reset_duals!(con)
        reset_penalties!(con)
    end
end

"""
	findmax_violation(conSet)

Return details on the where the largest violation occurs. Returns a string giving the
constraint type, time step index, and index into the constraint.
"""
function findmax_violation(conSet::ALConstraintSet2{T}) where T
    c_max0 = -Inf
    j_con = -Inf
    for (i,alcon) in enumerate(conSet.constraints) 
        max_violation!(alcon)
        v = maximum(alcon.c_max)
        if v > c_max0
            c_max0 = v
            j_con = i
        end
    end
	# max_violation(conSet)
	# c_max0, j_con = findmax(conSet.c_max) # which constraint
	if c_max0 < eps()
		return "No constraints violated"
	end
	conval = conSet[j_con]
	k_con = argmax(conval.c_max) # which index
    i_con = searchsortedfirst(conval.inds, k_con)
	# k_con = conval.inds[i_con] # time step
	c_max, i_max = findmax(abs,conval.viol[i_con])  # index into constraint
	@assert c_max == c_max0
	con_name = string(typeof(conval.con).name.name)
	return con_name * " at time step $k_con at " * TO.con_label(conval.con, i_max)
end