struct ALConstraintSet{T}
    constraints::Vector{ALConstraint{T}}
    c_max::Vector{T}
    μ_max::Vector{T}
    Zref::Ref{SampledTrajectory}
    Eref::Ref{CostExpansion}
end

function ALConstraintSet{T}() where T
    constraints = ALConstraint{T}[]
    c_max = T[]
    μ_max = T[]
    Z = SampledTrajectory(KnotPoint{Any,Any,Vector{T},T}[]) 
    Zref = Ref{RD.SampledTrajectory}(Z)
    E = CostExpansion{T}(0,0,0)
    Eref = Ref{CostExpansion}(E)

    ALConstraintSet{T}(constraints, c_max, μ_max, Zref, Eref)
end

function initialize!(conset::ALConstraintSet{T}, cons::TO.ConstraintList, 
                     Z::SampledTrajectory, opts::SolverOptions,
                     costs, E=CostExpansion{T}(RD.dims(Z)...)) where T
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
@inline Base.length(conset::ALConstraintSet) = length(conset.constraints)
@inline Base.getindex(conset::ALConstraintSet, i::Integer) = conset.constraints[i]
Base.firstindex(::ALConstraintSet) = 1
Base.lastindex(conset::ALConstraintSet) = length(conset.constraints)

function Base.iterate(conset::ALConstraintSet) 
    isempty(conset.constraints) ? nothing : (conset.constraints[1], 1)
end

function Base.iterate(conset::ALConstraintSet, state::Int) 
    state >= length(conset) ? nothing : (conset.constraints[state+1], state+1)
end

Base.IteratorSize(::ALConstraintSet) = Base.HasLength()
Base.IteratorEltype(::ALConstraintSet) = Base.HasEltype()
Base.eltype(::ALConstraintSet{T}) where T = ALConstraint{T}

# Methods
for method in (:evaluate_constraints!, :constraint_jacobians!) 
    @eval function $method(conset::ALConstraintSet)
        for alcon in conset.constraints
            $method(alcon)
        end
    end
    @eval function $method(conset::ALConstraintSet, Z::SampledTrajectory)
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

function alcost(conset::ALConstraintSet{T}) where T
    for alcon in conset.constraints
        alcost(alcon)
    end
end

for method in (:algrad!, :alhess!, :dualupdate!, :penaltyupdate!, 
        :max_penalty, :reset_duals!, :reset_penalties!)
    @eval function $method(conset::ALConstraintSet, args...)
        for alcon in conset.constraints
            $method(alcon, args...)
        end
    end
end

function add_alcost_expansion!(conset::ALConstraintSet, E::CostExpansion)
    for alcon in conset.constraints
        add_alcost_expansion!(alcon)
    end
end

function settraj!(conset::ALConstraintSet, Z::SampledTrajectory)
    conset.Zref[] = Z
    for alcon in conset.constraints
        settraj!(alcon, Z)
    end
end

function normviolation!(conset::ALConstraintSet, p=2)
    isempty(conset) && return 0.0
    conset.c_max .= 0
    for i = 1:length(conset) 
        normviolation!(conset.constraints[i], p, conset.c_max)
    end
    return norm(conset.c_max, p)
end

# Need to duplicate due to allocation
function max_violation(conset::ALConstraintSet)
    isempty(conset) && return 0.0
    conset.c_max .= 0
    for i = 1:length(conset) 
        normviolation!(conset.constraints[i], Inf, conset.c_max)
    end
    return norm(conset.c_max, Inf)
end

function max_penalty(conset::ALConstraintSet)
    for i = 1:length(conset) 
        conset.μ_max[i] = max_penalty(conset.constraints[i])
    end
    return maximum(conset.μ_max)
end

function reset!(conset::ALConstraintSet)
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
function findmax_violation(conSet::ALConstraintSet{T}) where T
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
	if c_max0 < eps()
		return "No constraints violated"
	end
	conval = conSet[j_con]
	k_con = argmax(conval.c_max) # which index
    i_con = searchsortedfirst(conval.inds, k_con)
	c_max, i_max = findmax(abs,conval.viol[i_con])  # index into constraint
	@assert c_max == c_max0
	con_name = string(typeof(conval.con).name.name)
	return con_name * " at time step $k_con at " * TO.con_label(conval.con, i_max)
end