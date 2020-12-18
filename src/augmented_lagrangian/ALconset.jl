
"""
	ALConstraintSet{T}

An [`AbstractConstraintSet`](@ref) that stores the constraint values as well as Lagrange
multiplier and penalty terms for each constraint.

The cost associated with constraint terms in the augmented Lagrangian can be evaluated for

	cost!(J::Vector, ::ALConstraintSet)

which adds the cost at each time step to the vector `J` of length `N`.

The cost expansion for these terms is evaluated along the trajectory `Z` using

	cost_expansion!(E::Objective, conSet::ALConstraintSet, Z)

which also adds the expansion terms to the terms in `E`.

The penalty and multiplier terms can be updated using

	penalty_update!(::ALConstraintSet)
	dual_update!(::ALConstraintSet)

The current set of active constraint (with tolerance `tol`) can be re-calculated using

	update_active_set!(::ALConstraintSet, ::Val{tol})

The maximum penalty can be queried using `max_penalty(::ALConstraintSet)`, and the
penalties and/or multipliers can be reset using

	reset!(::ALConstraintSet)
	reset_penalties!(::ALConstraintSet)
	reset_duals!(::ALConstraintSet)

# Constructor
	ALConstraintSet(::ConstraintList, ::AbstractModel)
	ALConstraintSet(::Problem)
"""
struct ALConstraintSet{T} <: TO.AbstractConstraintSet
    convals::Vector{ALConVal}
    errvals::Vector{ALConVal}          # TODO: is this needed?
	# ∇c_proj::Vector{<:Vector}        # Jacobians of projected constraints
    # λ::Vector{<:Vector}
    # μ::Vector{<:Vector}
	# active::Vector{<:Vector}
    c_max::Vector{T}
    μ_max::Vector{T}
    μ_maxes::Vector{Vector{T}}
	# params::Vector{TO.ConstraintParams{T}}
	p::Vector{Int}
end

function ALConstraintSet(cons::TO.ConstraintList, model::RD.AbstractModel)
    n,m = cons.n, cons.m
    n̄ = RobotDynamics.state_diff_size(model)
    ncon = length(cons)
    useG = model isa RD.LieGroupModel
	errvals = map(1:ncon) do i
        C,c = TO.gen_convals(n̄, m, cons[i], cons.inds[i])
        ALConVal(n̄, m, cons[i], cons.inds[i], C, c, useG)
    end
    convals = map(errvals) do errval
        ALConVal(n, m, errval)
    end
	errvals = convert(Vector{ALConVal}, errvals)
	convals = convert(Vector{ALConVal}, convals)
	# ∇c_proj = map(1:ncon) do i
	# 	[copy(errvals[i].jac[j,1]) for j in eachindex(cons.inds[i])]
	# end
    # λ = map(1:ncon) do i
	# 	p = length(cons[i])
	# 	convals[i].λ
    #     # [@SVector zeros(p) for j in cons.inds[i]]
    # end
    # μ = map(1:ncon) do i
	# 	p = length(cons[i])
	# 	convals[i].μ
    #     # [@SVector ones(p) for j in cons.inds[i]]
    # end
    # a = map(1:ncon) do i
    #     p = length(cons[i])
    #     [@SVector ones(Bool,p) for j in cons.inds[i]]
	# end
	# if ncon == 0
	# 	∇c_proj = Vector{Float64}[]
	# 	λ = Vector{Float64}[]
	# 	μ = Vector{Float64}[]
	# 	a = Vector{Float64}[]
	# end
    c_max = zeros(ncon)
    μ_max = zeros(ncon)
    μ_maxes = [zeros(length(ind)) for ind in cons.inds]
	# params = [TO.ConstraintParams() for con in cons.constraints]
    # ALConstraintSet(convals, errvals, ∇c_proj, λ, μ, a, c_max, μ_max, μ_maxes, params, copy(cons.p))
    ALConstraintSet(convals, errvals, c_max, μ_max, μ_maxes, copy(cons.p))
end

@inline ALConstraintSet(prob::Problem) = ALConstraintSet(prob.constraints, prob.model)

@inline TO.get_convals(conSet::ALConstraintSet) = conSet.convals
@inline TO.get_errvals(conSet::ALConstraintSet) = conSet.errvals

get_duals(conSet::ALConstraintSet) = [cval.λ for cval in conSet.convals]
function set_duals!(conSet::ALConstraintSet, λ)
	for i = 1:length(conSet)
		set_duals!(conSet.convals[i], λ[i])
	end
end
function set_duals!(cval::ALConVal, λ)
	for i = 1:length(cval.λ)
		cval.λ[i] .= λ[i]
	end
end

for method in (:violation!, :∇violation!, :dual_update!, :penalty_update!, :reset_duals!, :reset_penalties!)
	@eval function $method(conSet::ALConstraintSet)
		for i = 1:length(conSet)
			$method(conSet.convals[i])
		end
	end
end

############################################################################################
#                             Augmented Lagrangian Updates
############################################################################################

# function dual_update!(conSet::ALConstraintSet)
#     for i in eachindex(conSet.λ)
#         dual_update!(conSet.convals[i])
# 	end
# end

function dual_update!(conval::ALConVal)
	c = conval.vals
	λ = conval.λ
	μ = conval.μ
	λ_max = conval.params.λ_max
	cone = TO.sense(conval.con)
	# λ_min = TO.sense(conval.con) == Equality() ? -λ_max : zero(λ_max)
	for i in eachindex(conval.inds)
		λ[i] .= dual_update(cone, SVector(λ[i]), SVector(c[i]), SVector(μ[i]), λ_max) 
	end
end

function dual_update(::Equality, λ, c, μ, λmax)
	λbar = λ + μ .* c
	return clamp.(λbar, -λmax, λmax)
end

function dual_update(::Inequality, λ, c, μ, λmax)
 	λbar = λ + μ .* c
	return clamp.(λbar, 0, λmax)  # project onto the dual cone via max(0,x)
end

function dual_update(cone::SecondOrderCone, λ, c, μ, λmax)
	 λbar = λ - μ .* c
	 return TO.projection(cone, λbar)  # project onto the dual cone
end

# function penalty_update!(conSet::ALConstraintSet)
# 	for i in eachindex(conSet.μ)
# 		penalty_update!(conSet.convals[i])
# 	end
# end

function penalty_update!(cval::ALConVal)
	μ = cval.μ
	ϕ = cval.params.ϕ
	μ_max = cval.params.μ_max
	for i = 1:length(μ) 
		μ[i] .*= ϕ
		clamp!(μ[i], 0, μ_max)
	end
end

# Active Set
function update_active_set!(conSet::ALConstraintSet, val::Val{tol}=Val(0.0)) where tol
	for i in 1:length(conSet) 
		# update_active_set!(conSet.active[i], conSet.λ[i], conSet.convals[i], val)
		update_active_set!(conSet.convals[i], val)
	end
end

function update_active_set!(conval::ALConVal, ::Val{tol}) where tol
	a = conval.active
	λ = conval.λ
	if TO.sense(conval.con) == TO.Inequality()
		for i in eachindex(a)
			a[i] = @. (conval.vals[i] >= -tol) | (λ[i] > zero(tol))
		end
	end
end

"""
	max_penalty(conSet::ALConstraintSet)

Calculate the maximum constrained penalty across all constraints.
"""
function max_penalty(conSet::ALConstraintSet)
	max_penalty!(conSet)
	maximum(conSet.μ_max)
end

function max_penalty!(conSet::ALConstraintSet{T}) where T
    conSet.c_max .*= 0
	for i in 1:length(conSet) 
		maxes = conSet.μ_maxes[i]::Vector{T}
        max_penalty!(maxes, conSet.convals[i])
        conSet.μ_max[i] = maximum(maxes)
    end
end

function max_penalty!(μ_max::Vector{<:Real}, cval::ALConVal)
    for i in eachindex(cval.μ)
        μ_max[i] = maximum(cval.μ[i])
    end
    return nothing
end

############################################################################################
#                             Constraint Penalties
############################################################################################
# function projection_jacobians!(conSet::ALConstraintSet)
# 	for i in eachindex(conSet.convals)
# 		projection_jacobians!(conSet.convals[i])
# 	end
# end


# """
# 	constraint_penalty!(conSet::ALConstraintSet)

# Evaluate the penalty of the constraints. For equality constraints, this is the constraint 
# itself. For generalized inequalities, this is the projection of the constraint onto the
# appropriate cone.
# """
# function constraint_penalty!(conSet::ALConstraintSet)
# 	for i in eachindex(conSet.convals)
# 		constraint_penalty!(conSet.convals[i], conSet.λ[i])
# 	end
# end

# function constraint_penalty!(conval::TO.ConVal, λ)
# 	for i in eachindex(conval.inds)
# 		conval.vals2[i] = penalty(TO.sense(conval.con), conval.vals[i], λ[i])
# 	end
# end



# """
# 	penalty_jacobian!(conSet::ALConstraintSet)

# Evaluate the Jacobian of the constraint penalty. For equality constraints, this is simply
# the Jacobian of the constraint itself. For generalized inequalities, this is `∇proj*∇c`
# where `∇proj` is the Jacobian of the projection of the constraint onto the corresponding 
# cone.
# """
# function penalty_jacobian!(conSet::ALConstraintSet)
# 	for i in eachindex(conSet.errvals)
# 		penalty_jacobian!(conSet.∇c_proj[i], conSet.errvals[i], conSet.λ[i])
# 	end
# end

# function penalty_jacobian!(∇c_proj::Vector, conval::TO.ConVal, λ::Vector)
# 	if size(conval.jac, 2) > 1
# 		throw(ErrorException("Constraint projection not supported for CoupledConstraints"))
# 	end
# 	for i in eachindex(conval.inds)
# 		∇penalty!(TO.sense(conval.con), ∇c_proj[i], conval.jac[i], conval.vals[i], λ[i])
# 	end
# end

############################################################################################
#                                        Cost
############################################################################################

# function TO.cost!(J::Vector{<:Real}, conSet::ALConstraintSet)
# 	for i in eachindex(conSet.convals)
# 		TO.cost!(J, conSet.convals[i], conSet.λ[i], conSet.μ[i], conSet.active[i])
# 	end
# end

# function TO.cost!(J::Vector{<:Real}, conval::ALConVal, λ::Vector{<:StaticVector},
# 		μ::Vector{<:StaticVector}, a::Vector{<:StaticVector})
# 	for (i,k) in enumerate(conval.inds)
# 		c = SVector(conval.vals[i])
# 		Iμ = Diagonal(SVector(μ[i] .* a[i]))
# 		J[k] += λ[i]'c .+ 0.5*c'Iμ*c
# 	end
# end

# function TO.cost_expansion!(E::Objective, conSet::ALConstraintSet, Z::AbstractTrajectory,
# 		init::Bool=false, rezero::Bool=false)
# 	for i in eachindex(conSet.errvals)
# 		TO.cost_expansion!(E, conSet.convals[i], conSet.λ[i], conSet.μ[i], conSet.active[i])
# 	end
# end

# @generated function TO.cost_expansion!(E::QuadraticObjective{n,m}, conval::ALConVal{C}, λ, μ, a) where {n,m,C}
# 	if C <: TO.StateConstraint
# 		expansion = quote
# 			cx = ∇c
# 			E[k].Q .+= cx'Iμ*cx
# 			E[k].q .+= cx'g
# 		end
# 	elseif C <: TO.ControlConstraint
# 		expansion = quote
# 			cu = ∇c
# 			E[k].R .+= cu'Iμ*cu
# 			E[k].r .+= cu'g
# 		end
# 	elseif C<: TO.StageConstraint
# 		ix = SVector{n}(1:n)
# 		iu = SVector{m}(n .+ (1:m))
# 		expansion = quote
# 			cx = ∇c[:,$ix]
# 			cu = ∇c[:,$iu]
# 			E[k].Q .+= cx'Iμ*cx
# 			E[k].q .+= cx'g
# 			E[k].H .+= cu'Iμ*cx
# 			E[k].R .+= cu'Iμ*cu
# 			E[k].r .+= cu'g
# 		end
# 	else
# 		throw(ArgumentError("cost expansion not supported for CoupledConstraints"))
# 	end
# 	quote
# 		for (i,k) in enumerate(conval.inds)
# 			∇c = SMatrix(conval.jac[i])
# 			c = conval.vals[i]
# 			Iμ = Diagonal(a[i] .* μ[i])
# 			g = Iμ*c .+ λ[i]

# 			$expansion
# 		end
# 	end
# end

############################################################################################
#                                  RESET
############################################################################################

function reset!(conSet::ALConstraintSet)
    reset_duals!(conSet)
    reset_penalties!(conSet)
end

# function reset_duals!(conSet::ALConstraintSet)
# 	for i = 1:length(conSet)
# 		reset_duals!(conSet.convals[i])
# 	end
# end

# function reset_penalties!(conSet::ALConstraintSet)
# 	for i = 1:length(conSet)
# 		reset_penalties!(conSet.convals[i])
# 	end
# end


"""
	link_constraints!(set1, set2)

Link any common constraints between `set1` and `set2` by setting elements in `set1` to point
to elements in `set2`
"""
function link_constraints!(set1::ALConstraintSet, set2::ALConstraintSet)
	# Find common constraints
	links = Tuple{Int,Int}[]
	for (i,con1) in enumerate(set1)
		for (j,con2) in enumerate(set2)
			if con1 === con2
				push!(links, (i,j))
			end
		end
	end

	# Link values
	for (i,j) in links
		set1.convals[i] = set2.convals[j]
		set1.errvals[i] = set2.errvals[j]
		# set1.active[i] = set2.active[j]
		# set1.λ[i] = set2.λ[j]
		# set1.μ[i] = set2.μ[j]
	end
	return links
end

function shift_fill!(conSet::ALConstraintSet, n=1)
	for i = 1:length(conSet)
		shift_fill!(conSet.convals[i], n)
	end
end

############################################################################################
#                                    Solver Options
############################################################################################
function reset!(conSet::ALConstraintSet{T}, opts::SolverOptions{T}) where T
    # if !isnan(opts.dual_max)
    #     for params in conSet.params
    #         params.λ_max = opts.dual_max
    #     end
    # end
    # if !isnan(opts.penalty_max)
    #     for params in conSet.params
    #         params.μ_max = opts.penalty_max
    #     end
    # end
    # if !isnan(opts.penalty_initial)
    #     for params in conSet.params
    #         params.μ0 = opts.penalty_initial
    #     end
    # end
    # if !isnan(opts.penalty_scaling)
    #     for params in conSet.params
    #         params.ϕ = opts.penalty_scaling
    #     end
	# end
	for i = 1:length(conSet)
		set_params!(conSet.convals[i], opts)
	end
    if opts.reset_duals
        reset_duals!(conSet)
    end
    if opts.reset_penalties
        reset_penalties!(conSet)
    end
end