
# Infeasible methods
function InfeasibleProblem(prob::Problem{RK}, Z0::SampledTrajectory, R_inf::Real) where RK
    @assert !isnan(sum(sum.(states(Z0))))

    n,m,N = dims(prob)  # original sizes

    # Create model with augmented controls
    model_inf = InfeasibleModel(prob.model)

    # Get a trajectory that is dynamically feasible for the augmented problem
    #   and matches the states and controls of the original guess
    Z = infeasible_trajectory(model_inf, Z0)

    # Convert constraints so that they accept new dimensions
    conSet = TO.change_dimension(get_constraints(prob), n, m+n, 1:n, 1:m)

    # Constrain additional controls to be zero
    inf = InfeasibleConstraint(model_inf)
    TO.add_constraint!(conSet, inf, 1:N-1)

    # Infeasible Objective
    obj = infeasible_objective(prob.obj, R_inf)

    # Create new problem
    Problem(model_inf, obj, conSet, prob.x0, prob.xf, Z, N, prob.t0, prob.tf)
end

function infeasible_objective(obj::Objective, regularizer)
    n,m = TO.state_dim(obj.cost[1]), TO.control_dim(obj.cost[1])
    Rd = [@SVector zeros(m); @SVector fill(regularizer,n)]
    R = Diagonal(Rd)
    cost_inf = TO.DiagonalCost(Diagonal(@SVector zeros(n)), R, checks=false)
    costs = map(obj.cost) do cost
        cost_idx = TO.change_dimension(cost, n, n+m, 1:n, 1:m)
        cost_idx + cost_inf
    end
    TO.Objective(costs)
end


############################################################################################
#                               INFEASIBLE MODELS                                          #
############################################################################################
RD.@autodiff struct InfeasibleModel{Nx,Nu,D<:DiscreteDynamics} <: DiscreteDynamics 
    model::D
    _u::SVector{Nu,Int}  # inds to original controls
    _ui::SVector{Nx,Int} # inds to infeasible controls
    function InfeasibleModel{Nx,Nu}(model::D) where {Nx,Nu,D<:DiscreteDynamics}
        _u = SVector{Nu}(1:Nu)
        _ui = SVector{Nx}((1:Nx) .+ Nu)
        new{Nx,Nu,D}(model, _u, _ui)
    end
end

""" $(TYPEDEF)
An infeasible model is an augmented dynamics model that makes the system artifically fully
actuated by augmenting the control vector with `n` additional controls. The dynamics are
handled explicitly in discrete time:

``x_{k+1} = f(x_k,u_k,dt) + w_k``

where ``w_k`` are the additional `n`-dimensional controls. In practice, these are constrained
to be zero by the end of the solve.

# Constructors
```julia
InfeasibleModel(model::AbstractModel)
```
"""
InfeasibleModel

function InfeasibleModel(model::AbstractModel)
    n,m = RD.dims(model)
    InfeasibleModel{n,m}(model)
end

RD.statevectortype(::Type{<:InfeasibleModel{<:Any,<:Any,D}}) where D = RD.statevectortype(D)
RD.LieState(model::InfeasibleModel) = RD.LieState(model.model)
@inline RD.rotation_type(model::InfeasibleModel) where D = rotation_type(model.model)

# Generic Infeasible Methods
@inline RD.state_dim(model::InfeasibleModel{n}) where n = n
@inline RD.control_dim(model::InfeasibleModel{n,m}) where {n,m} = n+m
@inline RD.errstate_dim(model::InfeasibleModel) = RD.errstate_dim(model.model)

function RD.discrete_dynamics(model::InfeasibleModel,
        x, u, t, dt)
    u0 = u[model._u]
    ui = u[model._ui]
    RobotDynamics.discrete_dynamics(model.model, x, u0, t, dt) + ui
end

function RD.discrete_dynamics!(model::InfeasibleModel{Nx,Nu}, xn,
        x, u, t, dt) where {Nx,Nu}
    u0 = view(u, 1:Nu)
    ui = view(u, Nu+1:Nx+Nu)
    RobotDynamics.discrete_dynamics!(model.model, xn, x, u0, t, dt)
    xn .+= ui
    return
end

@inline RD.state_diff(model::InfeasibleModel, x::SVector, x0::SVector) = 
    RD.state_diff(model.model, x, x0)

@inline RobotDynamics.state_diff_jacobian!(G, model::InfeasibleModel, Z::SampledTrajectory) =
	RobotDynamics.state_diff_jacobian!(G, model.model, Z)

@inline RobotDynamics.∇²differential!(∇G, model::InfeasibleModel, x::SVector, dx::SVector) = 
    ∇²differential!(∇G, model.model, x, dx)

Base.position(model::InfeasibleModel, x::SVector) = position(model.model, x)

RobotDynamics.orientation(model::InfeasibleModel, x::SVector) = orientation(model.model, x)

"Calculate a dynamically feasible initial trajectory for an infeasible problem, given a
desired trajectory"
function infeasible_trajectory(model::InfeasibleModel{n,m}, Z0::SampledTrajectory) where {T,n,m}
    x,u = zeros(model)
    ui = @SVector zeros(n)
    Z = [KnotPoint(state(z), [control(z); ui], z.t, z.dt) for z in Z0]
    N = length(Z0)
    for k = 1:N-1
        RD.propagate_dynamics!(RD.default_signature(model), model, Z[k+1], Z[k])
        x′ = state(Z[k+1])
        u_slack = state(Z0[k+1]) - x′
        u = [control(Z0[k]); u_slack]
        RD.setcontrol!(Z[k], u)
        RD.setstate!(Z[k+1], x′ + u_slack)
    end
    return SampledTrajectory(Z)
end


############################################################################################
#  								INFEASIBLE CONSTRAINT 									   #
############################################################################################
""" $(TYPEDEF) Constraints additional ``infeasible'' controls to be zero.
Constructors: ```julia
InfeasibleConstraint(model::InfeasibleModel)
InfeasibleConstraint(n,m)
```
"""
struct InfeasibleConstraint{Nx,Nu} <: TO.ControlConstraint
	ui::SVector{Nx,Int}
	function InfeasibleConstraint{Nx,Nu}() where {Nx,Nu}
		ui = SVector{Nx}((1:Nx) .+ Nu)
		new{Nx,Nu}(ui)
	end
end

InfeasibleConstraint(model::InfeasibleModel{n,m}) where {n,m} = InfeasibleConstraint{n,m}()
RobotDynamics.state_dim(con::InfeasibleConstraint{n,m}) where {n,m} = n
RobotDynamics.control_dim(con::InfeasibleConstraint{n,m}) where {n,m} = n+m 
@inline TO.sense(::InfeasibleConstraint) = TO.Equality()
@inline RD.output_dim(::InfeasibleConstraint{n}) where n = n

RD.evaluate(con::InfeasibleConstraint, x, u) = u[con.ui] # infeasible controls
function RD.evaluate!(con::InfeasibleConstraint, c, x, u)
    for (i,j) in enumerate(con.ui)
        c[i] = u[j]
    end
    return
end

function RD.jacobian!(con::InfeasibleConstraint{Nx}, ∇c, c, x, u) where {Nx}
	for (i,j) in enumerate(con.ui)
		∇c[i,Nx+j] = 1
    end
    return
end
