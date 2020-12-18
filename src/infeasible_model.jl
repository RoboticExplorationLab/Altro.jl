
############################################################################################
#                               INFEASIBLE MODELS                                          #
############################################################################################
struct Infeasible{N,M,D<:AbstractModel} <: AbstractModel
    model::D
    _u::SVector{M,Int}  # inds to original controls
    _ui::SVector{N,Int} # inds to infeasible controls
end

struct InfeasibleLie{N,M,D<:AbstractModel} <: RobotDynamics.LieGroupModel 
    model::D
    _u::SVector{M,Int}  # inds to original controls
    _ui::SVector{N,Int} # inds to infeasible controls
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
const InfeasibleModel{N,M,D} = Union{Infeasible{N,M,D},InfeasibleLie{N,M,D}} where {N,M,D}

function InfeasibleModel(model::AbstractModel)
    n,m = size(model)
    _u  = SVector{m}(1:m)
    _ui = SVector{n}((1:n) .+ m)
    Infeasible(model, _u, _ui)
end

function InfeasibleModel(model::RobotDynamics.LieGroupModel)
    n,m = size(model)
    _u  = SVector{m}(1:m)
    _ui = SVector{n}((1:n) .+ m)
    InfeasibleLie(model, _u, _ui)
end

RobotDynamics.LieState(model::InfeasibleLie) = RobotDynamics.LieState(model.model)

# Generic Infeasible Methods
function Base.size(model::InfeasibleModel)
    n,m = size(model.model)
    return n, n+m
end

RobotDynamics.dynamics(::InfeasibleModel, x, u) =
    throw(ErrorException("Cannot evaluate continuous dynamics on an infeasible model"))

@generated function RobotDynamics.discrete_dynamics(::Type{Q}, model::InfeasibleModel{N,M},
        z::AbstractKnotPoint{T,N}) where {T,N,M,Q<:Explicit}
    _u = SVector{M}((1:M) .+ N)
    _ui = SVector{N}((1:N) .+ (N+M))
    quote
        x = state(z)
        dt = z.dt
        u0 = z.z[$_u]
        ui = z.z[$_ui]
        RobotDynamics.discrete_dynamics($Q, model.model, x, u0, z.t, dt) + ui
    end
end

@inline RobotDynamics.rotation_type(model::InfeasibleModel) where D = rotation_type(model.model)

@generated function RobotDynamics.discrete_jacobian!(::Type{Q}, ∇f, model::InfeasibleModel{N,M},
        z::AbstractKnotPoint{T,N}) where {T,N,M,Q<:Explicit}

    ∇ui = [(@SMatrix zeros(N,N+M)) Diagonal(@SVector ones(N)) @SVector zeros(N)]
    _x = SVector{N}(1:N)
    _u = SVector{M}((1:M) .+ N)
    _z = SVector{N+M}(1:N+M)
    _ui = SVector{N}((1:N) .+ (N+M))
    zi = [:(z.z[$i]) for i = 1:N+M]
    NM1 = N+M+1
	NM = N+M
    ∇u0 = @SMatrix zeros(N,N)

    quote
        # Build KnotPoint for original model
        s0 = SVector{$NM1}($(zi...), z.dt)

        u0 = z.z[$_u]
        ui = z.z[$_ui]
		z_ = StaticKnotPoint(z.z[$_z], $_x, $_u, z.dt, z.t)
		∇f_ = uview(∇f, 1:N, 1:$NM)
        discrete_jacobian!($Q, ∇f_, model.model, z_)
		# ∇f[$_x, N+NM] .= ∇f_[$_x, N+M] # ∇dt
		∇f[$_x, $_ui] .= Diagonal(@SVector ones(N))
		return
		# ∇f[$_x,$_ui]
        # [∇f[$_x, $_z] $∇u0 ∇dt] + $∇ui
    end
end

function RobotDynamics.state_diff(model::InfeasibleModel, x::SVector, x0::SVector)
	RobotDynamics.state_diff(model.model, x, x0)
end

function RobotDynamics.state_diff_jacobian!(G, model::InfeasibleModel, Z::Traj)
	RobotDynamics.state_diff_jacobian!(G, model.model, Z)
end

function RobotDynamics.∇²differential!(∇G, model::InfeasibleModel, x::SVector, dx::SVector)
	return ∇²differential!(∇G, model.model, x, dx)
end

RobotDynamics.state_diff_size(model::InfeasibleModel) = RobotDynamics.state_diff_size(model.model)

Base.position(model::InfeasibleModel, x::SVector) = position(model.model, x)

RobotDynamics.orientation(model::InfeasibleModel, x::SVector) = orientation(model.model, x)

"Calculate a dynamically feasible initial trajectory for an infeasible problem, given a
desired trajectory"
function infeasible_trajectory(model::InfeasibleModel{n,m}, Z0::Traj) where {T,n,m}
    x,u = zeros(model)
    ui = @SVector zeros(n)
    Z = [KnotPoint(state(z), [control(z); ui], z.dt, z.t) for z in Z0]
    N = length(Z0)
    for k = 1:N-1
        RobotDynamics.propagate_dynamics(RobotDynamics.RK4, model, Z[k+1], Z[k])
        x′ = state(Z[k+1])
        u_slack = state(Z0[k+1]) - x′
        u = [control(Z0[k]); u_slack]
        RobotDynamics.set_control!(Z[k], u)
        RobotDynamics.set_state!(Z[k+1], x′ + u_slack)
    end
    return Traj(Z)
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
struct InfeasibleConstraint{n} <: TO.ControlConstraint
	ui::SVector{n,Int}
	m::Int
	function InfeasibleConstraint(n::Int, m::Int)
		ui = SVector{n}((1:n) .+ m)
		new{n}(ui, m)
	end
end

InfeasibleConstraint(model::InfeasibleModel{n,m}) where {n,m} = InfeasibleConstraint(n,m)
RobotDynamics.control_dim(con::InfeasibleConstraint{n}) where n = n + con.m
@inline TO.sense(::InfeasibleConstraint) = TO.Equality()
@inline Base.length(::InfeasibleConstraint{n}) where n = n

function TO.evaluate(con::InfeasibleConstraint, u::SVector)
    ui = u[con.ui] # infeasible controls
end

function TO.jacobian!(∇c, con::InfeasibleConstraint{n}, u::SVector) where n
	for (i,j) in enumerate(con.ui)
		∇c[i,j] = 1
    end
    return true
end
