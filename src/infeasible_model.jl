
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

# struct InfeasibleLie{N,M,D<:AbstractModel} <: RobotDynamics.LieGroupModel 
#     model::D
#     _u::SVector{M,Int}  # inds to original controls
#     _ui::SVector{N,Int} # inds to infeasible controls
# end

# const InfeasibleModel{N,M,D} = Union{Infeasible{N,M,D},InfeasibleLie{N,M,D}} where {N,M,D}

function InfeasibleModel(model::AbstractModel)
    n,m = size(model)
    InfeasibleModel{n,m}(model)
end

# function InfeasibleModel(model::RobotDynamics.LieGroupModel)
#     n,m = size(model)
#     _u  = SVector{m}(1:m)
#     _ui = SVector{n}((1:n) .+ m)
#     InfeasibleLie(model, _u, _ui)
# end

RD.statevectortype(::Type{<:InfeasibleModel{<:Any,<:Any,D}}) where D = RD.statevectortype(D)
RD.LieState(model::InfeasibleModel) = RD.LieState(model.model)
@inline RD.rotation_type(model::InfeasibleModel) where D = rotation_type(model.model)

# Generic Infeasible Methods
@inline RD.state_dim(model::InfeasibleModel{n}) where n = n
@inline RD.control_dim(model::InfeasibleModel{n,m}) where {n,m} = n+m
@inline RD.errstate_dim(model::InfeasibleModel) = RD.errstate_dim(model.model)

# RD.dynamics(::InfeasibleModel, x, u) =
#     throw(ErrorException("Cannot evaluate continuous dynamics on an infeasible model"))

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

# @generated function RobotDynamics.discrete_jacobian!(::Type{Q}, ∇f, model::InfeasibleModel{N,M},
#         z::AbstractKnotPoint{T,N}, cache=nothing) where {T,N,M,Q<:Explicit}

#     ∇ui = [(@SMatrix zeros(N,N+M)) Diagonal(@SVector ones(N)) @SVector zeros(N)]
#     _x = SVector{N}(1:N)
#     _u = SVector{M}((1:M) .+ N)
#     _z = SVector{N+M}(1:N+M)
#     _ui = SVector{N}((1:N) .+ (N+M))
#     zi = [:(z.z[$i]) for i = 1:N+M]
#     NM1 = N+M+1
# 	NM = N+M
#     ∇u0 = @SMatrix zeros(N,N)

#     quote
#         # Build KnotPoint for original model
#         s0 = SVector{$NM1}($(zi...), z.dt)

#         u0 = z.z[$_u]
#         ui = z.z[$_ui]
# 		z_ = StaticKnotPoint(z.z[$_z], $_x, $_u, z.dt, z.t)
# 		∇f_ = uview(∇f, 1:N, 1:$NM)
#         discrete_jacobian!($Q, ∇f_, model.model, z_)
# 		# ∇f[$_x, N+NM] .= ∇f_[$_x, N+M] # ∇dt
# 		∇f[$_x, $_ui] .= Diagonal(@SVector ones(N))
# 		return
# 		# ∇f[$_x,$_ui]
#         # [∇f[$_x, $_z] $∇u0 ∇dt] + $∇ui
#     end
# end
# function RD._discrete_jacobian!(::RD.ForwardAD, ::Type{Q}, ∇f, model::InfeasibleModel{N,M},
#         z::AbstractKnotPoint{T,N}, cache=nothing) where {T,N,M,Q<:Explicit}
#     RD.discrete_jacobian!(Q, ∇f, model, z, cache)
# end


@inline RD.state_diff(model::InfeasibleModel, x::SVector, x0::SVector) = 
    RD.state_diff(model.model, x, x0)

@inline RobotDynamics.state_diff_jacobian!(G, model::InfeasibleModel, Z::Traj) =
	RobotDynamics.state_diff_jacobian!(G, model.model, Z)

@inline RobotDynamics.∇²differential!(∇G, model::InfeasibleModel, x::SVector, dx::SVector) = 
    ∇²differential!(∇G, model.model, x, dx)

Base.position(model::InfeasibleModel, x::SVector) = position(model.model, x)

RobotDynamics.orientation(model::InfeasibleModel, x::SVector) = orientation(model.model, x)

"Calculate a dynamically feasible initial trajectory for an infeasible problem, given a
desired trajectory"
function infeasible_trajectory(model::InfeasibleModel{n,m}, Z0::Traj) where {T,n,m}
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
