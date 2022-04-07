mutable struct DynamicRegularization{T}
    ρ::T
    dρ::T
end

struct iLQRSolver{L,O,Nx,Ne,Nu,T,V} <: UnconstrainedSolver{T}
    model::L
    obj::O

    x0::Vector{T}
    tf::T
    N::Int

    opts::SolverOptions{T}
    stats::SolverStats{T}

    Z::SampledTrajectory{Nx,Nu,T,KnotPoint{Nx,Nu,V,T}}
    Z̄::SampledTrajectory{Nx,Nu,T,KnotPoint{Nx,Nu,V,T}}
    dx::Vector{Vector{T}}
    du::Vector{Vector{T}}

    gains::Vector{Matrix{T}}  # N-1 × (Nu,Ne+1)
    K::Vector{SubArray{T,2,Matrix{T},Tuple{ColonSlice, UnitRange{Int}}, true}}  # N-1 × (Nu,Ne)
    d::Vector{SubArray{T,1,Matrix{T},Tuple{ColonSlice, Int}, true}}             # N-1 × (Nu,)

    D::Vector{DynamicsExpansion{T}}
    G::Vector{Matrix{T}}  # N × (Nx,Ne) 

    Efull::CostExpansion{T}  # Cost expansion (full state)
    Eerr::CostExpansion{T}   # Cost expansion (error state)
    Q::Vector{StateControlExpansion{T}}      # Action-value expansion
    S::Vector{StateControlExpansion{T}}      # Action-value expansion
    ΔV::Vector{T}

    Qtmp::StateControlExpansion{T}
    Quu_reg::Matrix{T}
    Qux_reg::Matrix{T}
    reg::DynamicRegularization{T}

    grad::Vector{T}
    xdot::Vector{T}

    logger::SolverLogging.Logger
end

function iLQRSolver(
        prob::Problem{T}, 
        opts::SolverOptions=SolverOptions{T}(), 
        stats::SolverStats=SolverStats{T}(parent=solvername(iLQRSolver));
        use_static::Val{USE_STATIC}=usestaticdefault(get_model(prob)[1]),
        kwarg_opts...
    ) where {T, USE_STATIC}
    set_options!(opts; kwarg_opts...)

    nx,nu,N = dims(prob)
    ne = RD.errstate_dim.(prob.model)
    push!(ne,ne[end])

    x0 = copy(prob.x0)
    samestatedim = all(x->x == nx[1], nx)
    samecontroldim = all(u->u == nu[1], nu)
    if USE_STATIC
        @assert  samestatedim && samecontroldim "Cannot use StaticArrays with varying state or control dimension."
        Nx = nx[1]
        Nu = nu[1]
        Ne = ne[1]
    else
        Nx = samestatedim ? nx[1] : Any
        Ne = samestatedim ? ne[1] : Any
        Nu = samecontroldim ? nu[1] : Any
    end
    V = USE_STATIC ? SVector{nx[1]+nu[1],T} : Vector{T}
    Z = SampledTrajectory(map(enumerate(prob.Z)) do (k,z)
        RD.KnotPoint{Nx,Nu}(nx[k], nu[k], V(RD.getdata(z)), RD.getparams(z)...)
    end)
    Z̄ = copy(Z)

    # Rollout out dynamics if initial state contains NaN values
    if any(isnan, state(Z[1]))
        RD.rollout!(dynamics_signature(Z), prob.model[1], Z, prob.x0)
    end
    RD.setstate!(Z[1], prob.x0)  # set initial state

    dx = [zeros(T,ne[k]) for k = 1:N]
    du = [zeros(T,nu[k]) for k = 1:N-1]

    gains = [zeros(T,nu[k],ne[k]+1) for k = 1:N-1]
    K = [view(gain,:,1:ne[k]) for (k,gain) in enumerate(gains)]
    d = [view(gain,:,ne[k]+1) for (k,gain) in enumerate(gains)]

    D = [DynamicsExpansion{T}(nx[k],ne[k],nu[k]) for k = 1:N-1]
    G = [Matrix(one(T)*I,nx[k],ne[k]) for k = 1:N]
    push!(G, copy(G[end]))

    Eerr = CostExpansion{T}(ne,nu)
    Efull = FullStateExpansion(Eerr, prob.model[1])
    Q = [StateControlExpansion{T}(ne[k],nu[k]) for k = 1:N] 
    S = [StateControlExpansion{T}(ne[k]) for k = 1:N] 
    ΔV = zeros(T,2)

    # TODO: Change this to accept multiple sizes
    Qtmp = StateControlExpansion{T}(ne[1],nu[1])
    Quu_reg = zeros(T,nu[1],nu[1])  
    Qux_reg = zeros(T,nu[1],ne[1])
    reg = DynamicRegularization{T}(opts.bp_reg_initial, 0)

    grad = zeros(T,N-1)
    xdot = zeros(T,nx[1])  # TODO: change this to support multiple sizes

    # logger = SolverLogging_v1.default_logger(opts.verbose >= 2)
    lg = SolverLogging.Logger()
    setentry(lg, "iter", Int, width=6)
    setentry(lg, "AL iter", Int, width=8, level=2)
    setentry(lg, "iLQR iter", Int, width=10, level=2)
    setentry(lg, "cost", fmt="%.3f")
    setentry(lg, "||v||")
    setentry(lg, "expected", level=2)
    setentry(lg, "dJ", level=2)
    setentry(lg, "grad", level=2)
    setentry(lg, "z", level=3, fmt="%.2f")
    setentry(lg, "α", level=3)
    setentry(lg, "ρ", level=3)
    setentry(lg, "μ_max", level=4)
    setentry(lg, "dJ_zero", Int, level=4)
    setentry(lg, "ls_iter", Int, width=8, level=5)
    setentry(lg, "info", String, width=40)

	L = typeof(prob.model)
	O = typeof(prob.obj)
    solver = iLQRSolver{L,O,Nx,Ne,Nu,T,V}(
        prob.model, prob.obj, x0, prob.tf, N, opts, stats, Z, Z̄, dx, du,
        gains, K, d, D, G, Efull, Eerr, Q, S, ΔV, Qtmp, Quu_reg, Qux_reg, reg, grad, xdot, 
        lg,
    )
    reset!(solver)
end

# Getters
RD.dims(solver::iLQRSolver) = RD.dims(solver.Z) 
RD.state_dim(::iLQRSolver{<:Any,<:Any,n}) where n = n
RD.errstate_dim(::iLQRSolver{<:Any,<:Any,<:Any,e}) where e = e
RD.control_dim(::iLQRSolver{<:Any,<:Any,<:Any,<:Any,m}) where m = m
RD.state_dim(solver::iLQRSolver, k::Integer) = RD.state_dim(solver.model[k])
RD.errstate_dim(solver::iLQRSolver, k::Integer) = RD.errstate_dim(solver.model[k])
RD.control_dim(solver::iLQRSolver, k::Integer) = RD.control_dim(solver.model[k])
@inline TO.get_trajectory(solver::iLQRSolver) = solver.Z
@inline TO.get_objective(solver::iLQRSolver) = solver.obj
@inline TO.get_model(solver::iLQRSolver) = solver.model
@inline get_initial_state(solver::iLQRSolver) = solver.x0
solvername(::Type{<:iLQRSolver}) = :iLQR
getlogger(solver::iLQRSolver) = solver.logger

RD.vectype(::iLQRSolver{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,V}) where V = V
usestatic(obj) = RD.vectype(obj) <: SVector
dynamics_signature(obj) = usestatic(obj) ? RD.StaticReturn() : RD.InPlace()
function_signature(obj) = usestatic(obj) ? RD.StaticReturn() : RD.InPlace()
usestaticdefault(model::RD.AbstractFunction) = Val(RD.default_signature(model) == RD.StaticReturn())

log_level(::iLQRSolver) = InnerLoop

function reset!(solver::iLQRSolver)
    opts = options(solver)::SolverOptions
    reset!(stats(solver), opts.iterations, solvername(solver))
    solver.reg.ρ = solver.opts.bp_reg_initial
    solver.reg.dρ = 0.0
    return solver 
end

function dynamics_expansion!(solver::iLQRSolver, Z=solver.Z)
    diff = solver.opts.dynamics_diffmethod
    D = solver.D
    model = solver.model

    # Dynamics Jacobians
    for k in eachindex(D)
        RobotDynamics.jacobian!(dynamics_signature(solver), diff, model[k], D[k], Z[k])
    end

    # Calculate the expansion on the error state
    error_expansion!(solver.model, D, solver.G)
end

function cost_expansion!(solver::iLQRSolver)
    # Calculate normal cost expansion
    Altro.cost_expansion!(solver.obj, solver.Efull, solver.Z)

    # Calculate the error state expansion
    Altro.error_expansion!(solver.model, solver.Err, solver.Efull, solver.G, solver.G)
end

function increaseregularization!(solver::iLQRSolver)
    reg = solver.reg
    ρdot = solver.opts.bp_reg_increase_factor
    ρmin = solver.opts.bp_reg_min
    reg.dρ =  max(reg.dρ*ρdot, ρdot)
    reg.ρ = max(reg.ρ*reg.dρ, ρmin) 
    return reg
end

function decreaseregularization!(solver::iLQRSolver)
    reg = solver.reg
    ρdot = solver.opts.bp_reg_increase_factor
    ρmin = solver.opts.bp_reg_min
    reg.dρ = min(reg.dρ/ρdot, 1/ρdot)
    reg.ρ = max(ρmin, reg.ρ*reg.dρ)
    return reg
end

"""
    reset_gains!(solver::iLQRSolver)

Resets the gains to zero. Useful if you want to call `solve!` multiple times,
without using information from the previous solve, since the feedback gains
are used to perform the initial rollout. See [`initialize!(::iLQRSolver)`](@ref).
"""
function reset_gains!(solver::iLQRSolver)
    for k in eachindex(solver.K)
        solver.K[k] .= 0
        solver.d[k] .= 0
    end
    return nothing
end

function state_diff_jacobian!(model, G, Z)
    for k = 1:length(Z)
        RD.errstate_jacobian!(model, G[k], Z[k])
    end
end