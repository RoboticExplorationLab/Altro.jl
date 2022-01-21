mutable struct DynamicRegularization{T}
    ρ::T
    dρ::T
end

struct iLQRSolver2{L,O,Nx,Ne,Nu,T} <: UnconstrainedSolver{T}
    model::L
    obj::O

    x0::Vector{T}
    tf::T
    N::Int

    opts::SolverOptions{T}
    stats::SolverStats{T}

    Z::Traj{Nx,Nu,T,KnotPoint{Nx,Nu,Vector{T},T}}
    Z̄::Traj{Nx,Nu,T,KnotPoint{Nx,Nu,Vector{T},T}}

    K::Vector{Matrix{T}}  # N-1 × (Nu,Ne)
    d::Vector{Vector{T}}  # N-1 × (Nu,)

    D::Vector{DynamicsExpansion2{T}}
    G::Vector{Matrix{T}}  # N × (Nx,Ne) 

    Efull::CostExpansion2{T}  # Cost expansion (full state)
    Eerr::CostExpansion2{T}   # Cost expansion (error state)
    Q::Vector{StateControlExpansion{T}}      # Action-value expansion
    S::Vector{StateControlExpansion{T}}      # Action-value expansion

    Quu_reg::Matrix{T}
    Qux_reg::Matrix{T}
    reg::DynamicRegularization{T}

    grad::Vector{T}
    xdot::Vector{T}

    logger::SolverLogger
end

function iLQRSolver2(
        prob::Problem{T}, 
        opts::SolverOptions=SolverOptions{T}(), 
        stats::SolverStats=SolverStats{T}(parent=solvername(iLQRSolver));
        kwarg_opts...
    ) where {T}
    set_options!(opts; kwarg_opts...)

    n,m,N = dims(prob)
    e = RD.errstate_dim(prob.model)

    x0 = copy(prob.x0)
    Z = Traj(map(prob.Z) do z
        Nx = state_dim(z)
        Nu = control_dim(z)
        RD.KnotPoint{Nx,Nu}(Vector(RD.getdata(z)), RD.getparams(z)...)
    end)
    Z̄ = copy(Z)
    K = [zeros(T,m,e) for k = 1:N-1]
    d = [zeros(T,m) for k = 1:N-1]

    D = [DynamicsExpansion2{T}(n,e,m) for k = 1:N-1]
    G = [zeros(T,n,e) for k = 1:N+1]

    Eerr = CostExpansion2{T}(e,m,N)
    Efull = FullStateExpansion(Eerr, prob.model)
    Q = [StateControlExpansion{T}(e,m) for k = 1:N] 
    S = [StateControlExpansion{T}(e) for k = 1:N] 
    
    Quu_reg = zeros(m,m)
    Qux_reg = zeros(m,n)
    reg = DynamicRegularization{T}(opts.bp_reg_initial, 0)

    grad = zeros(T,N-1)
    xdot = zeros(T,n)

    logger = SolverLogging.default_logger(opts.verbose >= 2)
	L = typeof(prob.model)
	O = typeof(prob.obj)
    solver = iLQRSolver2{L,O,n,e,m,T}(
        prob.model, prob.obj, x0, prob.tf, N,opts, stats,Z, Z̄, K, d, D, G, 
        Efull, Eerr, Q, S, Quu_reg, Qux_reg, reg, grad, xdot, logger
    )
    reset!(solver)
end

# Getters
RD.dims(solver::iLQRSolver2{<:Any,<:Any,n,<:Any,m}) where {n,m} = n,m,solver.N
@inline TO.get_trajectory(solver::iLQRSolver2) = solver.Z
@inline TO.get_objective(solver::iLQRSolver2) = solver.obj
@inline TO.get_model(solver::iLQRSolver2) = solver.model
@inline get_initial_state(solver::iLQRSolver2) = solver.x0
solvername(::Type{<:iLQRSolver2}) = :iLQR

log_level(::iLQRSolver2) = InnerLoop

function reset!(solver::iLQRSolver2)
    reset_solver!(solver)
    solver.reg.ρ = solver.opts.bp_reg_initial
    solver.reg.dρ = 0.0
    return solver 
end

function dynamics_expansion!(solver::iLQRSolver2)
    sig = solver.opts.dynamics_funsig
    diff = solver.opts.dynamics_diffmethod
    D = solver.D
    Z = solver.Z
    model = solver.model

    # Dynamics Jacobians
    for k in eachindex(D)
        RobotDynamics.jacobian!(sig, diff, model, D[k], Z[k])
    end

    # Calculate the expansion on the error state
    error_expansion!(solver.model, D, solver.G)
end

function cost_expansion!(solver::iLQRSolver2)
    # Calculate normal cost expansion
    Altro.cost_expansion!(solver.obj, solver.Efull, solver.Z)

    # Calculate the error state expansion
    Altro.error_expansion!(solver.model, solver.Err, solver.Efull, solver.G, solver.G)
end