
struct iLQRSolver{T,I<:QuadratureRule,L,O,n,n̄,m,L1} <: UnconstrainedSolver{T}
    # Model + Objective
    model::L
    obj::O

    # Problem info
    x0::MVector{n,T}
    xf::MVector{n,T}
    tf::T
    N::Int

    opts::SolverOpts{T}
    stats::SolverStats{T}

    # Primal Duals
    Z::Traj{n,m,T,KnotPoint{T,n,m,L1}}
    Z̄::Traj{n,m,T,KnotPoint{T,n,m,L1}}

    # Data variables
    # K::Vector{SMatrix{m,n̄,T,L2}}  # State feedback gains (m,n,N-1)
    K::Vector{SizedMatrix{m,n̄,T,2}}  # State feedback gains (m,n,N-1)
    d::Vector{SizedVector{m,T,1}}  # Feedforward gains (m,N-1)

    D::Vector{DynamicsExpansion{T,n,n̄,m}}  # discrete dynamics jacobian (block) (n,n+m+1,N)
    G::Vector{SizedMatrix{n,n̄,T,2}}        # state difference jacobian (n̄, n)

	quad_obj::QuadraticObjective{n,m,T}  # quadratic expansion of obj
	S::QuadraticObjective{n̄,m,T}         # Cost-to-go expansion
	Q::QuadraticObjective{n̄,m,T}         # Action-value expansion

	Quu_reg::SizedMatrix{m,m,T,2}
	Qux_reg::SizedMatrix{m,n̄,T,2}
    ρ::Vector{T}   # Regularization
    dρ::Vector{T}  # Regularization rate of change

    grad::Vector{T}  # Gradient

    logger::SolverLogger

end

function iLQRSolver(
        prob::Problem{QUAD,T}, 
        opts::SolverOpts=SolverOpts(), 
        stats::SolverStats=SolverStats(parent=solvername(iLQRSolver));
        kwarg_opts...
    ) where {QUAD,T}
    set_options!(opts; kwarg_opts...)

    # Init solver results
    n,m,N = size(prob)
    n̄ = RobotDynamics.state_diff_size(prob.model)

    x0 = prob.x0
    xf = prob.xf

    Z = prob.Z
    # Z̄ = Traj(n,m,Z[1].dt,N)
    Z̄ = copy(prob.Z)

	K = [zeros(T,m,n̄) for k = 1:N-1]
    d = [zeros(T,m)   for k = 1:N-1]

	D = [DynamicsExpansion{T}(n,n̄,m) for k = 1:N-1]
	G = [SizedMatrix{n,n̄}(zeros(n,n̄)) for k = 1:N+1]  # add one to the end to use as an intermediate result

	Q = QuadraticObjective(n̄,m,N)
	quad_exp = QuadraticObjective(Q, prob.model)
	S = QuadraticObjective(n̄,m,N)

	Quu_reg = SizedMatrix{m,m}(zeros(m,m))
	Qux_reg = SizedMatrix{m,n̄}(zeros(m,n̄))
    ρ = zeros(T,1)
    dρ = zeros(T,1)

    grad = zeros(T,N-1)

    logger = SolverLogging.default_logger(opts.verbose >= 2)
	L = typeof(prob.model)
	O = typeof(prob.obj)
    solver = iLQRSolver{T,QUAD,L,O,n,n̄,m,n+m}(prob.model, prob.obj, x0, xf,
		prob.tf, N, opts, stats,
        Z, Z̄, K, d, D, G, quad_exp, S, Q, Quu_reg, Qux_reg, ρ, dρ, grad, logger)

    reset!(solver)
    return solver
end

# Getters
Base.size(solver::iLQRSolver{<:Any,<:Any,<:Any,<:Any,n,<:Any,m}) where {n,m} = n,m,solver.N
@inline TO.get_trajectory(solver::iLQRSolver) = solver.Z
@inline TO.get_objective(solver::iLQRSolver) = solver.obj
@inline TO.get_model(solver::iLQRSolver) = solver.model
@inline get_initial_state(solver::iLQRSolver) = solver.x0
@inline TO.integration(solver::iLQRSolver{<:Any,Q}) where Q = Q
solvername(::Type{<:iLQRSolver}) = :iLQR

log_level(::iLQRSolver) = InnerLoop

function reset!(solver::iLQRSolver{T}) where T
    reset_solver!(solver)
    solver.ρ[1] = 0.0
    solver.dρ[1] = 0.0
    return nothing
end

