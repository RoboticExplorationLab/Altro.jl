
# @with_kw mutable struct iLQRStats{T}
#     iterations::Int = 0
#     cost::Vector{T} = [0.]
#     dJ::Vector{T} = [0.]
#     gradient::Vector{T} = [0.]
#     dJ_zero_counter::Int = 0
# end

# function reset!(stats::iLQRStats, N=0)
#     stats.iterations = 0
#     stats.cost = zeros(N)
#     stats.dJ = zeros(N)
#     stats.gradient = zeros(N)
#     stats.dJ_zero_counter = 0
# end

# """$(TYPEDEF)
# Solver options for the iterative LQR (iLQR) solver.
# $(FIELDS)
# """
# @with_kw mutable struct iLQRSolverOptions{T} <: AbstractSolverOptions{T}
#     # Options

#     "Print summary at each iteration."
#     verbose::Bool=false

#     # "Live plotting."
#     # live_plotting::Symbol=:off # :state, :control

#     "dJ < ϵ, cost convergence criteria for unconstrained solve or to enter outerloop for constrained solve."
#     cost_tolerance::T = 1.0e-4

#     # "gradient type: :todorov, :feedforward."
#     # gradient_type::Symbol = :todorov

#     "gradient_norm < ϵ, gradient norm convergence criteria."
#     gradient_norm_tolerance::T = 1.0e-5

#     "iLQR iterations."
#     iterations::Int = 300

#     "restricts the total number of times a forward pass fails, resulting in regularization, before exiting."
#     dJ_counter_limit::Int = 10

#     "use square root method backward pass for numerical conditioning."
#     square_root::Bool = false

#     "forward pass approximate line search lower bound, 0 < line_search_lower_bound < line_search_upper_bound."
#     line_search_lower_bound::T = 1.0e-8

#     "forward pass approximate line search upper bound, 0 < line_search_lower_bound < line_search_upper_bound < ∞."
#     line_search_upper_bound::T = 10.0

#     "maximum number of backtracking steps during forward pass line search."
#     iterations_linesearch::Int = 20

#     # Regularization
#     "initial regularization."
#     bp_reg_initial::T = 0.0

#     "regularization scaling factor."
#     bp_reg_increase_factor::T = 1.6

#     "maximum regularization value."
#     bp_reg_max::T = 1.0e8

#     "minimum regularization value."
#     bp_reg_min::T = 1.0e-8

#     "type of regularization- control: () + ρI, state: (S + ρI); see Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization."
#     bp_reg_type::Symbol = :control

#     "additive regularization when forward pass reaches max iterations."
#     bp_reg_fp::T = 10.0

#     # square root backward pass options:
#     "type of matrix inversion for bp sqrt step."
#     bp_sqrt_inv_type::Symbol = :pseudo

#     "initial regularization for square root method."
#     bp_reg_sqrt_initial::T = 1.0e-6

#     "regularization scaling factor for square root method."
#     bp_reg_sqrt_increase_factor::T = 10.0

# 	bp_reg::Bool = false

#     # Solver Numerical Limits
#     "maximum cost value, if exceded solve will error."
#     max_cost_value::T = 1.0e8

#     "maximum state value, evaluated during rollout, if exceded solve will error."
#     max_state_value::T = 1.0e8

#     "maximum control value, evaluated during rollout, if exceded solve will error."
#     max_control_value::T = 1.0e8

# 	static_bp::Bool = true

#     log_level::Base.CoreLogging.LogLevel = InnerLoop

# 	"Save the Cost-to-Go expansion at each time step for static backwardpass"
# 	save_S::Bool = false
# end

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

