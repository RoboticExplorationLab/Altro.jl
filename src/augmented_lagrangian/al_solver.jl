export
    AugmentedLagrangianSolver,
    AugmentedLagrangianSolverOptions,
    get_constraints


@with_kw mutable struct ALStats{T}
    iterations::Int = 0
    iterations_total::Int = 0
    iterations_inner::Vector{Int} = zeros(Int,0)
    cost::Vector{T} = zeros(0)
    c_max::Vector{T} = zeros(0)
    penalty_max::Vector{T} = zeros(0)
end

function reset!(stats::ALStats, L=0)
    stats.iterations = 0
    stats.iterations_total = 0
    stats.iterations_inner = zeros(Int,L)
    stats.cost = zeros(L)*NaN
    stats.c_max = zeros(L)*NaN
    stats.penalty_max = zeros(L)*NaN
end


"""$(TYPEDEF)
Solver options for the augmented Lagrangian solver.
$(FIELDS)
"""
@with_kw mutable struct AugmentedLagrangianSolverOptions{T} <: AbstractSolverOptions{T}
    "Print summary at each iteration."
    verbose::Bool=false

    "dJ < ϵ, cost convergence criteria for unconstrained solve or to enter outerloop for constrained solve."
    cost_tolerance::T = 1.0e-4

    "dJ < ϵ_int, intermediate cost convergence criteria to enter outerloop of constrained solve."
    cost_tolerance_intermediate::T = 1.0e-3

    "gradient_norm < ϵ, gradient norm convergence criteria."
    gradient_norm_tolerance::T = 1.0e-5

    "gradient_norm_int < ϵ, gradient norm intermediate convergence criteria."
    gradient_norm_tolerance_intermediate::T = 1.0e-5

    "max(constraint) < ϵ, constraint convergence criteria."
    constraint_tolerance::T = 1.0e-4

    "max(constraint) < ϵ_int, intermediate constraint convergence criteria."
    constraint_tolerance_intermediate::T = 1.0e-3

    "maximum outerloop updates."
    iterations_outer::Int = 30

    "global maximum Lagrange multiplier. If NaN, use value from constraint"
    dual_max::T = NaN

    "global maximum penalty term. If NaN, use value from constraint"
    penalty_max::T = NaN

    "global initial penalty term. If NaN, use value from constraint"
    penalty_initial::T = NaN

    "global penalty update multiplier; penalty_scaling > 1. If NaN, use value from constraint"
    penalty_scaling::T = NaN

    "penalty update multiplier when μ should not be update, typically 1.0 (or 1.0 + ϵ)."
    penalty_scaling_no::T = 1.0

    "ratio of current constraint to previous constraint violation; 0 < constraint_decrease_ratio < 1."
    constraint_decrease_ratio::T = 0.25

    "type of outer loop update (default, feedback)."
    outer_loop_update_type::Symbol = :default

    "numerical tolerance for constraint violation."
    active_constraint_tolerance::T = 0.0

    "terminal solve when maximum penalty is reached."
    kickout_max_penalty::Bool = false

    reset_duals::Bool = true

    reset_penalties::Bool = true

    log_level_outer::Base.CoreLogging.LogLevel = OuterLoop
end

function reset!(conSet::ALConstraintSet{T}, opts::AugmentedLagrangianSolverOptions{T}) where T
    if !isnan(opts.dual_max)
        for params in conSet.params
            params.λ_max = opts.dual_max
        end
    end
    if !isnan(opts.penalty_max)
        for params in conSet.params
            params.μ_max = opts.penalty_max
        end
    end
    if !isnan(opts.penalty_initial)
        for params in conSet.params
            params.μ0 = opts.penalty_initial
        end
    end
    if !isnan(opts.penalty_scaling)
        for params in conSet.params
            params.ϕ = opts.penalty_scaling
        end
    end
    if opts.reset_duals
        TO.reset_duals!(conSet)
    end
    if opts.reset_penalties
        TO.reset_penalties!(conSet)
    end
end

function set_verbosity!(opts::AugmentedLagrangianSolverOptions)
    log_level = opts.log_level_outer
    if opts.verbose
        set_logger()
        Logging.disable_logging(LogLevel(log_level.level-1))
        logger = global_logger()
        if opts.opts_uncon.verbose
            freq = 1
        else
            freq = 5
        end
        logger.leveldata[log_level].freq = freq
    else
        Logging.disable_logging(log_level)
    end
end


@doc raw""" ```julia
struct AugmentedLagrangianSolver <: ConstrainedSolver{T}
```
Augmented Lagrangian (AL) is a standard tool for constrained optimization. For a trajectory optimization problem of the form:
```math
\begin{aligned}
  \min_{x_{0:N},u_{0:N-1}} \quad & \ell_f(x_N) + \sum_{k=0}^{N-1} \ell_k(x_k, u_k, dt) \\
  \textrm{s.t.}            \quad & x_{k+1} = f(x_k, u_k), \\
                                 & g_k(x_k,u_k) \leq 0, \\
                                 & h_k(x_k,u_k) = 0.
\end{aligned}
```
AL methods form the following augmented Lagrangian function:
```math
\begin{aligned}
    \ell_f(x_N) + &λ_N^T c_N(x_N) + c_N(x_N)^T I_{\mu_N} c_N(x_N) \\
           & + \sum_{k=0}^{N-1} \ell_k(x_k,u_k,dt) + λ_k^T c_k(x_k,u_k) + c_k(x_k,u_k)^T I_{\mu_k} c_k(x_k,u_k)
\end{aligned}
```
This function is then minimized with respect to the primal variables using any unconstrained minimization solver (e.g. iLQR).
    After a local minima is found, the AL method updates the Lagrange multipliers λ and the penalty terms μ and repeats the unconstrained minimization.
    AL methods have superlinear convergence as long as the penalty term μ is updated each iteration.
"""
struct AugmentedLagrangianSolver{T,S<:AbstractSolver} <: ConstrainedSolver{T}
    opts::AugmentedLagrangianSolverOptions{T}
    stats::ALStats{T}
    stats_uncon::Vector{STATS} where STATS
    solver_uncon::S
end

AbstractSolver(prob::Problem{Q,T},
    opts::AugmentedLagrangianSolverOptions{T}=AugmentedLagrangianSolverOptions{T}()) where {Q,T} =
    AugmentedLagrangianSolver(prob,opts)

"""$(TYPEDSIGNATURES)
Form an augmented Lagrangian cost function from a Problem and AugmentedLagrangianSolver.
    Does not allocate new memory for the internal arrays, but points to the arrays in the solver.
"""
function AugmentedLagrangianSolver(prob::Problem{Q,T};
        solver_uncon=iLQRSolver, opts...) where {Q,T}
    # Init solver statistics
    stats = ALStats()
    stats_uncon = Vector{iLQRStats{T}}()

    # Create solver options
    opts_al = AugmentedLagrangianSolverOptions()

    # Build Augmented Lagrangian Objective
    alobj = ALObjective(prob)
    rollout!(prob)
    prob_al = Problem{Q}(prob.model, alobj, ConstraintList(size(prob)...),
        prob.x0, prob.xf, prob.Z, prob.N, prob.t0, prob.tf)

    # Instantiate the unconstrained solver
    solver_uncon = solver_uncon(prob_al)

    # Build solver
    solver = AugmentedLagrangianSolver(opts_al, stats, stats_uncon, solver_uncon)
    reset!(solver)
    set_options!(solver; opts...)
    return solver
end

# Getters
Base.size(solver::AugmentedLagrangianSolver) = size(solver.solver_uncon)
@inline TO.cost(solver::AugmentedLagrangianSolver) = TO.cost(solver.solver_uncon)
@inline TO.get_trajectory(solver::AugmentedLagrangianSolver) = get_trajectory(solver.solver_uncon)
@inline TO.get_objective(solver::AugmentedLagrangianSolver) = get_objective(solver.solver_uncon)
@inline TO.get_model(solver::AugmentedLagrangianSolver) = get_model(solver.solver_uncon)
@inline get_initial_state(solver::AugmentedLagrangianSolver) = get_initial_state(solver.solver_uncon)
@inline iterations(solver::AugmentedLagrangianSolver) = solver.stats.iterations_total

function TO.get_constraints(solver::AugmentedLagrangianSolver{T}) where T
    obj = get_objective(solver)::ALObjective{T}
    obj.constraints
end


# Options methods
function set_options!(solver::AugmentedLagrangianSolver; opts...)
    v = get_verbosity(opts)
    set_options!(solver.opts; opts..., verbose=v[2])
    set_options!(solver.solver_uncon; opts..., verbose=v[3])
end

function has_option(solver::AugmentedLagrangianSolver, field::Symbol)
    has_option(solver.solver_uncon, field) || has_option(solver.opts, field)
end

function get_option(solver::AugmentedLagrangianSolver, field::Symbol)
    if has_option(solver.opts, field)
        return solver.opts[field]
    elseif has_option(solver.solver_uncon, field)
        return get_option(solver.solver_uncon, field)
    else
        throw(ErrorException("$field is not a valid option"))
    end
end

function get_options(solver::AugmentedLagrangianSolver, recursive::Bool=true, group::Bool=false)
    d1 = _get_options(solver)
    if recursive
        d2 = get_options(solver.solver_uncon)
        if group 
            Dict(:AL => d1, :iLQR => d2)
        else
            d = merge(merge_opt, d1, d2)
            check_invalid_merge(d, :AL=>d1, :iLQR=>d2)
            return d
        end
    else
        d1
    end
end

function reset!(solver::AugmentedLagrangianSolver{T}) where T
    reset!(solver.stats, solver.opts.iterations_outer)
    reset!(solver.solver_uncon)
    reset!(get_constraints(solver), solver.opts)
end

############################################################################################
#                           AUGMENTED LAGRANGIAN OBJECTIVE                                 #
############################################################################################

struct ALObjective{T,O<:Objective} <: AbstractObjective
    obj::O
    constraints::ALConstraintSet{T}
end

function ALObjective(obj::Objective, cons::ConstraintList, model::AbstractModel)
    ALObjective(obj, ALConstraintSet(cons, model))
end
@inline ALObjective(prob::Problem) = ALObjective(prob.obj, prob.constraints, prob.model)

@inline TO.get_J(obj::ALObjective) = obj.obj.J
@inline Base.length(obj::ALObjective) = length(obj.obj)


function Base.copy(obj::ALObjective)
    ALObjective(obj.obj, ConstraintSet(copy(obj.constraints.constraints), length(obj.obj)))
end

function TO.cost!(obj::ALObjective, Z::AbstractTrajectory)
    # Calculate unconstrained cost
    TO.cost!(obj.obj, Z)

    # Calculate constrained cost
    TO.evaluate!(obj.constraints, Z)
    TO.update_active_set!(obj.constraints, Val(0.0))
    TO.cost!(TO.get_J(obj), obj.constraints)
end

function TO.cost_expansion!(E::QuadraticObjective, obj::ALObjective, Z::Traj, init::Bool=false, rezero::Bool=false)
    # Update constraint jacobians
    TO.jacobian!(obj.constraints, Z)

    # Calculate expansion of original objective
    TO.cost_expansion!(E, obj.obj, Z, true, rezero)

    # Add in expansion of constraints
    TO.cost_expansion!(E, obj.constraints, Z, true)
end
