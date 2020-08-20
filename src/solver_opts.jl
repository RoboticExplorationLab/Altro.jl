export
    SolverOptions


abstract type AbstractSolverOptions{T<:Real} end
abstract type DirectSolverOptions{T} <: AbstractSolverOptions{T} end

"""
    set_options!(opts::AbstractSolverOptions; kwargs...)
    set_options!(solver::AbstractSolver; opts...)

Set solver options via keyword arguments, supporting either solvers or solver option
types directly. Will set any and all options that match the provided arguments. For 
example, `set_options!(solver::ALTROSolver, constraint_tolerance=1e-4)` will set the
constraint tolerance option in the ALTRO, Augmented Lagrangian, and Project Newton 
solvers.

The only exeption is the `verbose` setting, which always accepts a boolean, while
ALTRO and Augmented Lagrangian solvers accept integers 0-2, with 1 providing output
for the outer AL iterations but not the iLQR iterations.
"""
function set_options!(opts::OPT; d...) where OPT <: AbstractSolverOptions
    for (key,val) in pairs(d)
        if hasfield(OPT, key) 
            val = convert(fieldtype(OPT, key), val)
            setfield!(opts, key, val) 
        end
    end
end


@with_kw mutable struct SolverOptions{T} <: AbstractSolverOptions{T}
    # Optimality Tolerances
    constraint_tolerance::T = 1e-6
    cost_tolerance::T = 1e-4
    cost_tolerance_intermediate::T = 1e-4
    gradient_tolerance::T = 10.0
    gradient_tolerance_intermediate::T = 1.0

    # iLQR
    iterations_inner::Int = 300
    dJ_counter_limit::Int = 10
    square_root::Bool = false
    line_search_lower_bound::T = 1e-8
    line_search_upper_bound::T = 10.0
    iterations_linesearch::Int = 20
    max_cost_value::T = 1.0e8
    max_state_value::T = 1.0e8
    max_control_value::T = 1.0e8
	static_bp::Bool = true
	save_S::Bool = false

    # Backward pass regularization
    bp_reg::Bool = false
    bp_reg_initial::T = 0.0
    bp_reg_increase_factor::T = 1.6
    bp_reg_max::T = 1.0e8
    bp_reg_min::T = 1.0e-8
    bp_reg_type::Symbol = :control
    bp_reg_fp::T = 10.0

    # Augmented Lagrangian
    penalty_initial::T = NaN
    penalty_scaling::T = NaN
    active_set_tolerance_al::T = 1e-3
    dual_max::T = NaN
    penalty_max::T = NaN
    iterations_outer::Int = 30
    kickout_max_penalty::Bool = false
    reset_duals::Bool = true
    reset_penalties::Bool = true

    # Projected Newton
    verbose_pn::Bool = false
    n_steps::Int = 2
    solve_type::Symbol = :feasible
    projected_newton_tolerance::T = 1e-3
    active_set_tolerance_pn::T = 1e-3
    multiplier_projection::Bool = true
    ρ_chol::T = 1e-2     # cholesky factorization regularization
    ρ_primal::T = 1.0e-8 # primal regularization
    ρ_dual::T = 1.0e-8   # regularization for multiplier projection 
    r_threshold::T = 1.1

    # General options
    projected_newton::Bool = true
    iterations::Int = 1000   # max number of iterations
    show_summary::Bool = false
    verbose::Int = 0 
end

function reset!(conSet::ALConstraintSet{T}, opts::SolverOptions{T}) where T
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

@with_kw mutable struct SolverStats{T}
    # Iteration counts
    iterations::Int = 0
    iterations_outer::Int = 0
    iterations_pn::Int = 0

    # Iteration stats
    iteration::Vector{Int} = Int[] 
    iteration_outer::Vector{Int} = Int[]
    iteration_pn::Vector{Bool} = Bool[]   # is the iteration a pn iteration
    cost::Vector{T} = Float64[]
    dJ::Vector{T} = Float64[]
    c_max::Vector{T} = Float64[]
    gradient::Vector{T} = Float64[] 
    penalty_max::Vector{T} = Float64[]

    # iLQR
    dJ_zero_counter::Int = 0

    # Other
    tstart::Float64 = time()
    tsolve::Float64 = Inf 
    status::TerminationStatus = UNSOLVED
    is_reset::Bool = false
    "Which solver is the top-level solver and responsible for resetting and trimming."
    parent::Symbol
end

function reset!(stats::SolverStats, N::Int, parent::Symbol)
    if parent == stats.parent
        stats.is_reset = false  # force a reset
        reset!(stats, N)
    end
    return nothing
end

function reset!(stats::SolverStats, N::Int=0)
    stats.is_reset && return nothing
    stats.iterations = 0
    stats.iterations_outer = 0
    stats.iterations_pn = 0
    function reset!(v::AbstractVector{T}, N::Int) where T 
        resize!(v, N)
        v .= zero(T)
    end
    reset!(stats.iteration, N)
    reset!(stats.iteration_outer, N)
    reset!(stats.iteration_pn, N)
    reset!(stats.cost, N)
    reset!(stats.dJ, N)
    reset!(stats.c_max, N)
    reset!(stats.gradient, N)
    reset!(stats.penalty_max, N)
    stats.tstart = time_ns()
    stats.tsolve = Inf
    stats.dJ_zero_counter = 0
    stats.is_reset = true
    return nothing
end

function record_iteration!(stats::SolverStats; 
        cost=Inf, dJ=Inf, c_max=Inf, gradient=Inf, penalty_max=Inf, 
        is_pn::Bool=false, is_outer::Bool=false
    )
    # don't increment the iteration for an outer loop
    if is_outer
        stats.iterations_outer += 1
    else
        stats.iterations += 1
    end
    i = stats.iterations
    function record!(vec, val, i)
        if val == Inf
            val = i > 1 ? vec[i-1] : val
        end
        vec[i] = val
    end
    stats.iteration[i] = i
    stats.iteration_outer[i] = stats.iterations_outer
    stats.iteration_pn[i] = is_pn
    record!(stats.cost, cost, i)
    record!(stats.dJ, dJ, i)
    record!(stats.c_max, c_max, i)
    record!(stats.gradient, gradient, i)
    record!(stats.penalty_max, penalty_max, i)
    is_pn && (stats.iterations_pn += 1)
    return stats.iterations
end

function trim!(stats::SolverStats, parent::Symbol)
    if parent == stats.parent
        trim!(stats)
    end
end
function trim!(stats::SolverStats)
    N = stats.iterations
    resize!(stats.iteration, N)
    resize!(stats.iteration_outer, N)
    resize!(stats.iteration_pn, N)
    resize!(stats.cost, N)
    resize!(stats.dJ, N)
    resize!(stats.c_max, N)
    resize!(stats.gradient, N)
    resize!(stats.penalty_max, N)
    stats.is_reset = false
    return N 
end

function Dict(stats::SolverStats)
    fields = (:iteration, :iteration_outer, :iteration_pn, 
        :cost, :dJ, :c_max, :gradient, :penalty_max)
    pairs = [field=>getfield(stats,field) for field in fields]
    Dict(pairs...)
end