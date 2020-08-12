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
            setfield!(opts, key, val) 
        end
    end
end

"""
    has_option(opt::AbstractSolverOptions, field::Symbol)
    has_option(opt::AbstractSolver, field::Symbol)

Check to see if a solver or solver option has the option `field`.
"""
@inline function has_option(opts::OPT, field::Symbol) where OPT <: AbstractSolverOptions
    hasfield(OPT, field)
end

"""
    merge_opt(a,b)

Used when merging option outputs from `get_options`. Will set the output to `:invalid`
if the values are different.
"""
merge_opt(a::T, b::T) where T = a == b ? a : :invalid
merge_opt(a::Real, b::Real) = a ≈ b ? a : :invalid
merge_opt(a, b) = :invalid

"""
    check_invalid_merge(d::AbstractDict, d0::Pair{Symbol,<:Dict}...)

Check the the merged dictionary `d` if it contains any invalidated options. If so,
it will print a warning message with details about option, it's values, and which
solver the options came from. 

The original option dictionaries need to passed in as Pairs, with the first element
being a symbol identifying the solver (e.g. :AL, :ALTRO, :iLQR).
"""
function check_invalid_merge(d::AbstractDict, d0::Pair{Symbol,<:Dict}...)
    bad_keys = [k for (k,v) in pairs(d) if v==:invalid]
    for key in bad_keys
        vals = join(["$(d.second[key]) ($(d.first))" for d in d0], ", ")
        @warn "Cannot combine values for option \"$key\". values = $vals.
        Value will be invalidated. Use get_options(solver, true, true) to group by solver."
    end
    return bad_keys 
end

"""
    SolverOptions{T}

Simple mutable struct containing common solver options used by ALTRO. Can be passed
in to any solver constructor, which automatically splats the fields into the keyword
arguments of the solver constructor.
"""
@with_kw mutable struct SolverOptions{T} <: AbstractSolverOptions{T}
    constraint_tolerance::T = 1e-6
    cost_tolerance::T = 1e-4
    cost_tolerance_intermediate::T = 1e-4
    active_set_tolerance::T = 1e-3
    penalty_initial::T = NaN
    penalty_scaling::T = NaN
    iterations::Int = 300
    iterations_inner::Int = 100
    verbose::Bool = false
end

"""
Allow any solver to accept a `SolverOptions` struct, which is automatically converted
to keyword argument pairs.
"""
function (::Type{S})(prob::Problem, opts::SolverOptions; kwargs...) where S <: AbstractSolver
    d = Parameters.type2dict(opts)
    S(prob; kwargs..., d...)
end

@with_kw mutable struct SolverOpts{T} <: AbstractSolverOptions{T}
    # Optimality Tolerances
    constraint_tolerance::T = 1e-6
    cost_tolerance::T = 1e-4
    cost_tolerance_intermediate::T = 1e-4
    gradient_tolerance::T = 1e-5
    gradient_tolerance_intermediate::T = 1e-5

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
    ρ_chol::T = 1e-2     # cholesky factorization regularization
    ρ_primal::T = 1.0e-8 # primal regularization
    r_threshold::T = 1.1

    # General options
    projected_newton::Bool = true
    constrained::Bool = true
    iterations::Int = 300
    verbose::Int = 0 
end

function reset!(conSet::ALConstraintSet{T}, opts::SolverOpts{T}) where T
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