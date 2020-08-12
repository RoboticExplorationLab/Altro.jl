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
