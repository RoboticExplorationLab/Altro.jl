export
    SolverOptions


abstract type AbstractSolverOptions{T<:Real} end
abstract type DirectSolverOptions{T} <: AbstractSolverOptions{T} end

function set_options!(opts::OPT; d...) where OPT <: AbstractSolverOptions
    for (key,val) in pairs(d)
        if hasfield(OPT, key) 
            setfield!(opts, key, val) 
        end
    end
end

@inline function has_option(opts::OPT, field::Symbol) where OPT <: AbstractSolverOptions
    hasfield(OPT, field)
end

function (::Type{SolverOpts})(opts::Dict{Symbol,<:Any}) where SolverOpts <: AbstractSolverOptions{T} where T
    # add_subsolver_opts!(opts)
    opts_ = filter(x->hasfield(SolverOpts, x[1]), opts)
    SolverOpts(;opts_...)
end

merge_opt(a::T, b::T) where T = a == b ? a : :invalid
merge_opt(a::Real, b::Real) = a ≈ b ? a : :invalid
merge_opt(a, b) = :invalid

function check_invalid_merge(d, d0::Pair{Symbol,<:Dict}...)
    bad_keys = [k for (k,v) in pairs(d) if v==:invalid]
    for key in bad_keys
        vals = join(["$(d.second[key]) ($(d.first))" for d in d0], ", ")
        @warn "Cannot combine values for option \"$key\". values = $vals.
        Value will be invalidated. Use get_options(solver, true, true) to group by solver."
    end
    return bad_keys 
end

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

function (::Type{S})(prob::Problem, opts::SolverOptions; kwargs...) where S <: AbstractSolver
    d = Parameters.type2dict(opts)
    S(prob; kwargs..., d...)
end


@with_kw mutable struct UnconstrainedSolverOptions{T} <: AbstractSolverOptions{T}
    cost_tolerance::T = 1e-4
    iterations::Int = 300
    verbose::Bool = false
end

function (::Type{<:UnconstrainedSolverOptions})(opts::SolverOptions)
    UnconstrainedSolverOptions(
        cost_tolerance=opts.cost_tolerance,
        iterations=opts.iterations,
        verbose=opts.verbose
    )
end
