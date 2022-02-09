const ALCONSTRAINT_PARAMS = Set((
    :use_conic_cost, 
    :penalty_initial, 
    :penalty_scaling, 
    :penalty_max, 
    :dual_max
))

Base.@kwdef mutable struct ConstraintOptions{T}
    use_conic_cost::Bool = false
    penalty_initial::T = 1.0
    penalty_scaling::T = 10.0 
    penalty_max::T = 1e8
    dual_max::T = 1e8
    usedefault::Dict{Symbol,Bool} = Dict(Pair.(
        ALCONSTRAINT_PARAMS, 
        trues(length(ALCONSTRAINT_PARAMS))
    ))
end

function setparams!(conopts::ConstraintOptions; kwargs...)
    for (key,val) in pairs(kwargs)
        if key in ALCONSTRAINT_PARAMS 
            setfield!(conopts, key, val)
            conopts.usedefault[key] = false
        end
    end
end

function setparams!(conopts::ConstraintOptions, opts::SolverOptions)
    for param in ALCONSTRAINT_PARAMS
        if conopts.usedefault[param]
            setfield!(conopts, param, getfield(opts, param))
        end
    end
end

struct ALConstraint{T, C<:TO.StageConstraint}
    n::Int  # state dimension
    m::Int  # control dimension
    con::C
    sig::FunctionSignature
    diffmethod::DiffMethod
    inds::Vector{Int}             # knot point indices for constraint
    vals::Vector{Vector{T}}       # constraint values
    jac::Vector{Matrix{T}}       # full state constraint Jacobian
    λ::Vector{Vector{T}}          # dual variables
    μ::Vector{Vector{T}}          # penalties 
    μinv::Vector{Vector{T}}       # inverted penalties 
    λbar::Vector{Vector{T}}       # approximate dual variable 
    λproj::Vector{Vector{T}}      # projected dual variables
    λscaled::Vector{Vector{T}}    # scaled projected dual variables
    c_max::Vector{T}

    ∇proj::Vector{Matrix{T}}   # Jacobian of projection
    ∇²proj::Vector{Matrix{T}}  # Second-order derivative of projection
    grad::Vector{Vector{T}}    # gradient of Augmented Lagrangian
    hess::Vector{Matrix{T}}    # Hessian of Augmented Lagrangian
    tmp_jac::Matrix{T}

    opts::ConstraintOptions{T}
    function ALConstraint{T}(n::Int, m::Int, con::TO.StageConstraint, 
                             inds::AbstractVector{<:Integer}; 
			                 sig::FunctionSignature=StaticReturn(), 
                             diffmethod::DiffMethod=UserDefined(),
                             kwargs...
    ) where T
        opts = ConstraintOptions{T}(;kwargs...)

        p = RD.output_dim(con)
        P = length(inds)
        nm = n + m

        vals = [zeros(T, p) for i = 1:P]
        jac = [zeros(T, p, nm) for i = 1:P]
        λ = [zeros(T, p) for i = 1:P]
        μ = [fill(opts.penalty_initial, p) for i = 1:P]
        μinv = [inv.(μi) for μi in μ]
        λbar = [zeros(T, p) for i = 1:P]
        λproj = [zeros(T, p) for i = 1:P]
        λscaled = [zeros(T, p) for i = 1:P]
        c_max = zeros(T, P)

        ∇proj = [zeros(T, p, p) for i = 1:P]
        ∇²proj = [zeros(T, p, p) for i = 1:P]
        
        grad = [zeros(T, nm) for i = 1:P]
        hess = [zeros(T, nm, nm) for i = 1:P]

        tmp_jac = zeros(T, p, n+m)

        new{T, typeof(con)}(
            n, m, con, sig, diffmethod, inds, vals, jac, λ, μ, μinv, λbar, 
            λproj, λscaled, c_max, ∇proj, ∇²proj, grad, hess, tmp_jac, opts
        )
    end
end

function evaluate_constraint!(alcon::ALConstraint, Z::AbstractTrajectory)
    for (i,k) in enumerate(alcon.inds)
        TO.evaluate_constraint!(alcon.sig, alcon.con, alcon.vals[i], Z[k])
    end
end

function constraint_jacobian!(alcon::ALConstraint, Z::AbstractTrajectory)
    for (i,k) in enumerate(alcon.inds)
        RD.jacobian!(alcon.sig, alcon.diffmethod, alcon.con, alcon.jac[i], alcon.vals[i], Z[k])
    end
end


@doc raw"""
    alcost(alcon, Z)

Calculates the additional cost added by the augmented Lagrangian:

```math
\sum_{i=1}^{P} \frac{1}{2 \mu_i} || \Pi_K(\lambda_i - \mu_i c(x_k)) ||^2 - || \lambda_i ||^2
```

where $k$ is the $i$th knot point of $P$ to which the constraint applies, and $K$ is the 
cone for the constraint.
"""
function alcost(alcon::ALConstraint{T}, Z::AbstractTrajectory) where T
    J = zero(T)
    evaluate_constraint!(alcon, Z)
    use_conic = alcon.opts.use_conic_cost
    cone = TO.sense(alcon.con)
    for i in eachindex(alcon.inds)
        if use_conic
            # Use generic conic cost
            J += alcost(alcon, i)
        else
            # Special-case on the cone
            J += alcost(cone, alcon, i)
        end
    end
    return J
end

function algrad(alcon::ALConstraint{T}, Z::AbstractTrajectory) where T
    J = zero(T)
    constraint_jacobian!(alcon, Z)
    use_conic = alcon.opts.use_conic_cost
    cone = TO.sense(alcon.con)
    for i in eachindex(alcon.inds)
        if use_conic
            # Use generic conic cost
            J += algrad(alcon, i)
        else
            # Special-case on the cone
            J += algrad(cone, alcon, i)
        end
    end
    return J
end

function alhess(alcon::ALConstraint{T}, Z::AbstractTrajectory) where T
    # Assumes Jacobians have already been computed
    J = zero(T)
    use_conic = alcon.opts.use_conic_cost
    cone = TO.sense(alcon.con)
    for i in eachindex(alcon.inds)
        if use_conic
            # Use generic conic cost
            J += alhess(alcon, i)
        else
            # Special-case on the cone
            J += alhess(cone, alcon, i)
        end
    end
    return J
end


##############################
# Equality Constraints
##############################
function alcost(::TO.Equality, alcon::ALConstraint, i::Integer)
    λ, μ, c = alcon.λ[i], alcon.μ[i], alcon.vals[i]
    Iμ = Diagonal(μ)
    return λ'c + 0.5 * c'Iμ*c
end

function algrad!(::TO.Equality, alcon::ALConstraint, i::Integer)
    λbar = alcon.λbar[i]
    λ, μ, c = alcon.λ[i], alcon.μ[i], alcon.vals[i]
    ∇c = alcon.jac[i]
    grad = alcon.grad[i]

    λbar .= λ .+ μ .* c
    mul!(grad, ∇c', λbar)
    return nothing
end

function alhess!(::TO.Equality, alcon::ALConstraint, i::Integer)
    ∇c = alcon.jac[i] 
    hess = alcon.hess[i]
    tmp = alcon.tmp_jac
    Iμ = Diagonal(alcon.μ[i])
    mul!(tmp, Iμ, ∇c)
    mul!(hess, ∇c', tmp)
    return nothing
end

##############################
# Inequality Constraints
##############################
function alcost(::TO.Inequality, alcon::ALConstraint, i::Integer)
    λ, μ, c = alcon.λ[i], alcon.μ[i], alcon.vals[i]
    a = alcon.λbar[i]
    for i = 1:length(a)
        isactive = (c[i] >= 0) | (λ[i] > 0)
        a[i] = isactive * μ[i] 
    end
    Iμ = Diagonal(a)
    return λ'c + 0.5 * c'Iμ*c
end

function algrad!(::TO.Inequality, alcon::ALConstraint, i::Integer)
    ∇c, λbar = alcon.jac[i], alcon.λbar[i]
    λ, μ, c = alcon.λ[i], alcon.μ[i], alcon.vals[i]
    grad = alcon.grad[i]
    a = alcon.λbar[i]
    for i = 1:length(a)
        isactive = (c[i] >= 0) | (λ[i] > 0)
        a[i] = isactive * μ[i] 
    end
    λbar .= λ .+ a .* c
    mul!(grad, ∇c', λbar)
    return nothing
end

function alhess!(::TO.Inequality, alcon::ALConstraint, i::Integer)
    ∇c = alcon.jac[i] 
    c = alcon.vals[i]
    λ, μ = alcon.λ[i], alcon.μ[i]
    hess = alcon.hess[i]
    tmp = alcon.tmp_jac
    a = alcon.λbar[i]
    for i = 1:length(a)
        isactive = (c[i] >= 0) | (λ[i] > 0)
        a[i] = isactive * μ[i] 
    end
    Iμ = Diagonal(a)
    mul!(tmp, Iμ, ∇c)
    mul!(hess, ∇c', tmp)
    return nothing
end

@inline alcost(::TO.ConstraintSense, alcon::ALConstraint, i::Integer) = alcost(alcon, i)

##############################
# Generic Cones
##############################

function alcost(alcon::ALConstraint, i::Integer)
    dualcone = TO.dualcone(TO.sense(alcon.con))

    λ, λbar, λp, λs = alcon.λ[i], alcon.λbar[i], alcon.λproj[i], alcon.λscaled[i]
    μ, μinv, c = alcon.μ[i], alcon.μinv[i], alcon.vals[i]

    # Approximate dual
    λbar .= λ .- μ .* c

    # Projected approximate dual
    TO.projection!(dualcone, λp, λbar)

    # Scaled dual
    λs .= μinv .* λp

    # Cost
    Iμ = Diagonal(μinv)
    return 0.5 * (λp'λs - λ'Iμ*λ)
end

function algrad!(alcon::ALConstraint, i::Integer)
    dualcone = TO.dualcone(TO.sense(alcon.con))

    # Assume λbar and λp have already been calculated
    λbar, λp, λs = alcon.λbar[i], alcon.λproj[i], alcon.λscaled[i]
    μ, c, ∇c = alcon.μ[i], alcon.vals[i], alcon.jac[i]
    ∇proj = alcon.∇proj[i]
    grad = alcon.grad[i]
    tmp = alcon.tmp_jac

    # grad = -∇c'∇proj'Iμ*λp
    TO.∇projection!(dualcone, ∇proj, λbar)
    mul!(tmp, ∇proj, ∇c)  # derivative of λp wrt x
    tmp .*= -1
    mul!(grad, tmp', λs)
    return nothing
end

function alhess!(alcon::ALConstraint, i::Integer)
    dualcone = TO.dualcone(TO.sense(alcon.con))

    λbar, λs = alcon.λbar[i], alcon.λscaled[i]
    μ, c, ∇c = alcon.μ[i], alcon.vals[i], alcon.jac[i]
    ∇proj, ∇²proj = alcon.∇proj[i], alcon.∇²proj[i]
    hess = alcon.hess[i]
    tmp = alcon.tmp_jac

    # Assume 𝝯proj is already computed
    # TODO: reuse this from before
    mul!(tmp, ∇proj, ∇c)  # derivative of λp wrt x
    tmp .*= -1

    # Calculate second-order projection term
    TO.∇²projection!(dualcone, ∇²proj, λbar, λs)

    # hess = ∇c'∇proj'Iμ*∇proj*∇c + ∇c'∇²proj(Iμ*λp)*∇c
    mul!(hess, tmp', tmp)
    mul!(tmp, ∇²proj, ∇c)
    mul!(alcon.hess[i], ∇c', tmp, 1.0, 1.0)
    return nothing
end

##############################
# Dual and Penalty Updates
##############################

function dualupdate!(alcon::ALConstraint)
    dualcone = TO.dualcone(TO.sense(alcon.con))
    use_conic = alcon.opts.use_conic_cost
    λ_max = alcon.opts.dual_max
    for i in eachindex(alcon.inds)
        λ, μ, c = alcon.λ[i], alcon.μ[i], alcon.c[i]
        if use_conic 
            dualupdate!(λ, μ, c)
        else
            # Special-case to the cone
            dualupdate!(dualcone, λ, μ, c)
        end
        # Saturate dual variables
        clamp!(λ, -λ_max, λ_max)
    end
end

function dualupdate!(::TO.Equality, alcon::ALConstraint, i::Integer)
    λbar, λ, μ, c = alcon.λbar[i], alcon.λ[i], alcon.μ[i], alcon.vals[i]
    λbar .= λ .+ μ .* c
    λ .= λbar
    return nothing
end

function dualupdate!(::TO.Inequality, alcon::ALConstraint, i::Integer)
    λbar, λ, μ, c = alcon.λbar[i], alcon.λ[i], alcon.μ[i], alcon.vals[i]
    λbar .= λ .+ μ .* c
    λ .= max.(0, λbar)
    return nothing
end

@inline dualupdate!(::TO.SecondOrderCone, alcon::ALConstraint, i::Integer) = 
    dualupdate!(λ, μ, c)

function dualupdate!(alcon::ALConstraint, i::Integer)
    dualcone = TO.dualcone(TO.sense(alcon.con))
    λbar, λ, μ, c = alcon.λbar[i], alcon.λ[i], alcon.μ[i], alcon.vals[i]
    λbar .= λ .+ μ .* c
    TO.projection!(dualcone, λ, λbar)
    return nothing
end

function penaltyupdate!(alcon::ALConstraint)
    μ = alcon.μ
    ϕ = alcon.opts.penalty_increase_factor
    μ_max = alcon.opts.penalty_max
    for i = 1:length(alcon.inds)
        μ[i] .*= ϕ 
        clamp!(alcon.μ[i], 0, μ_max)
        alcon.μinv[i] .= inv.(μ[i])
    end
end