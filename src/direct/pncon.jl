Base.@kwdef mutable struct PNConstraintParams{T}
    solveropts::SolverOptions{T} = SolverOptions{T}()
    active_set_tolerance::T = 1e-3
    usedefault::Dict{Symbol,Bool} = Dict(:active_set_tolerance=>true)
end

function setparams!(conopts::PNConstraintParams; kwargs...)
    for (key,val) in pairs(kwargs)
        if key == :active_xet_tolerance
            conopts.active_set_tolerance = val
            conopts.usedefault[:active_set_tolerance] = false
        end
    end
end

function reset!(conopts::PNConstraintParams)
    if conopts.usedefault[:active_set_tolerance]
        conopts.active_set_tolerance = conopts.solveropts.active_set_tolerance_pn
    end
end


struct PNConstraint{T, C<:TO.StageConstraint, R<:SampledTrajectory}
    n::Int  # state dimension
    m::Int  # control dimension
    con::C
    sig::FunctionSignature
    diffmethod::DiffMethod
    inds::Vector{Int}             # knot point indices for constraint
    vals::Vector{VectorView{T,Int}}
    jac::Vector{Matrix{T}}
    jacviews::Vector{SparseBlockView{T,Int}}
    active::Vector{SubArray{Bool, 1, BitVector, Tuple{UnitRange{Int}}, true}}
    opts::PNConstraintParams{T}
    Z::Vector{R}
end

function PNConstraint(Z::R, con::TO.StageConstraint, 
                      inds::AbstractVector{<:Integer}, d, a, 
                      jacviews::Vector{<:SparseBlockView}; 
                      sig::FunctionSignature=RD.default_signature(con), 
                      diffmethod::DiffMethod=RD.default_diffmethod(con),
                      solveropts::SolverOptions=SolverOptions()
    ) where {R<:SampledTrajectory}
    T = eltype(d)
    n,m,N = RD.dims(Z)
    p = RD.output_dim(con)
    P = length(inds)
    nm = n + m
    Np = RD.num_vars(Z)
    if TO.sense(con) == TO.SecondOrderCone()
        error("ProjectedNewtonSolver doesn't support SecondOrderCone constraints.")
    end
    cinds = [view.block.block.i2 for view in jacviews]

    @assert length(cinds) == P
    @assert all(x->length(x) == p, cinds)

    # xi(k) = (k-1)*(n+m) .+ (1:n)
    # ui(k) = n + (k-1)*(n+m) .+ (1:m)
    # zi(k) = (k-1)*(n+m) .+ (k == N ? (1:n) : (1:n+m))
    vals = [view(d, ci .- Np) for ci in cinds]
    # jac = [view(D, cinds[i], zi(k)) for (i,k) in enumerate(inds)]
    jac = [zeros(T, size(view')) for view in jacviews]
    active = [view(a, ci .- Np) for ci in cinds]

    conopts = PNConstraintParams(solveropts=solveropts)
    reset!(conopts)
    PNConstraint(n, m, con, sig, diffmethod, collect(inds), vals, jac, jacviews, active, conopts, [Z])
end

function evaluate_constraints!(pncon::PNConstraint)
    Z = pncon.Z[1]
    TO.evaluate_constraints!(pncon.sig, pncon.con, pncon.vals, Z, pncon.inds)
end

function constraint_jacobians!(pncon::PNConstraint)
    Z = pncon.Z[1]
    sig = function_signature(pncon)
    diff = pncon.diffmethod
    for (i,k) in enumerate(pncon.inds)
        jac = pncon.jac[i]
        jacview = pncon.jacviews[i]'
        RD.jacobian!(sig, diff, pncon.con, jac, pncon.vals[i], Z[k])
        jacview .= jac  # efficient copy to sparse array
    end
end

RD.vectype(pncon::PNConstraint) = RD.vectype(eltype(pncon.Z[1]))
settraj!(pncon::PNConstraint, Z::SampledTrajectory) = pncon.Z[1] = Z

function update_active_set!(pncon::PNConstraint)
    _update_active_set!(TO.sense(pncon.con), pncon)
    return nothing
end

_update_active_set!(::Equality, pncon::PNConstraint) = nothing
_update_active_set!(::SecondOrderCone, pncon::PNConstraint) = error("Cannot compute active set for SecondOrderCone constraints.") 

function _update_active_set!(::Inequality, pncon::PNConstraint)
    tol = pncon.opts.active_set_tolerance
    p = RD.output_dim(pncon.con)
    c, a = pncon.vals, pncon.active
    for i in eachindex(pncon.inds)
        for j = 1:p
            a[i][j] = c[i][j] > -tol
        end
    end
end

function _update_active_set!(::TO.PositiveOrthant, pncon::PNConstraint)
    tol = pncon.opts.active_set_tolerance
    p = RD.output_dim(pncon.con)
    c, a = pncon.vals, pncon.active
    for i in eachindex(pncon.inds)
        for j = 1:p
            a[i][j] = c[i][j] < tol
        end
    end
end

function reset!(pncon::PNConstraint)
    reset!(pncon.opts)
end