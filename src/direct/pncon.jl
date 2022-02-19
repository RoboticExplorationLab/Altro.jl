struct PNConstraint{T, C<:TO.StageConstraint}
    n::Int  # state dimension
    m::Int  # control dimension
    con::C
    sig::FunctionSignature
    diffmethod::DiffMethod
    inds::Vector{Int}             # knot point indices for constraint
    vals::Vector{VectorView{T,Int}}
    jac::Vector{SparseView{T,Int}}
    active::Vector{SubArray{Bool, 1, BitVector, Tuple{UnitRange{Int}}, true}}
end

function PNConstraint(n::Int, m::Int, N::Int, con::TO.StageConstraint, 
                      inds::AbstractVector{<:Integer}, D, d, a, cinds; 
                      sig::FunctionSignature=RD.default_signature(con), 
                      diffmethod::DiffMethod=RD.default_diffmethod(con))
    p = RD.output_dim(con)
    P = length(inds)
    nm = n + m
    if TO.sense(con) == TO.SecondOrderCone()
        error("ProjectedNewtonSolver doesn't support SecondOrderCone constraints.")
    end

    @assert length(cinds) == P
    @assert all(x->length(x) == p, cinds)

    xi(k) = (k-1)*(n+m) .+ (1:n)
    ui(k) = n + (k-1)*(n+m) .+ (1:m)
    zi(k) = (k-1)*(n+m) .+ (k == N ? (1:n) : (1:n+m))
    vals = [view(d, ci) for ci in cinds]
    jac = [view(D, cinds[i], zi(k)) for (i,k) in enumerate(inds)]
    active = [view(a, ci) for ci in cinds]

    PNConstraint(n, m, con, sig, diffmethod, collect(inds), vals, jac, active)
end

function evaluate_constraint!(pncon::PNConstraint, Z::AbstractTrajectory)
    for (i,k) in enumerate(pncon.inds)
        TO.evaluate_constraint!(pncon.sig, pncon.con, pncon.vals[i], Z[k])
    end
end

function constraint_jacobian!(pncon::PNConstraint, Z::AbstractTrajectory)
    for (i,k) in enumerate(pncon.inds)
        RD.jacobian!(pncon.sig, pncon.diffmethod, pncon.con, pncon.jac[i], pncon.vals[i], Z[k])
    end
end

@inline update_active_set!(pncon::PNConstraint, tol) = update_active_set!(TO.sense(pncon.con), pncon, tol)
update_active_set!(::Equality, pncon::PNConstraint, tol) = nothing
update_active_set!(::SecondOrderCone, pncon::PNConstraint, tol) = error("Cannot compute active set for SecondOrderCone constraints.") 

function update_active_set!(::Inequality, pncon::PNConstraint, tol)
    p = RD.output_dim(pncon.con)
    c, a = pncon.vals, pncon.active
    for i in eachindex(pncon.inds)
        for j = 1:p
            a[i][j] = c[i][j] > -tol
        end
    end
end

function update_active_set!(::TO.PositiveOrthant, pncon::PNConstraint, tol)
    p = RD.output_dim(pncon.con)
    c, a = pncon.vals, pncon.active
    for i in eachindex(pncon.inds)
        for j = 1:p
            a[i][j] = c[i][j] < tol
        end
    end
end