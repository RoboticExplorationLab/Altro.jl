"""
    StateControlExpansion

Holds all of the terms for the expansion of the cost function at a single time step.
Provides views for the individual components for both the state and control. Internally,
the data is all store in a contiguous block of memory as `[hess grad]`.

# Constructors
An expansion over both states and control is constructed using 

    StateControlExpansion{T}(n,m)

Whereas an expansion over just the states can be created using

    StateControlExpansion{T}(n)

# Usage
Given a `StateControlExpansion` `E`, we can pull out the pieces of the gradient using

    E.grad  # (n+m,) gradient
    E.x     # (n,) state gradient
    E.u     # (u,) state gradient

We can similarly extract the pieces for the Hessian:

    E.hess # (n+m,n+m) Hessian
    E.xx # (n,n) state Hessian 
    E.uu # (m,m) control Hessian 
    E.ux # (m,n) Hessian cross-term 
"""
struct StateControlExpansion{T}
    data::Matrix{T}
    hess::SubArray{T, 2, Matrix{T}, Tuple{Base.Slice{Base.OneTo{Int64}}, UnitRange{Int64}}, true}
    grad::SubArray{T, 1, Matrix{T}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}
    xx::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
    x::SubArray{T,1,Matrix{T},Tuple{UnitRange{Int},Int},true}
    uu::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
    ux::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
    u::SubArray{T,1,Matrix{T},Tuple{UnitRange{Int},Int},true}
    function StateControlExpansion{T}(n,m) where T
        ix,iu = 1:n, n .+ (1:m)
        data = zeros(T,n+m,n+m+1)
        hess = view(data, :, 1:n+m)
        grad = view(data, :, n+m+1)
        xx = view(hess, ix, ix)
        ux = view(hess, iu, ix)
        uu = view(hess, iu, iu)
        x = view(grad, ix)
        u = view(grad, iu)
        hess .= I(n+m)
        new{T}(data, hess, grad, xx,x,uu,ux,u)
    end
    function StateControlExpansion{T}(n) where T
        ix = 1:n
        data = zeros(T,n,n+1)
        hess = view(data, :, 1:n)
        grad = view(data, :, n+1)
        xx = view(hess, ix, ix)
        x = view(grad, ix)
        hess .= I(n)
        new{T}(data, hess, grad, xx,x)
    end
end

"""
    CostExpansion

Represents the expansion of the cost function over the time horizon.
"""
struct CostExpansion{T} <: AbstractVector{StateControlExpansion{T}}
    data::Vector{StateControlExpansion{T}}
    const_hess::BitVector
    const_grad::BitVector
    function CostExpansion{T}(n, m, N) where T
        data = [StateControlExpansion{T}(n,m) for k = 1:N]
        const_hess = BitVector(undef, N)
        const_grad = BitVector(undef, N)
        new{T}(data, const_hess, const_grad)
    end
end
@inline CostExpansion(n,m,N) = CostExpansion{Float64}(n,m,N)

# Array interface
Base.size(E::CostExpansion) = size(E.data) 
Base.getindex(E::CostExpansion, i::Int) = Base.getindex(E.data, i)

"""
    FullStateExpansion(E::CostExpansion, model::DiscreteDynamics)

Create a vector of expansions over the state and control, for the full 
state vector `x`, given the `E`, the expansion on the error state.
"""
FullStateExpansion(E::CostExpansion, model::DiscreteDynamics) = 
    FullStateExpansion(RD.statevectortype(model), E, model)

function FullStateExpansion(::RD.EuclideanState, E::CostExpansion, model::DiscreteDynamics)
    # Create a CostExpansion linked to error cost expansion
    @assert RobotDynamics.errstate_dim(model) == state_dim(model) 
    return E 
end

function FullStateExpansion(
    ::RD.RotationState, E::CostExpansion{T}, model::DiscreteDynamics
) where T
    # Create an expansion for the full state dimension
    @assert length(E[1].x) == RobotDynamics.errstate_dim(model)
    n0 = state_dim(model)
    m = control_dim(model)
    return CostExpansion{T}(n0,m,length(E))
end

"""
    cost_expansion!(obj, E, Z)

Evaluate the expansion of the objective `obj` about the trajectory `Z`, storing 
the result in `E`.
"""
function cost_expansion!(obj::Objective, E::CostExpansion, Z)
    for k in eachindex(Z)
        RD.gradient!(obj.diffmethod[k], obj.cost[k], E[k].grad, Z[k])
        RD.hessian!(obj.diffmethod[k], obj.cost[k], E[k].hess, Z[k])
    end
end

"""
    error_expansion!(model, Eerr, Efull, G, Z)

Evaluate the error state expansion for the cost function about the trajectory `Z`,
    given the cost expansion stored in `Efull` computed using [`cost_expansion!`](@ref)
    and the error state jacobians `G` computed using [`errstate_jacobians!`](@ref).
    The result is stored in `Eerr`.
"""
function error_expansion!(
    model::DiscreteDynamics, Eerr::CostExpansion, Efull::CostExpansion, G, Z
)
    _error_expansion!(RD.statevectortype(model), model, Eerr, Efull, G, Z)
end

function _error_expansion!(::RD.EuclideanState, model, Eerr, Efull, G, Z) 
    @assert Eerr === Efull
    return nothing
end

function _error_expansion!(::RD.RotationState, model, Eerr, Efull, G, Z)
    for k in eachindex(Eerr)
        _error_expansion!(model, Eerr[k], Efull[k], G[k], G[end], Z[k])
    end
end

function _error_expansion!(model::DiscreteDynamics, E, cost, G, tmp, z)
    E.xx .= 0
    E.uu .= cost.uu
    E.u .= cost.u
    RD.∇errstate_jacobian!(model, E.xx, state(z), cost.x)
    matmul!(E.ux, cost.ux, G)
    matmul!(E.x, G', cost.x)
    matmul!(tmp, cost.xx, G)
    matmul!(E.xx, G', tmp, 1.0, 1.0)
    return nothing
end