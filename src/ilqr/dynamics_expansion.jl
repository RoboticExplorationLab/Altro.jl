"""
    DynamicsExpansion

Stores the expansion of the dynamics about a point. Stores the jacobians for 
both the full state and error state. The full state Jacobians are accessed via 
`D.A` and `D.B`. The error state Jacobians are accessed via `D.fx` and `D.fu`.

# Constructor

    DynamicsExpansion{T}(n, e, m)

where `n` is the state dimension, `e` is the error state dimension, and `m` is 
the control dimension.
"""
struct DynamicsExpansion{T}
    f::Vector{T}   # (Nx,)
    ∇f::Matrix{T}  # (Nx, Nx+Nu)
    A::SubArray{T,2,Matrix{T},Tuple{ColonSlice, UnitRange{Int}}, true}
    B::SubArray{T,2,Matrix{T},Tuple{ColonSlice, UnitRange{Int}}, true}
    ∇e::Matrix{T}
    fx::SubArray{T,2,Matrix{T},Tuple{ColonSlice, UnitRange{Int}}, true}
    fu::SubArray{T,2,Matrix{T},Tuple{ColonSlice, UnitRange{Int}}, true}
    tmp::Matrix{T}  # (Nx,Ne)
    function DynamicsExpansion{T}(n, e, m; errstate::Bool=n!=e) where T
        f = zeros(T,n)
        ∇f = zeros(T,n,n+m)
        A = view(∇f, :, 1:n)
        B = view(∇f, :, n+1:n+m)
        if errstate
            ∇e = zeros(T,e,e+m)
        else  # alias the error expansion to avoid extra work
            ∇e = ∇f
        end
        fx = view(∇e, :, 1:e)
        fu = view(∇e, :, e+1:e+m)
        tmp = zeros(T,n,e)
        new{T}(f,∇f, A,B, ∇e,fx,fu, tmp)
    end
end

@inline function RD.jacobian!(sig::InPlace, diff::DiffMethod, model::DiscreteDynamics,
    D::DynamicsExpansion, z::KnotPoint{n,m}
) where {n,m}
    RD.jacobian!(sig, diff, model, D.∇f, D.f, z)
end

@inline function RD.jacobian!(sig::StaticReturn, diff::DiffMethod, model::DiscreteDynamics,
    D::DynamicsExpansion, z::KnotPoint{n,m}
) where {n,m}
    _z = SVector{n+m}(RD.getdata(z))
    RD.jacobian!(sig, diff, model, D.∇f, D.f, RD.StaticKnotPoint(z, _z))
end

"""
    errstate_jacobian!(model, G, Z)

Calculate the error state Jacobians for the trajectory `Z`, storing the 
    result in `G`.
"""
function errstate_jacobians!(model::Vector{<:DiscreteDynamics}, G, Z)
    # NOTE: assumes all models have the same statevectortype
    errstate_jacobians!(RD.statevectortype(model[1]), model, G, Z)
end

function errstate_jacobians!(::RD.EuclideanState, model::Vector{<:DiscreteDynamics}, G, Z)
    return nothing
end

function errstate_jacobians!(::RD.StateVectorType, models::Vector{<:DiscreteDynamics}, G, Z)
    N = length(Z)
	for k in eachindex(Z)
        model = models[min(k, N-1)]
		G[k] .= 0
		RD.errstate_jacobian!(RD.statevectortype(model), model, G[k], Z[k])
	end
end

# """
#     dynamics_expansion!(sig, diff, model, D, Z)

# Evaluate the expansion of the dynamics about the trajectory `Z`, storing the 
# result in `D`.
# """
# function dynamics_expansion!(sig::FunctionSignature, diff::DiffMethod, 
#                              model::DiscreteDynamics, D::Vector{<:DynamicsExpansion}, Z::Traj)
#     for k in eachindex(D)
#         RobotDynamics.jacobian!(sig, diff, model, D[k].∇f, D[k].f, Z[k])
#     end
# end

"""
    error_expansion!(model, D, G)

Evaluate the dynamics error state expansion. Assumes the error state Jacobians `G`
have already been calculated using [`errstate_jacobians!`](@ref) and that the 
full state Jacobians have been calculated and stored in `D` using 
[`dynamics_expansion`](@ref).
"""
error_expansion!(model::Vector{<:DiscreteDynamics}, D::Vector{<:DynamicsExpansion}, G) = 
    _error_expansion!(RD.statevectortype(model[1]), D, model, G)

_error_expansion!(::RD.EuclideanState, D::Vector{<:DynamicsExpansion}, 
                 model::Vector{<:DiscreteDynamics}, G) = nothing

function _error_expansion!(::RD.RotationState, D::Vector{<:DynamicsExpansion}, 
    model::Vector{<:DiscreteDynamics}, G
)
    for k in eachindex(D)
        _error_expansion!(D[k], G[k], G[k+1])
    end
end

function _error_expansion!(D, G1, G2)
    matmul!(D.tmp, D.A, G1)  # D.tmp = A * G1
    matmul!(D.fx, G2', D.tmp)   # D.fx = G2'A*G1
    matmul!(D.fu, G2', D.B)
end