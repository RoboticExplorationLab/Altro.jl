using Altro
using TrajectoryOptimization
using Test
using StaticArrays
using ForwardDiff
using RobotDynamics
using LinearAlgebra
const RD = RobotDynamics
const TO = TrajectoryOptimization

function alcost(::TO.Equality, con, λ, μ, z)
    c = RD.evaluate(con, z) 
    Iμ = Diagonal(μ)
    λ'c + 0.5*c'Iμ*c
end

function alcost(::TO.Inequality, con, λ, μ, z)
    c = RD.evaluate(con, z) 
    a = (λ .> 0) .| (c .> 0)
    Iμ = Diagonal(a .* μ)
    λ'c + 0.5*c'Iμ*c
end

alcost(::TO.SecondOrderCone, con, λ, μ, z) = alcost(con, λ, μ, z)

function alcost(con, λ, μ, z) 
    cone = TO.dualcone(TO.sense(con))
    c = RD.evaluate(con, z) 
    λbar = λ .- μ .* c
    λp = TO.projection(cone, λbar)
    Iμ = Diagonal(inv.(μ))
    return 0.5*(λp'Iμ*λp - λ'Iμ*λ)
end

function algrad(::TO.Equality, con, λ, μ, ∇c, z)
    c = RD.evaluate(con, z) 
    λbar = λ .+ μ .* c
    ∇c'λbar 
end

function algrad(::TO.Inequality, con, λ, μ, ∇c, z)
    c = RD.evaluate(con, z) 
    a = (c .>= 0) .| (λ .> 0)
    λbar = λ .+ (a .* μ) .* c
    ∇c'λbar 
end

algrad(::TO.SecondOrderCone, con, λ, μ, ∇c, z) = algrad(con, λ, μ, ∇c, z)

function algrad(con, λ, μ, ∇c, z)
    cone = TO.dualcone(TO.sense(con))
    c = RD.evaluate(con, z) 
    λbar = λ .-  μ .* c
    λp = TO.projection(cone, λbar)
    Iμ = Diagonal(μ)
    p = length(λ)
    ∇proj = zeros(eltype(z), p,p)
    TO.∇projection!(cone, ∇proj, λbar)
    -∇c'Iμ*∇proj'*(Iμ\λp)
end

##
T = Float64
n, m, N = 3, 2, 11
tf = 2.0
xf = [1,2,3.]

con_eq = TO.GoalConstraint(xf, SA[1,3])
inds_eq = 1:10
confun_eq(x) = (x - xf)[[1,3]]

xc, yc, rad = [4,2,2,0.], [0,1,2,3.], [0.2, 0.1, 0.3, 0.4]
con_in = TO.CircleConstraint(n, xc, yc, rad)
inds_in = 2:6
confun_in(x) = rad.^2 .- (x[1] .- xc).^2 .- (x[2] .- yc).^2

con_so = TO.NormConstraint(n, m, 3, TO.SecondOrderCone(), :state)
inds_so = N:N 
confun_so(x) = [x; 3]

# Other data needed for constructor
Z = Traj(randn(n,N), randn(m,N-1), dt=0.1)
opts = SolverOptions()
costs = zeros(N)

##
@testset "$(TO.conename(TO.sense(con))) ALConstraint" for (con,inds,confun) in [
    (con_eq, inds_eq, confun_eq),
    (con_in, inds_in, confun_in),
    (con_so, inds_so, confun_so),
]
# con, inds, confun = 
#     (con_so, inds_so, confun_so)
alcon = Altro.ALConstraint{T}(Z, con, inds, costs, solveropts=opts)

# Test that it got filled in correctly
p = TO.output_dim(alcon.con)
P = length(inds) 
@test alcon.n == n
@test alcon.m == m
@test alcon.inds == inds 
for field in (:vals, :λ, :μ, :μinv, :λbar, :λproj, :λscaled, :viol)
    @test all(x->length(x)==p, getfield(alcon, field))
    @test length(getfield(alcon, field)) == P
end
w = RD.input_dim(con)
@test size(alcon.c_max) == (N,)
@test length(alcon.jac) == P
@test all(x->size(x)==(p,w), alcon.jac)
@test length(alcon.∇proj) == P
@test all(x->size(x)==(p,p), alcon.∇proj)
@test length(alcon.∇²proj) == P
@test all(x->size(x)==(p,p), alcon.∇²proj)
@test length(alcon.grad) == P
@test all(x->size(x)==(n+m,), alcon.grad)
@test length(alcon.hess) == P
@test all(x->size(x)==(n+m,n+m), alcon.hess)
@test size(alcon.tmp_jac) == (p,w)
@test all(x->x == ones(p), alcon.μ)

# Test evaluation methods
Altro.settraj!(alcon, Z)
Altro.evaluate_constraints!(alcon)
c = map(Z[inds]) do z
    confun(TO.state(z))
    # (TO.state(z) - xf)[[1,3]]
end
@test c ≈ alcon.vals

Altro.constraint_jacobians!(alcon)
C = map(Z[inds]) do z
    J = ForwardDiff.jacobian(confun, TO.state(z))
    [J zeros(p,w-n)]
end
@test C ≈ alcon.jac

# Set non-zero duals
for i = 1:P
    alcon.λ[i] .= randn(p)
end
λ = alcon.λ
μ = alcon.μ

# Test alcost
alcon.opts.use_conic_cost = false 
cone = TO.sense(con)

Altro.evaluate_constraints!(alcon)
for i = 1:P
    Altro.alcost(alcon, i)
    Altro.algrad!(cone, alcon, i)
    z = Z[inds[i]]
    _z = RD.StaticKnotPoint{n,m}(MVector{n+m}(z.z), z.t, z.dt)
    grad = ForwardDiff.gradient(
        x->alcost(cone, alcon.con, SVector{p}(λ[i]), SVector{p}(μ[i]), x), 
        _z
    )
    @test grad ≈ alcon.grad[i]
    @test alcon.grad[i][1:n] ≈ algrad(
        cone, alcon.con, SVector{p}(λ[i]), SVector{p}(μ[i]), alcon.jac[i], _z
    )[1:n]
end

for i = 1:P
    cone = TO.sense(alcon.con)
    Altro.alhess!(cone, alcon, i)
    hess = zeros(n+m, n+m)
    # Get Gauss-Newton approximation
    z = Z[inds[i]]
    _z = RD.StaticKnotPoint{n,m}(MVector{n+m}(z.z), z.t, z.dt)
    ix = Altro.getinputinds(alcon)
    jac = ForwardDiff.jacobian(
        x->algrad(cone, alcon.con, SVector{p}(λ[i]), SVector{p}(μ[i]), alcon.jac[i], x), 
        _z
    )
    hess[ix,:] .= jac
    @test hess[ix,ix] ≈ alcon.hess[i][ix,ix]
end

# Use Conic cost
Altro.evaluate_constraints!(alcon)
alcon.opts.use_conic_cost = true
for i = 1:P
    z = Z[inds[i]]
    _z = RD.StaticKnotPoint{n,m}(MVector{n+m}(z.z), z.t, z.dt)
    @test Altro.alcost(alcon, i) ≈ alcost(alcon.con, SVector{p}(λ[i]), SVector{p}(μ[i]), _z)
end

Altro.algrad!(alcon, 1)
for i = 1:P
    z = Z[inds[i]]
    _z = RD.StaticKnotPoint{n,m}(MVector{n+m}(z.z), z.t, z.dt)
    grad = ForwardDiff.gradient(
        x->alcost(alcon.con, SVector{p}(λ[i]), SVector{p}(μ[i]), x), _z
    )
    Altro.algrad!(alcon, i)
    @test alcon.grad[i] ≈ grad
    grad = algrad(alcon.con, SVector{p}(λ[i]), SVector{p}(μ[i]), alcon.jac[i], _z)
    @test alcon.grad[i][1:n] ≈ grad[1:n]
end

let i = P
    z = Z[inds[i]]
    _z = RD.StaticKnotPoint{n,m}(MVector{n+m}(z.z), z.t, z.dt)
    Altro.alhess!(alcon, i)
    hess = zeros(n+m, n+m)
    jac = ForwardDiff.jacobian(x->algrad(alcon.con, SVector{p}(λ[i]), SVector{p}(μ[i]), 
        alcon.jac[i], x), _z
    )
    iz = Altro.getinputinds(alcon)
    hess[iz,:] .= jac
    @test alcon.hess[i] ≈ hess
end

## Dual and penalty updates
alcon.opts.use_conic_cost = false 
λ0 = deepcopy(alcon.λ)
Altro.alcost(alcon)
Altro.dualupdate!(alcon)
for i = 1:P
    if cone == TO.Equality()
        @test λ[i] ≈ λ0[i] + μ[i] .* c[i]
    elseif cone == TO.Inequality()
        @test λ[i] ≈ clamp.(λ0[i] + μ[i] .* c[i], 0, Inf)
    else
        @test λ[i] ≈ alcon.λproj[i]
    end
    λ[i] .= λ0[i]  # reset them to what they were before
end

alcon.opts.use_conic_cost = true 
Altro.dualupdate!(alcon)
for i = 1:P
    @test λ[i] ≈ alcon.λproj[i] 
end

Altro.reset_penalties!(alcon)
Altro.penaltyupdate!(alcon)
@test Altro.max_penalty(alcon) == 10.0

Altro.reset_penalties!(alcon)
@test Altro.max_penalty(alcon) == 1.0

Altro.reset_duals!(alcon)
@test norm(λ) ≈ 0

end