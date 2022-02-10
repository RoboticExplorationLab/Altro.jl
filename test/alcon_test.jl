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
    -∇c'∇proj'Iμ*λp
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

##
@testset "$(TO.conename(TO.sense(con))) ALConstraint" for (con,inds,confun) in [
    (con_eq, inds_eq, confun_eq),
    (con_in, inds_in, confun_in),
    (con_so, inds_so, confun_so),
]
# con, inds, confun = 
#     (con_in, inds_in, confun_in)
alcon = Altro.ALConstraint{T}(n, m, con, inds)

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
@test size(alcon.c_max) == (P,)
@test length(alcon.jac) == P
@test all(x->size(x)==(p,n+m), alcon.jac)
@test length(alcon.∇proj) == P
@test all(x->size(x)==(p,p), alcon.∇proj)
@test length(alcon.∇²proj) == P
@test all(x->size(x)==(p,p), alcon.∇²proj)
@test length(alcon.grad) == P
@test all(x->size(x)==(n+m,), alcon.grad)
@test length(alcon.hess) == P
@test all(x->size(x)==(n+m,n+m), alcon.hess)
@test size(alcon.tmp_jac) == (p,n+m)
@test all(x->x == ones(p), alcon.μ)

# Generate a random trajectory
X = [randn(T,n) for k = 1:N]
U = [randn(T,m) for k = 1:N-1]
Z = Traj(X, U, tf=2.0)

# Test evaluation methods
Altro.evaluate_constraint!(alcon, Z)
c = map(Z[inds]) do z
    confun(TO.state(z))
    # (TO.state(z) - xf)[[1,3]]
end
@test c ≈ alcon.vals

Altro.constraint_jacobian!(alcon, Z)
C = map(Z[inds]) do z
    [ForwardDiff.jacobian(confun, TO.state(z)) zeros(p,m)]
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

for i = 1:P
    Altro.algrad!(cone, alcon, i)
    grad = ForwardDiff.gradient(x->alcost(cone, alcon.con, λ[i], μ[i], x), Z[inds[i]])
    @test grad ≈ alcon.grad[i]
    @test alcon.grad[i] ≈ algrad(cone, alcon.con, λ[i], μ[i], alcon.jac[i], Z[inds[i]])
end

for i = 1:P
    cone = TO.sense(alcon.con)
    Altro.alhess!(cone, alcon, i)
    hess = zeros(n+m, n+m)
    # Get Gauss-Newton approximation
    ForwardDiff.jacobian!(hess, x->algrad(cone, alcon.con, λ[i], μ[i], alcon.jac[i], x), Z[inds[i]])
    @test hess ≈ alcon.hess[i]
end

# Use Conic cost
Altro.evaluate_constraint!(alcon, Z)
alcon.opts.use_conic_cost = true
for i = 1:P
    @test Altro.alcost(alcon, i) ≈ alcost(alcon.con, λ[i], μ[i], Z[inds[i]]) 
end

Altro.algrad!(alcon, 1)
for i = 1:P
    grad = ForwardDiff.gradient(x->alcost(alcon.con, λ[i], μ[i], x), Z[inds[i]])
    Altro.algrad!(alcon, i)
    @test alcon.grad[i] ≈ grad
    grad = algrad(alcon.con, λ[i], μ[i], alcon.jac[i], Z[inds[i]])
    @test alcon.grad[i] ≈ grad
end

let i = P
    Altro.alhess!(alcon, i)
    hess = zeros(n+m, n+m)
    ForwardDiff.jacobian!(hess, x->algrad(alcon.con, λ[i], μ[i], alcon.jac[i], x), Z[inds[i]])
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