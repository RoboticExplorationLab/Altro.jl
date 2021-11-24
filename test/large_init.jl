using Test
using Altro
using BenchmarkTools
using TrajectoryOptimization
using RobotDynamics
using RobotZoo
using StaticArrays, LinearAlgebra
using ForwardDiff
using FiniteDiff
const RD = RobotDynamics
const TO = TrajectoryOptimization

## Create a relatively large problem
N = 101
tf = 5.0
dt = tf / (N - 1)
model = RobotZoo.DoubleIntegrator(32)
n,m = size(model)
Q = Diagonal(@SVector fill(1.0, n))
R = Diagonal(@SVector fill(1e-2, m))
Qf = Q * (N - 1)
xf = @SVector fill(10.0, n)
x0 = @SVector zeros(n)

obj = LQRObjective(Q, R, Qf, xf, N) 
prob = Problem(model, obj, xf, tf, x0 = x0)

##
x,u = rand(model)
z = KnotPoint(x,u, 0.1)
ix, iu = z._x, z._u
xn = zeros(n)

f(s) = RobotDynamics.discrete_dynamics(RK4, model, s[ix], s[iu], 0.0, 0.1)
function f!(xn, s)
    xn .= f(s)
end
@generated function dynamics!(xn, di::RobotZoo.DoubleIntegrator{N,M}, x, u) where {N,M}
    vel = [:(xn[$(i - M)] = x[$i]) for i = M+1:N]
    us = [:(xn[$(i + M)] = u[$i] + di.gravity[$i]) for i = 1:M]

    quote
        $(Expr(:block, vel...)) 
        $(Expr(:block, us...)) 
    end
end
f2!(xn, s) = dynamics!(xn, model, s[ix], s[iu])

function jac!(model, ∇f, xn, z, fd!, cfg)
    ix, iu, dt = z._x, z._u, z.dt
    t = z.t
    # fd!(xn, s) = dynamics!(xn, model, view(s, ix[1]:ix[end]), view(s, iu[1]:iu[end]))
    # cfg = ForwardDiff.JacobianConfig(fd!, xn, z.z)
    ForwardDiff.jacobian!(∇f, fd!, xn, z.z, cfg)
    # fd!(xn, z.z)
    return nothing
end

function jac_fd!()

function gen_fd(model, z)
    ix, iu = z._x, z._u
    fd!(xn, s) = dynamics!(xn, model, view(s, ix[1]:ix[end]), view(s, iu[1]:iu[end]))
    return fd!
end

function gen_fd_RD(model, z)
    ix, iu = z._x, z._u
    fd!(xn, s) = begin
        xn .= RobotDynamics.dynamics(model, s[ix], s[iu])
    end
    return fd!
end

fd2! = gen_fd(model, z)
cfg = ForwardDiff.JacobianConfig(fd2!, xn, z.z)

fd_RD! = gen_fd_RD(model, z)
cfg_RD = ForwardDiff.JacobianConfig(fd_RD!, xn, z.z)

dynamics!(xn, model, x, u)
f2!(xn, z.z)

fd!(xn, s) = dynamics!(xn, model, view(s, ix[1]:ix[end]), view(s, iu[1]:iu[end]))

∇f = zeros(n, n+m)
xn .= 0
@time jac!(model, ∇f, xn, z, fd2!, cfg)
@time jac!(model, ∇f, xn, z, fd_RD!, cfg_RD)
@btime jac!($model, $∇f, $xn, $z, $fd2!, $cfg)
@btime jac!($model, $∇f, $xn, $z, $fd_RD!, $cfg_RD)
@btime jac!($model, $∇f, $xn, $z)
@btime dynamics!($xn, $model, $x, $u)

@time ∇f = ForwardDiff.jacobian(f, z.z)
∇f2 = zeros(n, n+m)
xn = zeros(n)
@time ForwardDiff.jacobian!(∇f2, f, z.z)
@time ForwardDiff.jacobian!(∇f2, f!, xn, z.z) 

@time ForwardDiff.jacobian!(∇f2, f!, xn, z.z) 
@time ForwardDiff.jacobian!(∇f2, f2!, xn, z.z) 

@btime ForwardDiff.jacobian!($∇f2, $f!, $xn, $(z.z)) 

##
t = @elapsed altrosolver = ALTROSolver(prob)
@test t < 60  # it should finish initializing the solver in under a minute (usually about 8 seconds on a desktop)

##
# solver = ALTROSolver(Problems.Pendulum()...)
solver = Altro.get_ilqr(altrosolver)
solver.opts.verbose = 2
solver.opts.static_bp = false
solver = ilqr

@time Altro.initialize!(solver)

Z = solver.Z; Z̄ = solver.Z̄;

n,m,N = size(solver)
J = Inf
_J = TO.get_J(solver.obj)
J_prev = sum(_J)
grad_only = false
@time TO.state_diff_jacobian!(solver.G, solver.model, solver.Z)
@time TO.dynamics_expansion!(TO.integration(solver), solver.D, solver.model, solver.Z, solver.cache)
@time TO.error_expansion!(solver.D, solver.model, solver.G)
@time TO.cost_expansion!(solver.quad_obj, solver.obj, solver.Z, solver.exp_cache, init=true, rezero=true)
TO.error_expansion!(solver.E, solver.quad_obj, solver.model, solver.Z, solver.G)
@time ΔV = Altro.backwardpass!(solver)
forwardpass!(solver, ΔV, J)