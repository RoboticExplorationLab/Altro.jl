using Altro
using TrajectoryOptimization
using ForwardDiff
using RobotDynamics
using BenchmarkTools
using StaticArrays
using LinearAlgebra
const TO = TrajectoryOptimization
const RD = RobotDynamics

Base.@kwdef struct Car <: AbstractModel
    d::Float64 = 2.0  # distance between axles
end

function RD.discrete_dynamics(::Type{RD.Euler}, model::Car, x, u, t, dt)
    d = model.d
    w = u[1]  # front wheel angle
    a = u[2]  # front wheel acceleration
    o = x[3]  # car angle
    z = SVector{2}(reverse(sincos(o)))
    v = x[4]  # front wheel velocity
    f = dt*v  # front wheel rolling distance
    b = d + f .* cos(w) - sqrt(d^2 - (f * sin(w))^2 )
    d2 = asin(sin(w).*f/d)
    dy = [b .* z; d2; dt*a]
    y = x + dy
    return y
end
RD.state_dim(::Car) = 4
RD.control_dim(::Car) = 2

model = Car()
n,m = size(model)
N = 501
dt = 0.03
tf = (N-1)*dt
x0 = SA[1,1,pi*3/2,0.1]
u0 = SA[0.1,0.1]
xf = SA[0,0,0,0.]
Q = Diagonal(SA[1,1,0,0])*1e-3/dt*2
R = Diagonal(SA[1, 0.01])*1e-2/dt*2
Qf = Diagonal(SA[0.1 + 1e-3, 0.1 + 1e-3, 1.0 ,0.3])*2
obj = LQRObjective(Q,R,Qf,xf,N)

prob = Problem(model, obj, xf, tf, x0=x0, N=N, integration=RD.Euler)
initial_controls!(prob, u0)
ilqr = Altro.iLQRSolver(prob, show_summary=true, ddp=true, verbose=2, save_S=true)
ilqr.opts.bp_reg_initial = 1.0
ilqr.opts.bp_reg_min = 1e-3 
# solve!(ilqr)
Altro.initialize!(ilqr)
Altro.step!(ilqr, cost(ilqr), expansion_only=false)
ilqr.K[end]
ilqr.S[475].Q

ilqr.Q[250].R
ilqr.D[N-1].fxx
ilqr.D[N-1].fuu
ilqr.S[end].q

# Altro.step!(ilqr, cost(ilqr), expansion_only=true)
# solve!(ilqr)
ilqr.E[N].q
ilqr.S[N].q
Qf*states(ilqr)[end]
x = states(ilqr)[end]
dx = [0,0,1e-10,0]
J1 = TO.stage_cost(obj[end], x)
J2 = TO.stage_cost(obj[end], x+dx)
(J2-J1)/1e-10
ilqr.S[N].q


prob,opts = Altro.Problems.DubinsCar(:parallel_park)
# prob,opts = Altro.Problems.Cartpole()
# solver = ALTROSolver(prob,opts, show_summary=true, ddp=true, verbose=2, gradient_tolerance=1e-2)
# ilqr = Altro.get_ilqr(solver)
solver = Altro.iLQRSolver(prob,opts, show_summary=true, ddp=true, verbose=2, gradient_tolerance=1e-2)

solve!(solver)
ilqr.D[2].fxx
ilqr.Q[3].r
controls(solver.Z̄)
norm(states(solver.Z̄),Inf)

Altro.initialize!(ilqr)
Altro.step!(ilqr, cost(ilqr))
@btime Altro.step!($ilqr, $(cost(ilqr)))
Altro.copy_trajectories!(ilqr)
ilqr.D[2].fux
ilqr.D[2].∇²f

function ∇jacobian!(∇f, model::AbstractModel, z, b)
    ix,iu = z._x, z._u
    t = z.t
    f_aug(z) = dynamics(model, z[ix], z[iu], t)'b
    ForwardDiff.hessian!(∇f, f_aug, z.z)
end
model = prob.model
n,m = size(model)
F2 = zeros(n+m,n+m)
F = zeros(n,n+m)
b = @SVector rand(n)


@btime ∇jacobian!($F2, $model, $(prob.Z[1]), $b)
@btime jacobian!($F, $model, $(prob.Z[1]))
@btime RobotDynamics.∇discrete_jacobian!(RK3, $F2, $model, $(prob.Z[1]), $b)

RobotDynamics.∇discrete_jacobian!(RK3, F2, model, (prob.Z[1]), b)
RobotDynamics.∇discrete_jacobian!(RK3, F2, model, (prob.Z[1]), b)