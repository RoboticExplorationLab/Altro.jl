using Test
using Altro
using BenchmarkTools
using TrajectoryOptimization
using RobotDynamics
using RobotZoo
using StaticArrays, LinearAlgebra
using ForwardDiff
using FiniteDiff
using Logging
const RD = RobotDynamics
const TO = TrajectoryOptimization

## Create a relatively large problem
N = 101
tf = 5.0
dt = tf / (N - 1)
model = RobotZoo.DoubleIntegrator(50)
n,m = size(model)
Q = Diagonal(@SVector fill(1.0, n))
R = Diagonal(@SVector fill(1e-2, m))
Qf = Q * (N - 1)
xf = @SVector fill(10.0, n)
x0 = @SVector zeros(n)
cons = ConstraintList(n,m,N)
obs = CircleConstraint(n, fill(4.0,50), rand(50)*10, fill(0.1,50))
goal = GoalConstraint(xf)
# unorm = NormConstraint(n, m, 1000, TO.SecondOrderCone(), :control)
# add_constraint!(cons, unorm, 1:N-1)
add_constraint!(cons, obs, 2:N-1)
add_constraint!(cons, goal, N)

obj = LQRObjective(Q, R, Qf, xf, N)
# prob = Problem(model, obj, x0, tf, xf = xf)
prob = Problem(model, obj, x0, tf, xf = xf, constraints=cons)

# Initialize the solver
t = @elapsed altrosolver = ALTROSolver2(prob, dynamics_funsig=RD.InPlace())
@test t < 60  # it should finish initializing the solver in under a minute (usually about 8 seconds on a desktop)
t

##
altrosolver.opts.verbose = 2
altrosolver.opts.static_bp = false
altrosolver.opts.projected_newton = false
altrosolver.opts.trim_stats = false

solver = altrosolver.solver_al
conset = get_constraints(solver)
J_prev = cost(solver)

ilqr = Altro.get_ilqr(altrosolver)
# t_step = @elapsed Altro.step!(ilqr, J_prev)
t_step = @elapsed let solver = ilqr
    J_prev = cost(solver)
    Altro.errstate_jacobians!(solver.model, solver.G, solver.Z)
    Altro.dynamics_expansion!(solver)
    Altro.error_expansion!(solver.model, solver.D, solver.G)
    Altro.cost_expansion!(solver.obj, solver.Efull, solver.Z)
    Altro.error_expansion!(solver.model, solver.Eerr, solver.Efull, solver.G, solver.Z)

    # Get next iterate
    Altro.backwardpass!(solver)
    Jnew = Altro.forwardpass!(solver, J_prev)
end
@test t_step < 45*10  # Should be about 45 seconds.

##
t_solve = @elapsed solve!(altrosolver)
@test t_solve < 120  # should be close to 12 seconds
t_solve

## Projected Newton
pn = altrosolver.solver_pn
t_pn = @elapsed solve!(pn)
@test t_pn < 10*17  # should be about 17 seconds