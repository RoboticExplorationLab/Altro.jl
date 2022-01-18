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

obj = LQRObjective(Q, R, Qf, xf, N) 
prob = Problem(model, obj, x0, tf, xf = xf)

# Initialize the solver
t = @elapsed altrosolver = ALTROSolver(prob)
@test t < 60  # it should finish initializing the solver in under a minute (usually about 8 seconds on a desktop)

# Test the iLQR solve
solver = Altro.get_ilqr(altrosolver)
solver.opts.verbose = 2
solver.opts.static_bp = false
solver.opts.dynamics_funsig = RD.InPlace()
z = solver.Z[1]

t_solve = @elapsed solve!(solver)
@test t_solve < 60*3  # should be about 40 seconds