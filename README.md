![CI](https://github.com/RoboticExplorationLab/ALTRO.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/RoboticExplorationLab/ALTRO.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RoboticExplorationLab/ALTRO.jl)

# ALTRO.jl
Implementation of the Augmented Lagrangian TRajectory Optimizer (ALTRO) solver, a very fast solver for constrained trajectory optimization problems. 
ALTRO uses iterative LQR (iLQR) with an augmented Lagrangian framework and can solve problems with nonlinear inequality and equality path constraints 
and nonlinear dynamics. 

ALTRO.jl solves trajectory optimization problems set up using [TrajectoryOptimization.jl](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl). 

For details on the solver, see the original [conference paper](https://rexlab.stanford.edu/papers/altro-iros.pdf) or related 
[tutorial](https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf).

## Simple Example
```julia
# Set up problem using TrajectoryOptimization.jl and RobotZoo.jl
using TrajectoryOptimization
using ALTRO
import RobotZoo.Cartpole
using StaticArrays, LinearAlgebra

# Use the Cartpole model from RobotZoo
model = Cartpole()
n,m = size(model)

# Define model discretization
N = 101
tf = 5.
dt = tf/(N-1)

# Define initial and final conditions
x0 = @SVector zeros(n)
xf = @SVector [0, pi, 0, 0]  # i.e. swing up

# Set up
Q = 1.0e-2*Diagonal(@SVector ones(n))
Qf = 100.0*Diagonal(@SVector ones(n))
R = 1.0e-1*Diagonal(@SVector ones(m))
obj = LQRObjective(Q,R,Qf,xf,N)

# Add constraints
conSet = ConstraintList(n,m,N)
u_bnd = 3.0
bnd = BoundConstraint(n,m, u_min=-u_bnd, u_max=u_bnd)
goal = GoalConstraint(xf)
add_constraint!(conSet, bnd, 1:N-1)
add_constraint!(conSet, goal, N)

# Initialization
u0 = @SVector fill(0.01,m)
U0 = [u0 for k = 1:N-1]

# Define problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
initial_controls!(prob, U0)

# Solve with ALTRO
opts = SolverOptions(
    cost_tolerance_intermediate=1e-2,
    penalty_scaling=10.,
    penalty_initial=1.0
)
altro = ALTROSolver(prob, opts)
solve!(altro)

# Get some info on the solve
max_violation(altro)  # 3.42e-9
cost(altro)           # 1.55
iterations(altro)     # 40

# Extract the solution
X = states(altro)
U = controls(altro)

# Solve with iLQR (ignores constraints)
ilqr = iLQRSolver(prob, opts)
initial_controls!(ilqr, U0)   # reset controls to initial guess, since they are modified by the previous solve
solve!(ilqr)
cost(ilqr)            # 1.45
iterations(ilqr)      # 84

```
