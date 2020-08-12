![CI](https://github.com/RoboticExplorationLab/ALTRO.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/RoboticExplorationLab/ALTRO.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RoboticExplorationLab/ALTRO.jl)

# Altro.jl
Implementation of the Augmented Lagrangian TRajectory Optimizer (ALTRO) solver, a very fast solver for constrained trajectory optimization problems.
ALTRO uses iterative LQR (iLQR) with an augmented Lagrangian framework and can solve problems with nonlinear inequality and equality path constraints
and nonlinear dynamics. The key features of the ALTRO solver are:
  * General nonlinear cost functions, including minimum time problems
  * General nonlinear state and input constraints
  * Infeasible state initialization
  * Square-root methods for improved numerical conditioning
  * Active-set projection method for solution polishing

Altro.jl solves trajectory optimization problems set up using [TrajectoryOptimization.jl](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl).

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

## Solver Options
Like any nonlinear programming solver, ALTRO comes with a host of solver options.
While the default values yield good/acceptable performance on many problem, extra
performance can always be gained by tuning these parameters. In practice, there are
only a few parameters that need to be tuned. See the [AL-iLQR Tutorial](https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf) for more details.

The ALTRO solver is actually a composition of several different solvers with their own
options. Early versions of Altro.jl required the user to manipulate a rather confusing
heirarchy of solver options. Altro new exposes the following API that simplifies
querying and setting solver options.

### Setting Solver Options
Solver options can be specified when the solver is instantiated or afterwards using 
the `set_options!` command. If we have a previously constructed `Problem`, this looks
like
```julia
solver = ALTROSolver(prob, verbose=1, constraint_tolerance=1e-3, square_root=true)
```
Note that the options can apply any of the nested solvers within ALTRO. For example,
the `square_root` option applies to the iLQR solver. Since `constraint_tolerance` is 
an option for ALTRO, Augmented Lagrangian, and Projected Newton, all of these will be
to the same value.

Alternatively, solver options can be set using the `set_options!` command after the
solver has been instantiated:
```julia
set_options!(solver, reset_duals=true, penalty_initial=100, penalty_scaling=50)
```

### Querying Solver Options
With so many options, it can be difficult to remember the exact names for each 
option. You can query if an option exists using `has_option`:, for example
```julia
has_option(solver, :constraint_decrease_ratio)
```
will return `true` if `solver` is an ALTRO or Augmented Lagrangian solver, and false
otherwise.

If we want to retrieve the current value for an option we know exists (presumably 
checked with `has_option`) we use `get_option`, for example:
```julia
iters_AL = get_option(solver, :iterations_outer)
iters_iLQR = get_option(solver, :iterations)
```

Lastly, if we want to retrieve all the solver options at the same time we can use
the `get_options(solver, recursive::Bool=true, group::Bool=false)` which has two
boolean flags that modify the behavior. By default, the command will collect all
the solver options for the current solver and any sub-solvers into a single 
dictionary. Any options with the same name but different values will display a
helpful warning and invalidate the option in the out (setting it to `:invalid`). 
Calling this on an ALTRO solver will give about 55 options.

If you want the options for only a single solver, without the options of any 
sub-solvers, set the `recursive` option to `false`, e.g. 
```julia
get_options(solver.solver_al, false)
```
which will return the options for the Augmented Lagrangian solver. Or
```julia
get_options(solver.solver_al.solver_uncon, false)
```
which will return the options for the iLQR solver. Note that in the case the result
will be the same if pass `recursive=true` since iLQR doesn't have any nested solvers.

Alternatively, we can group the options by solver, in which case we set 
`recursive=true` and `group=true`, which will return a dictionary with an entry for
each solver, indexed by a short idenifier (such as `:AL` for Augmented Lagrangian).
For example, calling this on an ALTRO solver
```julia
get_options(solver, true, true)
```
will give a dictionary that looks like 
`Dict(:iLQR=>opts_ilqr, :AL=>opts_al, :ALTRO=>opts_altro, :PN=>opts_pn)` where
`opts_[xxx]` is a dictionary of options for the solver.