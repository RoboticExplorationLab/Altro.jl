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
## Solver Statistics
ALTRO logs intermediate values during the course of the solve. These values are all
stored in the `SolverStats` type, accessible via `solver.stats` or `Altro.stats(solver)`. This currently stores the following information:
| Field | Description |
| ----- | ----------- |
| `iterations` | Total number of iterations |
| `iterations_outer` | Number of outer loop (Augmented Lagrangian) iterations |
| `iterations_pn` | Number of projected newton iterations |
| `iteration` | Vector of iteration number |
| `iteration_outer` | Vector of outer loop iteration number |
| `cost` | Vector of costs |
| `dJ` | Change in cost |
| `c_max` | Maximum constrained violation |
| `gradient` | Approximation of dual optimality residual (2-norm of gradient of the Lagrangian) |
| `penalty_max` | Maximum penalty parameter |

The other fields are used interally by the solver and not important to the end user.

The vector fields of the `SolverStats` type can be converted to a dictionary via `Dict(stats::SolverStats)`,
which can then be cast into a tabular format such as `DataFrame` from DataFrames.jl.

## Solver Options
Like any nonlinear programming solver, ALTRO comes with a host of solver options.
While the default values yield good/acceptable performance on many problem, extra
performance can always be gained by tuning these parameters. In practice, there are
only a few parameters that need to be tuned. See the [AL-iLQR Tutorial](https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf) for more details.

The ALTRO solver is actually a composition of several different solvers with their own
options. Early versions of Altro.jl required the user to manipulate a rather confusing
heirarchy of solver options. Newer version of Altro.jl provide a single options struct
that dramatically simplifies setting and working with the solver parameters.

### Setting Solver Options
Solver options can be specified when the solver is instantiated or afterwards using 
the `set_options!` command. If we have a previously constructed `Problem`, this looks
like
```julia
solver = ALTROSolver(prob, verbose=1, constraint_tolerance=1e-3, square_root=true)
```
Alternatively, solver options can be set using the `set_options!` command after the
solver has been instantiated:
```julia
set_options!(solver, reset_duals=true, penalty_initial=100, penalty_scaling=50)
```

### Querying Solver Options
The options struct for the `ALTROSolver` can be directly accessed via `solver.opts` or
`Altro.options(solver)`. Options can be directly set or retrieved from this mutable
struct.

### List of Options
For convenience, we provide a list of options in the ALTRO solver, along with a brief
description:

| Option | Description | Importance | Default |
| ------ | ----------- | ---------- | ------- |
| `constraint_tolerance` | All constraint violations must be below this value. | High | `1e-6` |
| `cost_tolerance` | The difference in costs between subsequent iterations must be below this value. | High | `1e-4` |
| `cost_tolerance_intermediate` | Cost tolerance for intermediate iLQR solves. Can speed up convergence by increase to 10-100x the `cost_tolerance`. | Med | `1e-4` |
| `gradient_tolerance` | Tolerance for 2-norm of primal optimality residual. | Low | `1` |
| `gradient_tolerance_intermediate` | Primal optimality residual tolerance for intermediate solve. | Low | `10` |
| `iterations_inner` | Max iLQR iterations per iLQR solve. | Med | `300` |
| `dJ_counter_limit` | Max number of times iLQR can fail to make progress before exiting. | Low | `10` |
| `square_root` | Enable the square root backward pass for improved numerical conditioning (WIP). | Med | `false` |
| `line_search_lower_bound` | Lower bound for Armijo line search. | Low | `1e-8` |
| `line_search_upper_bound` | Upper bound for Armijo line search. | Low | `10.0` |
| `iterations_linesearch` | Max number of backtracking steps in iLQR line search | Low | 20 |
| `max_cost_value` | Maximum cost value. Will terminate solve if cost exeeds this limit. | Low | `1e8` |
| `max_state_value` | Maximum value of any state. Will terminate solve if any state exeeds this limit. | Low | `1e8` |
| `max_control_value` | Maximum value of any control. Will terminate solve if any control exeeds this limit. | Low | `1e8` |
| `static_bp` | Enable the static backward pass. Only advisable for state + control dimensions < 20. Turn off if compile time is exessive. | Low | `true` |
| `save_S` | Save the intermediate cost-to-go expansions in the iLQR backward pass. |Low | `false` |
| `bp_reg` | Enable iLQR backward pass regularization (WIP). | Med | `false` |
| `bp_reg_initial` | Initial backward pass regularization. | Low | `0.0` |
| `bp_reg_increase_factor` | Multiplicative factor by which the regularization is increased. | Low | `1.6` |
| `bp_reg_max` | Maximum regularization. | Low | `1e8` |
| `bp_reg_min` | Minimum regularization. | Low | `1e-8` |
| `bp_reg_fp` | Amount of regularization added when foward pass fails | Low | `10.0` |
| `penalty_initial` | Initial penalty term on all constraints. Set low if the unconstrained solution is a good approximate solution to the constrained problem, and high if the initial guess provided is a good esimate. If `NaN` uses values in each constraint param, which defaults to `1.0`. | `Very High` | `NaN` |
| `penalty_scaling` | Multiplicative factor by which the penalty is increased each outer loop iteration. High values can speed up convergence but quickly lead to poor numerical conditioning on difficult problems. Start with small values and then increase.If `NaN` defaults to `10` in the per-constraint parameter. | `Very High` | `NaN` |
| `iterations_outer` | Max number of outer loop (Augmented Lagrangian) iterations. | Med | `30` |
| `verbose_pn` | Turn on printing in the projected newton solver. | Low | `false` |
| `n_steps` | Maximum number of projected newton steps. | Low | `2` |
| `projected_newton_tolerance` | Constraint tolerance at which the solver will exit the Augmented Lagrangian solve and start the projected newton solve. Typically `sqrt(constraint_tolerance)` | High | `1e-3` |
| `active_set_tolerance_pn` | Tolerance for the active constraints during the projected newton solve. Includes some barely satisfied constraints into the active set. Can fix singularity issues during projected newton solve. | Med | `1e-3` |
| `multiplier_projected` | Enable updating the dual variables during the projected newton solve. Also provides a calculation of the optimality residual in the stats output. | Low | `true` |
| `ρ_chol` | Regularization on the projected newton Cholesky solve. | Med | `1e-2` |
| `ρ_primal` | Regularization on the primal variables during the projected newton solve. Required if cost Hessian is positive-semi-definite. | Low | `1e-8` |
| `ρ_dual` | Regularization on the dual variables during the multiplier projection step. | Low | `1e-8` |
| `r_threshold` | Improvement ratio threshold for projected newton solve. If the ratio of constraint violations between subsequent steps is less than this value, it will update the cost and constraint expansions | Low | `1.1` |
| `projected_newton` | Enable projected newton solve. If enabled, `projected_newton_solve` is used as the `constraint_tolerance` for the AL-iLQR solve. Projected newton solve is still a WIP and not very robust. | High | `true` |
| `iterations` | Max number of total iterations (iLQR + projected newton). | Med | 1000 |
| `verbose` | Controls output during solve. `0` is zero output, `1` outputs AL iterations, and `2` outputs both AL and iLQR iterations | Low | `0` |
