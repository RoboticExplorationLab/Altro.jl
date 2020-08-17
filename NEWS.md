# New in `v0.2`

## Flattened API
* The new API now only exports the `ALTROSolver` type, no longer exporting `AugmentedLagrangianSolver`, `iLQRSolver`, and `ProjectedNewtonSolver`.
* Use the `projected_newton` solver option to turn off to get the same behavior as `AugmentedLagrangianSolver`. Passing an unconstrained problem is equivalent to solving with `iLQRSolver`.
* Solver options are now shared between all solvers, and implemented as `SolverOptions`. Use `solver.opts` or `set_options!(solver; opts...)` to change solver options. Solver options can also be passed into the solver constructor.
* Solver stats have also been improved, and contains a single log for all iteration types. Accessible via `solver.stats` or `Altro.stats(solver)`.

## Improved Verbosity Output
* iLQR iterations now provide a brief description for why they terminate.
* The `show_summary` option can be used to display a brief summary of the solve upon completion.

## Solver status and convergence 
* Each solve now saves a solver status in `solver.stats.status`, accessible via `status(solver)`, which provides some information about how the solver exited.
* Runtime errors in Altro.jl were removed in favor of exiting the solver with a specific solver status. Runtime errors are still possible if they originate from TrajetoryOptimization.jl or bugs (nothing is wrapped in `try/catch` loops).
* Changed convergence criteria for the iLQR solver: it must satisfy both the change in cost tolerance and the gradient tolerance at the same time.
* The `ProjectedNewtonSolver` now does a projection of the dual variables in order to accurately calculation the residual of dual optimality (gradient of the Lagrangian). Can be disabled via the `multiplier_projection` option.
