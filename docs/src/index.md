# Altro.jl Documention

```@meta
CurrentModule = Altro
```

Documentation for Altro.jl

```@contents
Pages = ["index.md"]
```

## Overview
ALTRO (Augmented Lagrangian TRajectory Optimizer) is a fast solver for solving 
nonlinear, constrained trajectory optimization problems of the form:

```math
\begin{aligned}
  \min_{x_{0:N},u_{0:N-1}} \quad & \ell_f(x_N) + \sum_{k=0}^{N-1} \ell_k(x_k, u_k, dt) \\
  \textrm{s.t.}            \quad & x_{k+1} = f(x_k, u_k), \\
                                 & g_k(x_k,u_k) \leq 0, \\
                                 & h_k(x_k,u_k) = 0.
\end{aligned}
```

ALTRO uses iterative LQR (iLQR) as the primary solver, which is used to generate 
locally-optimal linear feedback policies and satisfy the nonlinear dynamics 
constraints. Generic stage-wise state and control constraints are handled using
an augmented Lagrangian. 

Once the augmented Lagrangian solver has converged to coarse tolerances, ALTRO
can switch to an active-set projected Newton phase that provides fast convergence to
tight constraint satisfaction.

ALTRO has demonstrated state-of-the-art performance for convex conic MPC problems, 
beating SOCP solvers such as Mosek, ECOS, and SCS. For quadratic MPC problems, 
ALTRO has performance on-par or better than OSQP. 

ALTRO builds off the interfaces provided by
[TrajectoryOptimization.jl](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl) and 
[RobotDynamics.jl](https://github.com/RoboticExplorationLab/RobotDynamics.jl).
Please see the documentation for those packages for a more in-depth treatment 
of defining dynamics models and setting up trajectory optimization problems. 
The purpose of this documentation is to provide insight into the ALTRO 
algorithm, it's Julia implementation, and the options this solver provides.

## Key Features
* State-of-the-art performance for both convex (linear) and nonlinear trajectory optimization problems
* Convenient interface for dynamics and problem definition via  [TrajectoryOptimization.jl](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl) and [RobotDynamics.jl](https://github.com/RoboticExplorationLab/RobotDynamics.jl).
* Supports generic nonlinear state and control constraints at each time step.
* Supports second-order-cone programs (SOCPs).
* Allows initialization of both state and control trajectories.
* Supports integration up to 4th-order Runge-Kutta methods. Higher-order methods are possible but not yet implemented.
* Supports optimization on the space of 3D rotations.
* Provides convenient methods for warm-starting MPC problems.
* Provides efficient methods for auto-differentiation of costs, constraints, and dynamics via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).

## Installation
Altro.jl can be installed via the Julia package manager. Within the Julia
REPL:
```
] # activate the package manager
(v1.5) pkg> add Altro 
```
A specific version can be specified using
```
(v1.5) pkg> add Altro@0.3
```
Or you can check out the master branch with
```
(v1.5) pkg> add Altro#master
```
Lastly, if you want to clone the repo into your `.julia/dev/` directory for development, you can use
```
(v1.5) pkg> dev Altro 
```

This will automatically add all package dependencies (see [`Project.toml`](https://github.com/RoboticExplorationLab/Altro.jl/blob/master/Project.toml)).
If you want to explicitly use any of these dependencies (such as [RobotDynamics.jl](https://github.com/RoboticExplorationLab/RobotDynamics.jl)), 
you'll need to individually add those packages to your environment via the package manager.