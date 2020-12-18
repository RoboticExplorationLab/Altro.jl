# Advanced Options

```@meta
CurrentModule = Altro
```

```@contents
Pages = ["advanced.md"]
```

## Infeasible Start
Standard indirect methods such as iLQR cannot be initialized with a state trajectory
since they are always dynamically feasible. However, for some problems an initial 
state trajectory is very informative and easy to generate, while supplying an 
initial guess for the controls is extremely difficult. For example, consider a 
quadrotor flying around some obstacles. Guessing a good path, or even the velocities,
would be pretty easy, but supplying the control sequence to generate that path 
is nearly as hard as just solving the entire trajectory optimization problem. 

ALTRO allows for ``infeasible'' starts by augmenting the discrete dynamics so that 
they become fully actuated, i.e. for any state we can provide a control that will
acheive it. This increases the size of the control dimension by `n`, the number of
states in the original problem, so the problem becomes more expensive to solve.

We specify that we want an infeasible start by passing the `infeasible` flag to 
the ALTRO constructor:
```@example infeasible
using Altro
prob,opts = Problems.DubinsCar(:escape)
solver = ALTROSolver(prob, opts, infeasible=true, R_inf=0.1)
size(solver)
```
where `R_inf` is the norm of the regularizer on the additional controls. Notice how
the new control dimension is 5, since the original control and state dimensions were
2 and 3.

The initial state trajectory can be provided using
```julia
initial_controls!(solver, U0)
initial_trajectory!(solver, Z0)
```
where `U0::Union{SVector, Matrix, Vector{<:StaticVector}}` and 
`Z0::AbstractTrajectory`. 

```@docs
InfeasibleModel
InfeasibleConstraint
infeasible_trajectory
```
