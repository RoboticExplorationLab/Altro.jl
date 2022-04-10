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

ALTRO allows for "infeasible" starts by augmenting the discrete dynamics so that 
they become fully actuated, i.e. for any state we can provide a control that will
acheive it. This increases the size of the control dimension by `n`, the number of
states in the original problem, so the problem becomes more expensive to solve.

We specify that we want an infeasible start by passing the `infeasible` flag to 
the ALTRO constructor:
```@example infeasible
using Altro
prob,opts = Problems.DubinsCar(:escape)
solver = ALTROSolver(prob, opts, infeasible=true, R_inf=0.1)
nothing  # hide
```
where `R_inf` is the norm of the regularizer on the additional controls. Notice how
the new control dimension is 5, since the original control and state dimensions were
2 and 3.

The initial state trajectory can be provided using one of
```julia
initial_states!(solver, X0)
initial_trajectory!(solver, Z0)
```
where `X0::Union{SVector, Matrix, Vector{<:StaticVector}}` and 
`Z0::SampledTrajectory`. 

```@docs
InfeasibleModel
InfeasibleConstraint
infeasible_trajectory
```

## Using Implicit Integrators
By leveraging the functionality of RobotDynamics.jl, Altro can easily solve problems 
using implicit integrators like implicit midpoint, which as a symplectic integrator 
has energy-conserving behaviors and also preserves any implicit norms in the dynamics
(such as the norm on a quaternion representing rotations). The following example 
shows how to use an implicit integrator:

```@example implicit
using RobotZoo: Cartpole
using RobotDynamics
using TrajectoryOptimization
using Altro
using LinearAlgebra
const RD = RobotDynamics

model = Cartpole()
dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

# Temporary "hack" to make sure it doesn't try to use the `UserDefined` method
RD.default_diffmethod(::Cartpole) = RD.ForwardAD()

tf = 2.0
N = 51
n,m = RD.dims(model)
x0 = [0,0,0,0.]
xf = [0,pi,0,0]

Q = Diagonal(fill(1.0, n))
R = Diagonal(fill(0.1, m))
Qf = Q*(N-1) 
obj = LQRObjective(Q,R,Qf,xf,N)

prob = Problem(dmodel, obj, x0, tf)

solver = ALTROSolver(
    prob, 
    dynamics_diffmethod=RD.ImplicitFunctionTheorem(RD.ForwardAD())
)
nothing  # hide
```

The key here is to specify the solver option `dynamics_diffmethod` to be
`RobotDynamics.ImplicitFunctionTheorem()` which takes another `RobotDynamics.DiffMethod`
as an argument, which specified how the Jacobians of the dynamics residual should be 
computed. The implicit function theorem then uses the partial derivatives to compute 
the Jacobians with respect to the next state, which are the Jacobians requried by 
algorithms like iLQR.

## Disabling Octavian
By default, Altro.jl uses 
[Octavian.jl](https://github.com/JuliaLinearAlgebra/Octavian.jl) for matrix 
multiplication. This typically yields very good runtime performance but can take a 
while to compile the first time. If you want to disable the use of Octavian, 
set the environment variable `ALTRO_USE_OCTAVIAN = false` prior to using Altro.
If Altro has already precompiled, you'll need to delete the compiled cache using

```
rm -rf ~/.julia/compiled/v1.x/Altro/*
```

and then when you enter `using Altro` in the Julia REPL you should see it print a 
message that it's precompiling. You can check to see if Altro is using Octavian 
by checking the `Altro.USE_OCTAVIAN` variable in the Altro module.