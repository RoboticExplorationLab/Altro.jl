# Getting Started 

```@meta
CurrentModule = Altro
```

```@contents
Pages = ["quickstart.md"]
```
Setting up and solving a problem with ALTRO is very straight-forward. Let's
walk through an example of getting a Dubins car to drive through some circular
obstacles.

## 1. Load the packages
Our first step is to load the required packages. Since we need to define our
dynamics model, we need [RobotDynamics.jl](https://github.com/RoboticExplorationLab/RobotDynamics.jl), and we need [TrajectoryOptimization.jl](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl) to define our problem. We'll 
also import [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) for 
fast, allocation-free matrix methods, and the `LinearAlgebra` module. To avoid 
having to type `TrajectoryOptimization` and `RobotDynamics` all the time, we also
create some convenient aliases.

```@example car
using Altro
using TrajectoryOptimization
using RobotDynamics
using StaticArrays, LinearAlgebra
const TO = TrajectoryOptimization
const RD = RobotDynamics
```

## 2. Set up the dynamics model
We now define our dynamics model using RobotDynamics.jl. We define a new type `Car` 
that inherits from `RobotDynamics.AbstractModel`. We can store any of our model 
parameters in this type. After defining the state and control dimensions and the 
continuous dynamics, we're done defining our model. Integration of the dynamics
and the dynamics derivatives can be done automatically.

```@example car
struct DubinsCar <: AbstractModel end
RD.state_dim(::DubinsCar) = 3
RD.control_dim(::DubinsCar) = 2

function RD.dynamics(::DubinsCar,x,u)
    ẋ = @SVector [u[1]*cos(x[3]),
                  u[1]*sin(x[3]),
                  u[2]]
end
```

## 3. Set up the objective
Once we've defined the model, we can now start defining our problem. Let's start
by defining the discretization,
```@example car
model = DubinsCar()
n,m = size(model)    # get state and control dimension
N = 101              # number of time steps (knot points). Should be odd.
tf = 3.0             # total time (sec)
dt = tf / (N-1)      # time step (sec)
```
and our initial and final conditions
```@example car
x0 = SA_F64[0,0,0]   # start at the origin
xf = SA_F64[1,2,pi]  # goal state
nothing  # hide
```
Let's define a quadratic cost function that penalizes distance from the goal state:
```@example car
Q  = Diagonal(SA[0.1,0.1,0.01])
R  = Diagonal(SA[0.01, 0.1])
Qf = Diagonal(SA[1e2,1e2,1e3])
obj = LQRObjective(Q,R,Qf,xf,N)
nothing # hide
```

## 4. Add the constraints
Now let's define the constraints for our problem. We're going to bound the workspace of the robot, and add two obstacles. We start by defining a `ConstraintList`, which 
is going to hold all of the constraints and make sure they're dimensions are 
consistent. Here we add a goal constraint at the last time step, a workspace 
constraint, and then the circular obstacle constraint.
```@example car
cons = ConstraintList(n,m,N)

# Goal constraint
goal = GoalConstraint(xf)
add_constraint!(cons, goal, N)

# Workspace constraint
bnd = BoundConstraint(n,m, x_min=[-0.1,-0.1,-Inf], x_max=[5,5,Inf])
add_constraint!(cons, bnd, 1:N-1)

# Obstacle Constraint
obs = CircleConstraint(n, SA_F64[1,2], SA_F64[1,1], SA[0.2, 0.3])
add_constraint!(cons, bnd, 1:N-1)
nothing # hide
```

## 5. Define the problem
With the dynamics model, discretization, objective, constraints, and initial condition
defined, we're ready to define the problem, which we do with 
`TrajectoryOptimization.Problem`. Here we can also specify an integrator, which we
choose to be a 4th-order Runge-Kutta method.
```@example car
prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons, integration=RK4)
nothing # hide
```
Initialization is key to nonlinear optimization problems. Since this problem is pretty
easy, we'll just initialize it with small random noise on the controls and then
simulate the system forward in time.
```@example car
initial_controls!(prob, [@SVector rand(m) for k = 1:N-1])
rollout!(prob)   # simulate the system forward in time with the new controls
```

## 6. Intialize the solver
With the problem now defined, we're ready to start using Altro.jl (everything up to
this point used only RobotDynamics or TrajectoryOptimization). All we need to
do is create an `ALTROSolver`.
```@example car
solver = ALTROSolver(prob)
nothing # hide
```

### Setting Solver Options
We can set solver options via keyword arguments to the constructor, or by passing in 
a `SolverOptions` type:
```@example car
# Set up solver options
opts = SolverOptions()
opts.cost_tolerance = 1e-5

# Create a solver, adding in additional options
solver = ALTROSolver(prob, opts, show_summary=false)
nothing # hide
```

## 7. Solve the problem
With the solver initialized, we can now solve the problem with a simple call to 
`solve!`:
```@example car
solve!(solver)
nothing # hide
```

## 8. Post-analysis

### Checking solve status
Once the solve is complete, we can look at a few things. The first is to check if the
solve is successful:
```@example car
status(solver)
```
We can also extract some more information
```@example car
println("Number of iterations: ", iterations(solver))
println("Final cost: ", cost(solver))
println("Final constraint satisfaction: ", max_violation(solver))
```

### Extracting the solution
We can extract the state and control trajectories, which are returned as vectors of
`SVector`s:
```@example car
X = states(solver)     # alternatively states(prob)
U = controls(solver)   # alternatively controls(prob)
```
If you prefer to work with matrices, you can convert them easily:
```@example car
Xm = hcat(Vector.(X)...)  # convert to normal Vector before concatenating so it's fast
Um = hcat(Vector.(U)...)
```


!!! tip
    Converting a matrix into a vector of vectors is also very easy:
    ```julia
    X = [col for col in eachcol(Xm)]
    ```
    Or if you want static vectors:
    ```julia
    X = [SVector{n}(col) for col in eachcol(Xm)]
    ```

### Extracting the final feedback gains 
Since ALTRO uses iLQR, the solver computes a locally optimal linear feedback policy
which can be useful for tracking purposes. We can extract it from the internal 
`Altro.iLQRSolver`:
```@example car
ilqr = Altro.get_ilqr(solver)
K = ilqr.K  # feedback gain matrices
d = ilqr.d  # feedforward gains. Should be small.
```

### Additional solver stats
We can extract more detailed information on the solve from `solver.stats`:
```@example car
solver.stats
```
The most relevant fields are the `cost`, `c_max`, and `gradient`.
These give the history of these values for each iteration. The `iteration_outer` can
also be helpful to know which iterations were outer loop (augmented Lagrangian) 
iterations. The `tsolve` field gives the total solve time in milliseconds.