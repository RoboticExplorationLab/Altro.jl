using StaticArrays
using RobotZoo
using LinearAlgebra
const TO = TrajectoryOptimization

# Test max iterations
solver = ALTROSolver(Problems.Pendulum()...)
set_options!(solver, 
    iterations=100,
    cost_tolerance_intermediate=1e-10, 
    gradient_tolerance_intermediate=1e-10, 
    constraint_tolerance=1e-10,
    show_summary=true, 
    projected_newton=false, 
    verbose=2
)
solve!(solver)
status(solver)
@test iterations(solver) == solver.opts.iterations 
@test status(solver) == Altro.MAX_ITERATIONS

## test max cost
model = RobotZoo.Quadrotor()
N = 101
tf = 5.0
x0,u0 = zeros(model)
xf = MArray(x0); xf[1] = 100
Q = Diagonal(@SVector ones(13))*1e6
R = Diagonal(@SVector ones(4))
obj = LQRObjective(Q,R,Q*N,xf,N,uf=u0)
prob = Problem(model, obj, xf, tf, x0=x0)
U0 = [u0*1e1 for k = 1:N-1]
initial_controls!(prob, U0)

solver = ALTROSolver(prob)
Altro.is_constrained(solver)
solve!(solver)
@test iterations(solver) == 1
@test status(solver) == Altro.MAXIMUM_COST

## test max state value 
model = RobotZoo.Quadrotor()
N = 101
tf = 5.0
x0,u0 = zeros(model)
xf = MArray(x0); xf[1] = -1e3 
Q = Diagonal(SA[10,10,10, 1,1,1,1, 10,10,10, 10,10,10.])*1e-1
R = Diagonal(@SVector ones(4))
obj = LQRObjective(Q,R,Q*N,xf,N,uf=u0)
prob = Problem(model, obj, xf, tf, x0=x0)
U0 = [u0 for k = 1:N-1]

solver = ALTROSolver(prob)
set_options!(solver, max_state_value=100)
@test Altro.is_constrained(solver) == false
solve!(solver)
@test status(solver) == Altro.STATE_LIMIT