@testset "Solves" begin
# Test max iterations
solver = ALTROSolver2(Problems.Pendulum()...)
set_options!(solver, 
    iterations=100,
    cost_tolerance_intermediate=1e-10, 
    gradient_tolerance_intermediate=1e-10, 
    constraint_tolerance=1e-10,
    show_summary=false
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
prob = Problem(model, obj, x0, tf, xf=xf)
U0 = [u0*1e1 for k = 1:N-1]
initial_controls!(prob, U0)

solver = ALTROSolver2(prob, show_summary=false)
Altro.is_constrained(solver)
solve!(solver)
@test iterations(solver) == 1
@test status(solver) == Altro.MAXIMUM_COST

## test max state value 
model = RobotZoo.Quadrotor()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
N = 101
tf = 5.0
dt = tf / (N-1)
x0,u0 = zeros(model)
xf = MArray(x0); xf[3] = 200
Q = Diagonal(SA[10,10,10, 1,1,1,1, 10,10,10, 10,10,10.])*1e-1 * dt
R = Diagonal(@SVector ones(4)) * dt
obj = LQRObjective(Q,R,Q*N,xf,N,uf=u0)
prob = Problem(dmodel, obj, x0, tf, xf=xf)
U0 = [u0*2 for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob)
cost(prob)

solver = ALTROSolver2(prob, show_summary=false)
cost(solver)
set_options!(solver, max_state_value=100)
@test Altro.is_constrained(solver) == false
solve!(solver)
@test status(solver) == Altro.MAX_ITERATIONS

## Max outer iterations
prob, opts = Problems.DubinsCar(:parallel_park, N=N)
opts.projected_newton = false
solver = ALTROSolver2(prob, opts, show_summary=false)
solve!(solver)
@test max_violation(solver) > opts.constraint_tolerance
status(solver) == Altro.MAX_ITERATIONS_OUTER
end