
@testset "Rocket Landing SOCP" begin
## Rocket landing problem
D,N = 3,101
n,m = 2D,D
dt = 0.1
tf = (N-1)*dt
x0 = SA_F64[10,10,100, 0,0,-10] 
xf = @SVector zeros(n)
Q = Diagonal(@SVector fill(1.0, n)) * dt
R = Diagonal(@SVector fill(1e-1, m)) * dt
Qf = (N-1)*Q * 100

##
model = RobotZoo.DoubleIntegrator(3, gravity=SA[0,0,-9.81])
obj = LQRObjective(Q,R,Qf,xf,N)
cons = ConstraintList(n,m,N)
u_bnd = 100.0
connorm = NormConstraint(n, m, u_bnd, TO.SecondOrderCone(), :control)
add_constraint!(cons, GoalConstraint(xf), N)
add_constraint!(cons, connorm, 1:N-1)

prob = Problem(model, obj, x0, tf, xf=xf, constraints=cons)
solver = ALTROSolver(prob, projected_newton=false, show_summary=false) 
solve!(solver)
@test iterations(solver) == 13 # 12

@test abs(maximum(norm.(controls(solver))) - u_bnd) < 1e-6
@test norm(states(solver)[end] - xf) < 1e-6
end