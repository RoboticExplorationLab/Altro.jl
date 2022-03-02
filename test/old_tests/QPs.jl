using TrajectoryOptimization, RobotDynamics
using StaticArrays, LinearAlgebra
using TrajectoryOptimization: QuadraticObjective, cost_expansion! 
using Altro
using Test
const TO = TrajectoryOptimization

n,m,N = 12,6,11
dt = 0.1
tf = dt*(N-1)
A = rand(n,n)
B = rand(n,m)
model = LinearModel(A,B,dt=dt)

# Objective
Q = Diagonal(@SVector rand(n))
R = Diagonal(@SVector rand(m))
Qf = Q*N
x0 = rand(n)
xf = 10*x0
obj = LQRObjective(Q,R,Qf,xf,N) 
U0 = [randn(m) for k = 1:N-1]

# Linear Constraints
cons0 = ConstraintList(n,m,N)
add_constraint!(cons0, GoalConstraint(xf), N)

cons_linear = copy(cons0)
add_constraint!(cons_linear, BoundConstraint(n,m, x_min=-2, x_max=2, u_min=-rand(m), u_max=rand(m)), 1:N-1)
add_constraint!(cons_linear, LinearConstraint(n,m,rand(3,n+m), rand(3),Inequality()), 1:N-1)

cons_nonlinear = copy(cons_linear)
add_constraint!(cons_nonlinear, CircleConstraint(n, [1,1],[1,2],[0.1,0.1]), 1:N-1)

prob = Problem(model, obj, xf, tf, x0=x0, U0=U0, integration=PassThrough)
rollout!(prob)
Z = prob.Z

E = QuadraticObjective(n,m,N)
cost_expansion!(E, obj, Z)
@test TO.is_quadratic(E)

# AL Objective
alobj = Altro.ALObjectiveOld(obj, cons0, model)
cost_expansion!(E, alobj, Z)
@test TO.is_quadratic(E)

alobj = Altro.ALObjectiveOld(obj, cons_linear, model)
cost_expansion!(E, alobj, Z)
@test !TO.is_quadratic(E)

# with with nonlinear constraint, should no longer be quadratic
alobj = Altro.ALObjectiveOld(obj, cons_nonlinear, model)
cost_expansion!(E, alobj, Z)
@test !TO.is_quadratic(E)