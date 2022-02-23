using Altro
using TrajectoryOptimization
using Test
using StaticArrays
using ForwardDiff
using RobotDynamics
using LinearAlgebra
const RD = RobotDynamics
const TO = TrajectoryOptimization


T = Float64
n, m, N = 3, 2, 11
tf = 2.0
xf = [1,2,3.]

con_eq = TO.GoalConstraint(xf, SA[1,3])
inds_eq = 1:10
confun_eq(x) = (x - xf)[[1,3]]

xc, yc, rad = [4,2,2,0.], [0,1,2,3.], [0.2, 0.1, 0.3, 0.4]
con_in = TO.CircleConstraint(n, xc, yc, rad)
inds_in = 2:6
confun_in(x) = rad.^2 .- (x[1] .- xc).^2 .- (x[2] .- yc).^2

con_so = TO.NormConstraint(n, m, 3, TO.SecondOrderCone(), :state)
inds_so = N:N 
confun_so(x) = [x; 3]

cons = ConstraintList(n, m, N)
add_constraint!(cons, con_eq, inds_eq)
add_constraint!(cons, con_in, inds_in, sig=RD.InPlace())
add_constraint!(cons, con_so, inds_so)

Z = Traj(randn(n,N), randn(m,N), dt=0.1)
opts = SolverOptions()
alcosts = zeros(N)
conset = Altro.ALConstraintSet2{T}()
Altro.initialize!(conset, cons, Z, opts, alcosts)

@test length(conset) == 3
@test length(conset.constraints) == 3
@test length(conset.c_max) == N 
@test length(conset.μ_max) == 3
@test conset[begin].con == con_eq
@test conset[2].con == con_in
@test conset[end].con == con_so
@test conset[1].sig == RD.StaticReturn()
@test conset[2].sig == RD.InPlace()

# Generate a random trajectory
X = [randn(T,n) for k = 1:N]
U = [randn(T,m) for k = 1:N-1]
Z = Traj(X, U, tf=2.0)

# Methods
Altro.evaluate_constraints!(conset, Z)
Altro.constraint_jacobians!(conset, Z)
Altro.alcost(conset)
@test sum(alcosts) != 0
Altro.algrad!(conset)
Altro.alhess!(conset)
Altro.max_violation!(conset[1])
Altro.dualupdate!(conset)
Altro.reset_penalties!(conset)
Altro.penaltyupdate!(conset)
@test Altro.max_penalty(conset) == 10.0

## Reset
opts.penalty_initial = 0.1
Altro.reset!(conset)
@test Altro.max_penalty(conset) ≈ 0.1
