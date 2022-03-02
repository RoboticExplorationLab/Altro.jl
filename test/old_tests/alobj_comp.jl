using Altro
using TrajectoryOptimization
using Test
using StaticArrays
using ForwardDiff
using RobotDynamics
using LinearAlgebra
using RobotZoo
const RD = RobotDynamics
const TO = TrajectoryOptimization

function build_test_conlist()
    n, m, N = 3, 2, 11
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

    return cons
end

function build_test_objective()
    n,m,N = 3,2,11
    Q = Diagonal(@SVector fill(1.0, n))
    R = Diagonal(@SVector fill(0.1, m))
    Qf = Diagonal(@SVector fill(10.0, n))
    xf = @SVector zeros(n)
    LQRObjective(Q, R, Qf, xf, N)
end

T = Float64
cons = build_test_conlist()

model = RobotZoo.DubinsCar()
n,m = RD.dims(model)
N = 11

# Generate random trajectory
X = [randn(T,n) for k = 1:N]
U = [randn(T,m) for k = 1:N-1]
Z = Traj(X, U, tf=2.0)

# Build AL Objectives
obj = build_test_objective()
alobj1 = Altro.ALObjectiveOld(obj, cons, model)
alobj2 = Altro.ALObjective2{T}(obj, cons)
conset1 = alobj1.constraints
conset2 = alobj2.conset

##############################
# Test AL Constraint sets 
##############################

# Compare constraint evaluation
RD.evaluate!(conset1, Z)
Altro.evaluate_constraints!(conset2, Z)
for i = 1:length(conset2)
    @test conset1.convals[i].vals ≈ conset2.constraints[i].vals
end

# Compare constraint Jacobians
RD.jacobian!(conset1, Z, true)
Altro.constraint_jacobians!(conset2, Z)
for i = 1:length(conset2)
    @test conset1.convals[i].jac ≈ conset2.constraints[i].jac
end

##############################
# AL Objectives
##############################

# Evaluate cost expansions
E1 = TO.CostExpansion{T}(n,m,N)
E2 = Altro.CostExpansion{T}(n,m,N)
TO.cost_expansion!(E1, alobj1, Z)
Altro.cost_expansion!(alobj2, E2, Z)

# Check entire AL cost
@test TO.cost(alobj1, Z) ≈ TO.cost(alobj2, Z)

# Check AL penalty expansions 
for i = 1:length(conset2)
    @test conset1.convals[i].grad ≈ conset2[i].grad
    @test conset1.convals[i].hess ≈ conset2[i].hess
end

# Check entire cost expansion
for k = 1:N
    @test E1[k].grad ≈ E2[k].grad
    @test E1[k].hess ≈ E2[k].hess
end