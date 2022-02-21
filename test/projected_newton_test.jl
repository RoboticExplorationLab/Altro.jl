using Altro
using TrajectoryOptimization
using RobotDynamics
using Test
using BenchmarkTools
using SparseArrays
using LinearAlgebra
using RobotDynamics: KnotPoint, Traj
using SparseArrays
const TO = TrajectoryOptimization
const RD = RobotDynamics

# Generate problem
N = 11
prob, opts = Problems.DubinsCar(:parallel_park, N=N)

# Create a trajectory with views into a single array
n,m,N = RD.dims(prob)
h = 0.1
NN = N*n + (N-1)*m
Zdata = zeros(NN + m)
ix = [(1:n) .+ (k-1)*(n+m) for k = 1:N]
iu = [n .+ (1:m) .+ (k-1)*(n+m) for k = 1:N-1]
iz = [(1:n+m) .+ (k-1)*(n+m) for k = 1:N]
Z = Traj([KnotPoint{n,m}(view(Zdata, iz[k]), (k-1)*h, h) for k = 1:N])
Z[end].dt = 0.0
RD.state(Z[2]) .= 1
RD.control(Z[3]) .= 2.1
@test Zdata[ix[2]] ≈ fill(1,n)
@test Zdata[iu[3]] ≈ fill(2.1,m)
Nc = sum(TO.num_constraints(prob))  # number of stage constraints
Np = N*n + (N-1)*m  # number of primals
Nd = N*n + Nc       # number of duals (no constraints)

## Create PNConstraintSet
D = spzeros(Nd, Np)
d = zeros(Nd)
a = trues(Nd)
pnconset = Altro.PNConstraintSet(prob.constraints, D, d, a)
alconset = Altro.ALConstraintSet2{Float64}(prob.constraints)
@test length(pnconset) == length(alconset)

Altro.evaluate_constraints!(pnconset, Z)
Altro.evaluate_constraints!(alconset, Z)
for i = 1:length(alconset)
    @test pnconset[i].vals ≈ alconset[i].vals
end

Altro.constraint_jacobians!(pnconset, Z)
Altro.constraint_jacobians!(alconset, Z)
for i = 1:length(alconset) - 1
    @test pnconset[i].jac ≈ alconset[i].jac
end


# Create a PN Solver
al = Altro.ALSolver(prob)
ilqr = al.ilqr 
pn = Altro.ProjectedNewtonSolver2(prob)
Np = Altro.num_primals(pn)

pnconset = pn.conset
@test pnconset.cinds[1][1] == n .+ (1:2m)
@test pnconset.cinds[end][1] == 1:n
@test pnconset.cinds[end][2] == n + 2m .+ (1:n)
@test pnconset.cinds[1][2] == n + 2m + n .+ (1:2m)
@test pnconset.cinds[2][1] == n + 2m + n + 2m .+ (1:4)
@test pnconset.cinds[end][3] == n + 2m + n + 2m + 4 .+ (1:n)


# Cost expansion 
copyto!(pn.Z, ilqr.Z)
Altro.cost_gradient!(pn)
Altro.cost_hessian!(pn)
Altro.cost_expansion!(ilqr.obj.obj, ilqr.Efull, ilqr.Z)
for k = 1:N
    @test ilqr.Efull[k].hess ≈ pn.hess[k]
    @test ilqr.Efull[k].grad ≈ pn.grad[k]
end

@test nnz(pn.A) == Np

# Dynamics Jacobian 
Altro.constraint_jacobians!(pn)
Altro.dynamics_expansion!(ilqr)
for k = 1:N-1
    @test ilqr.D[k].∇f ≈ pn.∇f[k]
end

# Dynamics error
Altro.dynamics_error!(pn)
for k = 1:N-1
    @test pn.e[k] ≈ RD.discrete_dynamics(prob.model, prob.Z[k]) - RD.state(prob.Z[k+1])
end

# Update everything
let pn = pn
    Altro.evaluate_constraints!(pn)
    Altro.constraint_jacobians!(pn)
    Altro.cost_gradient!(pn)
    Altro.cost_hessian!(pn)
    Altro.update_active_set!(pn)
end

D,d = Altro.active_constraints(pn)
@test length(d) < length(pn.d)

# Prepare for line search
ρ_primal = pn.opts.ρ_primal
ρ_chol = pn.opts.ρ_chol
HinvD = Diagonal(pn.H + I*ρ_primal) \ D'
S = Symmetric(D*HinvD)
Sreg = cholesky(S + ρ_chol*I, check=false)
@test issuccess(Sreg)
@test norm(d, Inf) > 1e-6

# line search
δλ = Altro.reg_solve(S, d, Sreg, 1e-8, 25)
δZ = -HinvD*δλ
pn.Z̄data .= pn.Zdata + 1.0 * δZ

Altro.evaluate_constraints!(pn, pn.Z̄)
@test Altro.max_violation(pn, nothing) < 1e-6

# Try entire pn solve
prob, opts = Problems.DubinsCar(:parallel_park, N=11)

solver = Altro.ALTROSolver2(prob, copy(opts))
solver.opts.projected_newton = false
solver.opts.constraint_tolerance = 1e-3
solve!(solver.solver_al)
viol = max_violation(solver) 
@test viol > 1e-6

solver.opts.constraint_tolerance = 1e-8
copyto!(get_trajectory(solver.solver_pn), get_trajectory(solver.solver_al))
@test max_violation(solver.solver_pn) ≈ viol
solve!(solver.solver_pn)
@test max_violation(solver.solver_pn) < 1e-8

##############################
## Comparison
##############################
prob, opts = Problems.DubinsCar(:parallel_park, N=11)

# s1 = Altro.iLQRSolver(prob, copy(opts))
# s2 = Altro.iLQRSolver2(prob, copy(opts))

opts.trim_stats = false
s1 = Altro.AugmentedLagrangianSolver(prob, copy(opts))
s2 = Altro.ALSolver(prob, copy(opts))

solve!(s1)
solve!(s2)


pn1 = Altro.ProjectedNewtonSolver(prob, opts, s1.stats)
pn2 = Altro.ProjectedNewtonSolver2(prob, opts, s2.stats)
copyto!(pn1.Z, get_trajectory(s1))
copyto!(pn2.Z, get_trajectory(s2))
@test states(pn1.Z) ≈ states(pn2.Z)
@test controls(pn1.Z) ≈ controls(pn2.Z)

# Altro.solve!(pn2)

@test cost(pn1) ≈ cost(pn2)
let solver = pn1
    Altro.update_constraints!(solver)
    Altro.copy_constraints!(solver)
    Altro.copy_multipliers!(solver)
    Altro.constraint_jacobian!(solver)
    Altro.copy_jacobians!(solver)
    TO.cost_expansion!(solver)
    Altro.update_active_set!(solver; tol=solver.opts.active_set_tolerance_pn)
    Altro.copy_active_set!(solver)
end
let pn = pn2
    Altro.evaluate_constraints!(pn)
    Altro.constraint_jacobians!(pn)
    Altro.cost_gradient!(pn)
    Altro.cost_hessian!(pn)
    Altro.update_active_set!(pn)
end
@test pn1.H ≈ pn2.H
@test pn1.g ≈ pn2.g
@test pn1.D ≈ pn2.D
@test pn1.d ≈ pn2.d atol=1e-12
@test pn1.active_set ≈ pn2.active

D1,d1 = Altro.active_constraints(pn1)
D2,d2 = Altro.active_constraints(pn2)
@test D1 ≈ D2
@test d1 ≈ d2

ρ_primal = pn1.opts.ρ_primal
ρ_chol = pn1.opts.ρ_chol
HinvD1 = Diagonal(pn1.H + I*ρ_primal) \ D1'
HinvD2 = Diagonal(pn2.H + I*ρ_primal) \ D2'
@test HinvD1 ≈ HinvD2
S1 = Symmetric(D1*HinvD1)
S2 = Symmetric(D2*HinvD2)
@test S1 ≈ S2
Sreg1 = cholesky(S1 + ρ_chol*I, check=false)
Sreg2 = cholesky(S2 + ρ_chol*I, check=false)
@test issuccess(Sreg1)
@test issuccess(Sreg2)

norm(d1, Inf)
norm(d2, Inf)
##

copyto!(pn1.P, pn1.Z)
α = 1.0
δλ1 = Altro.reg_solve(S1, d1, Sreg1, 1e-8, 25)
δλ2 = Altro.reg_solve(S2, d2, Sreg2, 1e-8, 25)
δλ1 ≈ δλ2
δZ1 = -HinvD1*δλ1
δZ2 = -HinvD2*δλ2
δZ1 ≈ δZ2
pn1.P.Z ≈ pn2.Zdata
pn1.P̄.Z .= pn1.P.Z + α * δZ1
pn2.Z̄data .= pn2.Zdata + α * δZ2
pn1.P̄.Z ≈ pn2.Z̄data

copyto!(pn1.Z̄, pn1.P̄)
Altro.update_constraints!(pn1, pn1.Z̄)
Altro.evaluate_constraints!(pn2, pn2.Z̄)
Altro.max_violation(pn2, nothing)
Altro.copy_constraints!(pn1)


d1_ = pn1.d[pn1.active_set]
norm(d1_, Inf)
d2_ = pn2.d[pn2.active]
norm(d2_, Inf)

norm(pn1.g + D1'pn1.λ[pn1.active_set])

let solver = pn1
    Altro.update_constraints!(solver)
    Altro.copy_constraints!(solver)
    Altro.copy_multipliers!(solver)
    Altro.constraint_jacobian!(solver)
    Altro.copy_jacobians!(solver)
    TO.cost_expansion!(solver)
    Altro.update_active_set!(solver; tol=solver.opts.active_set_tolerance_pn)
    Altro.copy_active_set!(solver)
end
let pn = pn2
    Altro.evaluate_constraints!(pn)
    Altro.constraint_jacobians!(pn)
    Altro.cost_gradient!(pn)
    Altro.cost_hessian!(pn)
    Altro.update_active_set!(pn)
end
Altro.multiplier_projection!(pn1)
Altro.multiplier_projection!(pn2)

pn1.P
pn2.Zdata



##
prob, opts = Problems.DubinsCar(:parallel_park, N=11)

# opts.trim_stats = false
# s1 = Altro.AugmentedLagrangianSolver(prob, copy(opts))
# s2 = Altro.ALSolver(prob, copy(opts))

s1 = Altro.ALTROSolver(prob, copy(opts))
s2 = Altro.ALTROSolver2(prob, copy(opts))

# s1.opts.projected_newton = false
# s2.opts.projected_newton = false
# s1.opts.constraint_tolerance = 1e-3
# s2.opts.constraint_tolerance = 1e-3

solve!(s1)
solve!(s2)

max_violation(s1)
max_violation(s2)

##
pn1 = s1.solver_pn
pn2 = s2.solver_pn
copyto!(get_trajectory(s2.solver_pn), get_trajectory(s2.solver_al))

@test states(pn1.Z) ≈ states(pn2.Z)
@test controls(pn1.Z) ≈ controls(pn2.Z)

# Altro.solve!(pn2)

@test cost(pn1) ≈ cost(pn2)
let solver = pn1
    Altro.update_constraints!(solver)
    Altro.copy_constraints!(solver)
    Altro.copy_multipliers!(solver)
    Altro.constraint_jacobian!(solver)
    Altro.copy_jacobians!(solver)
    TO.cost_expansion!(solver)
    Altro.update_active_set!(solver; tol=solver.opts.active_set_tolerance_pn)
    Altro.copy_active_set!(solver)
end
let pn = pn2
    Altro.evaluate_constraints!(pn)
    Altro.constraint_jacobians!(pn)
    Altro.cost_gradient!(pn)
    Altro.cost_hessian!(pn)
    Altro.update_active_set!(pn)
end
@test pn1.H ≈ pn2.H
@test pn1.g ≈ pn2.g
@test pn1.D ≈ pn2.D
@test pn1.d ≈ pn2.d atol=1e-12
@test pn1.active_set ≈ pn2.active
pn1.opts.active_set_tolerance_pn
pn2.opts.active_set_tolerance_pn
sum(pn1.active_set)
sum(pn2.active)

findall(pn1.active_set .!= pn2.active)
tol = pn1.opts.active_set_tolerance_pn
pn1.d[81] > -tol
pn2.d[81]
pn1.active_set[81]
pn2.active[81]