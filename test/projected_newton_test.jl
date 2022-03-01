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
Z = SampledTrajectory([KnotPoint{n,m}(view(Zdata, iz[k]), (k-1)*h, h) for k = 1:N])
Z[end].dt = 0.0
RD.state(Z[2]) .= 1
RD.control(Z[3]) .= 2.1
@test Zdata[ix[2]] ≈ fill(1,n)
@test Zdata[iu[3]] ≈ fill(2.1,m)
Nc = sum(TO.num_constraints(prob))  # number of stage constraints
Np = N*n + (N-1)*m  # number of primals
Nd = N*n + Nc       # number of duals (no constraints)

## Create PNConstraintSet
A = spzeros(Np + Nd, Np + Nd)
b = zeros(Np +  Nd)
a = trues(Nd)

# Add Hessian blocks
blocks = Altro.SparseBlocks(size(A)...)
for k = 1:N-1
    isdiag = TO.is_diag(prob.obj[k])
    Altro.addblock!(blocks, iz[k], iz[k], isdiag)
end
let k = N
    isdiag = TO.is_diag(prob.obj[k])
    Altro.addblock!(blocks, ix[k], ix[k], isdiag)
end

con = prob.constraints[1]
Altro.getinputinds(con, n, m)

pnconset = Altro.PNConstraintSet(prob.constraints, Z, opts, A, b, a, blocks)
alconset = Altro.ALConstraintSet2{Float64}()
Altro.initialize!(alconset, prob.constraints, prob.Z, opts, zeros(N))
@test length(pnconset) == length(alconset)

copyto!(prob.Z, Z)
prob.Z ≈ Z
Altro.evaluate_constraints!(pnconset, Z)
Altro.evaluate_constraints!(alconset, prob.Z)
for i = 1:length(alconset)
    @test pnconset[i].vals ≈ alconset[i].vals
end

Altro.constraint_jacobians!(pnconset, Z)
Altro.constraint_jacobians!(alconset, prob.Z)
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
