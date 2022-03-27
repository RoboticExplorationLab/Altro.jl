using Altro
using TrajectoryOptimization
using RobotDynamics
using Test
using BenchmarkTools
using SparseArrays
using LinearAlgebra
using RobotDynamics: KnotPoint, SampledTrajectory 
using SparseArrays
using Altro.Cqdldl
const TO = TrajectoryOptimization
const RD = RobotDynamics

# Generate problem
N = 11
prob, opts = Problems.DubinsCar(:parallel_park, N=N)

opts.constraint_tolerance = opts.projected_newton_tolerance
al = Altro.ALSolver(prob, opts, show_summary=false)
solve!(al)
@test iterations(al) == 16

Zsol = get_trajectory(al)
max_violation(al)

pn = Altro.ProjectedNewtonSolver2(prob, opts)
copyto!(pn.Z, Zsol)
copyto!(pn.Z̄data, pn.Zdata)
@test max_violation(pn) ≈ max_violation(al)

# Check trajectories
@test pn.Z ≈ Zsol
@test pn.Z̄ ≈ Zsol

# Check sizes
nx,nu,N = RD.dims(prob)
Np = Altro.num_primals(pn)
@test Np == sum(nx) + sum(nu)

@test size(pn.hess[end]) == (nx[end]+nu[end], nx[end]+nu[end])
@test size(pn.grad[end]) == (nx[end]+nu[end],)

# Check Hessian and regularization
Altro.cost_hessian!(pn)
iz = pn.iz
@test pn.Atop[iz[end], iz[end]] ≈ cat(prob.obj[end].Q, prob.obj[end].R*0, dims=(1,2))
pn.Atop[iz[end],iz[end]]
Altro.primalregularization!(pn)
ρ_primal = opts.ρ_primal
@test pn.Atop[iz[end], iz[end]] ≈ 
    cat(prob.obj[end].Q, prob.obj[end].R*0, dims=(1,2)) + I * ρ_primal

# Compute the factorization
Altro.evaluate_constraints!(pn)
Altro.constraint_jacobians!(pn)
Altro.update_active_set!(pn)

A = Altro.getKKTMatrix(pn)
resize!(pn.qdldl, size(A,1))
Cqdldl.eliminationtree!(pn.qdldl, A)
Cqdldl.factor!(pn.qdldl)
F = Cqdldl.QDLDLFactorization(pn.qdldl)

# Try a step
b = Altro.update_b!(pn)
Np = Altro.num_primals(pn)
@test b[1:Np] ≈ zeros(Np)
@test b[Np+1:end] ≈ -pn.d[pn.active[Np+1:end]]
dY = F \ b
pn.Z̄data .= pn.Zdata .+ dY[1:Np]
Altro.evaluate_constraints!(pn)
viol = max_violation(pn, nothing)
@test viol < max_violation(al)
pn.Zdata .= pn.Z̄data  # accept the step

# Take a second step
b = Altro.update_b!(pn)
dY2 = F \ b
pn.Z̄data .= pn.Zdata .+ dY2[1:Np]
Altro.evaluate_constraints!(pn)
viol2 = max_violation(pn, nothing)
@test viol2 < viol

## Solve the entire thing
N = 11
prob, opts = Problems.DubinsCar(:parallel_park, N=N)

opts.constraint_tolerance = opts.projected_newton_tolerance
al = Altro.ALSolver(prob, opts, show_summary=false)
solve!(al)

Zsol = get_trajectory(al)
pn = Altro.ProjectedNewtonSolver2(prob, opts)
copyto!(pn.Z, Zsol)
pn.Z[end].dt = Inf
Z0data = copy(pn.Zdata)

opts.constraint_tolerance = 1e-10
viol = solve!(pn)
@test viol < 1e-10

# Test allocations
bpn = @benchmark let pn = $pn
    copyto!(pn.Zdata, $Z0data)
    solve!(pn)
end samples=1 evals=1
@test bpn.allocs == 0

#################################
## Basic tests
#################################

# Create a trajectory with views into a single array
nx,nu,N = RD.dims(prob)
h = 0.1
NN = N*nx[1] + (N-1)*nu[1]
Zdata = zeros(NN + nu[1])
n,m = nx[1], nu[1]
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
alconset = Altro.ALConstraintSet{Float64}()
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
copyto!(pn.Z̄data, pn.Zdata)
Altro.cost_gradient!(pn)
Altro.cost_hessian!(pn)
Altro.cost_expansion!(ilqr.obj.obj, ilqr.Efull, ilqr.Z)
for k = 1:N
    @test ilqr.Efull[k].hess ≈ pn.hess[k]
    @test ilqr.Efull[k].grad ≈ pn.grad[k]
end

# Dynamics Jacobian 
Altro.constraint_jacobians!(pn)
Altro.dynamics_expansion!(ilqr)
for k = 1:N-1
    @test ilqr.D[k].∇f ≈ pn.∇f[k]
end

# Dynamics error
Altro.dynamics_error!(pn)
for k = 1:N-1
    @test pn.e[k] ≈ RD.discrete_dynamics(prob.model[1], prob.Z[k]) - RD.state(prob.Z[k+1])
end

# Update everything
let pn = pn
    Altro.evaluate_constraints!(pn)
    Altro.constraint_jacobians!(pn)
    Altro.cost_gradient!(pn)
    Altro.cost_hessian!(pn)
    Altro.update_active_set!(pn)
end
