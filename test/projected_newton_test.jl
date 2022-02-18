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
prob, opts = Problems.DubinsCar(:turn90, N=N)

# Create a trajectory with views into a single array
n,m,N = RD.dims(prob)
h = 0.1
NN = N*n + (N-1)*m
Zdata = zeros(NN)
ix = [(1:n) .+ (k-1)*(n+m) for k = 1:N]
iu = [n .+ (1:m) .+ (k-1)*(n+m) for k = 1:N-1]
iz = push!([(1:n+m) .+ (k-1)*(n+m) for k = 1:N-1], ix[end])
Z = Traj([KnotPoint{n,m}(view(Zdata, iz[k]), (k-1)*h, h) for k = 1:N])
Z[end].dt = 0.0
RD.state(Z[2]) .= 1
RD.control(Z[3]) .= 2.1
@test Zdata[ix[2]] ≈ fill(1,n)
@test Zdata[iu[3]] ≈ fill(2.1,m)

# Create a PN Solver
ilqr = Altro.iLQRSolver2(prob)
pn = Altro.ProjectedNewtonSolver2(prob)
Np = Altro.num_primals(pn)


# Cost expansion 
copyto!(pn.Z, ilqr.Z)
Altro.cost_gradient!(pn)
Altro.cost_hessian!(pn)
Altro.cost_expansion!(ilqr.obj, ilqr.Efull, ilqr.Z)
for k = 1:N-1
    @test ilqr.Efull[k].hess ≈ pn.hess[k]
    @test ilqr.Efull[k].grad ≈ pn.grad[k]
end
@test ilqr.Efull[N].x ≈ pn.grad[N]
@test ilqr.Efull[N].xx ≈ pn.hess[N]

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


##############################
## Comparison
##############################
prob, opts = Problems.DubinsCar(:turn90, N=11)

s1 = Altro.iLQRSolver(prob, copy(opts))
s2 = Altro.iLQRSolver2(prob, copy(opts))

solve!(s1)
solve!(s2)

pn1 = Altro.ProjectedNewtonSolver(prob, opts)
pn2 = Altro.ProjectedNewtonSolver2(prob, opts)
copyto!(pn1.Z, s1.Z)
copyto!(pn2.Z, s2.Z)
@test states(pn1.Z) ≈ states(pn2.Z)
@test controls(pn1.Z) ≈ controls(pn2.Z)

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

D1,d1 = Altro.active_constraints(pn1)
D2,d2 = Altro.active_constraints(pn1)
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

d1 = pn1.d[pn1.active_set]
norm(d1, Inf)
s2 = pn2.d[pn2.active]
norm(d2, Inf)

Altro._projection_linesearch!(pn1, (S1,Sreg1), HinvD1)
Altro._projection_linesearch!(pn2, (S2,Sreg2), HinvD2)
pn1.P
pn2.Zdata