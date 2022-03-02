using Altro
using TrajectoryOptimization
using RobotDynamics
using Test
using BenchmarkTools
using SparseArrays
using LinearAlgebra
using RobotDynamics: KnotPoint, SampledTrajectory
using SparseArrays
import QDLDL
const TO = TrajectoryOptimization
const RD = RobotDynamics

##############################
## Comparison
##############################
prob, opts = Problems.Quadrotor()
# prob, opts = Problems.DubinsCar(:parallel_park, N=11)
n,m = RD.dims(prob)

# prob, opts = Problems.DubinsCar(:parallel_park, N=11)

# s1 = Altro.iLQRSolver(prob, copy(opts))
# s2 = Altro.iLQRSolver2(prob, copy(opts))

opts.trim_stats = false
opts.constraint_tolerance = opts.projected_newton_tolerance
s1 = Altro.AugmentedLagrangianSolver(prob, copy(opts))
s2 = Altro.ALSolver(prob, copy(opts))

solve!(s1)
solve!(s2)

pn1 = Altro.ProjectedNewtonSolver(prob, opts, s1.stats)
pn2 = Altro.ProjectedNewtonSolver2(prob, opts, s2.stats)
copyto!(pn1.Z, get_trajectory(s1))
copyto!(pn2.Z, get_trajectory(s2))
copyto!(pn2.Z̄, pn2.Z)
iterations(s1)
iterations(s2)
norm(states(pn1.Z) - states(pn2.Z))
norm(controls(pn1.Z) - controls(pn2.Z))
@test states(pn1.Z) ≈ states(pn2.Z) atol=1e-8
@test controls(pn1.Z) ≈ controls(pn2.Z) atol=1e-8

Z0 = copy(pn2.Zdata)

@test pn2.conset[1].vals[1].parent === pn2.d
@test pn2.e[1].parent === pn2.d

#
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
@test max_violation(pn1) ≈ max_violation(pn2)

Np = Altro.num_primals(pn2)
Nd = Altro.num_duals(pn2)
ip = 1:Np
id = Np .+ (1:Nd)
A1 = [pn1.H pn1.D'; pn1.D spzeros(Nd, Nd)]
A2 = Symmetric([pn2.Atop; spzeros(Nd, Np + Nd)], :U)
A1 ≈ A2

H1 = pn1.H
H2 = view(pn2.Atop, ip, ip)
H1 ≈ H2
g2 = view(pn2.b, ip)
pn1.g ≈ g2
# @test pn1.D ≈ pn2.D
[pn1.d pn2.d]
@test pn1.d ≈ pn2.d 
@test pn1.active_set ≈ pn2.active[id]
@test all(pn2.active[ip])

Z0data = copy(pn2.Zdata)
Z0 = copy(pn1.Z)
pn2.opts.constraint_tolerance = 1e-10

##
@btime Altro.cost_hessian!($pn2) samples=1 evals=1
@btime let pn = $pn2
    copyto!(pn.Zdata, $Z0data)
    Altro.solve!(pn)
end samples=1 evals=5

pn2.opts.constraint_tolerance = 1e-10
copyto!(pn2.Zdata, Z0)
Altro.solve!(pn2)

@btime let pn = $pn1
    copyto!(pn.Z, $Z0) 
    Altro.solve!(pn)
end
max_violation(pn1)

Altro._qdldl_solve!(pn2)

pn1.opts.constraint_tolerance = 1e-10
pn1.opts.verbose_pn = true
Altro._projection_solve!(pn1)

## Solve using Schur Compliment
ρ_primal = opts.ρ_primal * 0
ρ_chol = opts.ρ_chol

D1,d1 = Altro.active_constraints(pn1)
HinvD1 = Diagonal(pn1.H + I*ρ_primal) \ D1'
S1 = Symmetric(D1*HinvD1)
Sreg1 = cholesky(S1 + ρ_chol*I, check=false)
@test issuccess(Sreg1)
copyto!(pn1.P, pn1.Z)
d1_1 = copy(d1)


α = 1.0
copyto!(pn1.Z̄, pn1.P̄)
δλ1 = Altro.reg_solve(S1, d1, Sreg1, 1e-8, 25)
δZ1 = -HinvD1*δλ1
pn1.P̄.Z .= pn1.P.Z .+ α * δZ1

conset1 = get_constraints(pn1)
copyto!(pn1.Z̄, pn1.P̄)
Altro.update_constraints!(pn1, pn1.Z̄)
TO.max_violation!(conset1)
viol_ = maximum(conset1.c_max)
Altro.copy_constraints!(pn1)
d1 = pn1.d[pn1.active_set]
viol = norm(d1,Inf)

copyto!(pn1.P.Z, pn1.P̄.Z)

# 2nd Iteration
dy1 = copy(δλ1)
dz1 = copy(δZ1)
d1_2 = copy(d1)
δλ1 = Altro.reg_solve(S1, d1, Sreg1, 1e-8, 25)
δZ1 = -HinvD1*δλ1
pn1.P̄.Z .= pn1.P.Z .+ α * δZ1

copyto!(pn1.Z̄, pn1.P̄)
Altro.update_constraints!(pn1, pn1.Z̄)
TO.max_violation!(conset1)
viol_ = maximum(conset1.c_max)
Altro.copy_constraints!(pn1)
d1 = pn1.d[pn1.active_set]
viol = norm(d1,Inf)
d1_3 = copy(d1)

A1 = [pn1.H + I*ρ_primal D1'; D1 -I(length(d1))*1e-8]
b1 = -[zeros(Np); d1]
dY1 = A1 \ b1
A1 * dY1 - b1
norm(A1 * [δZ1; δλ1] - b1)

## Solve using QDLDL
# pn2.Ireg .= -ρ_chol*I(Nd)
# pn2.A[pn2.regblock] .= pn2.Ireg
nnz_new = let pn = pn2
    nnz0 = nnz(pn.Atop) + Nd
    resize!(pn.colptr, Np + Nd + 1)
    resize!(pn.rowval, nnz0)
    resize!(pn.nzval, nnz0)
    Altro.triukkt!(pn.Atop, pn.active, pn.colptr, pn.rowval, pn.nzval, reg=1e-8)
end

colptr = resize!(pn2.colptr, Np + Na + 1) 
rowval = resize!(pn2.rowval, nnz_new) 
nzval = resize!(pn2.nzval, nnz_new) 
Na = sum(pn2.active) - Np
ia = Np .+ (1:Na)
A2 = SparseMatrixCSC(Np + Na, Np + Na, colptr, rowval, nzval)
A2[ip,ia]' ≈ D1
A2[ip,ip] ≈ H1
F = QDLDL.qdldl(A2)

b2 = Altro.update_b!(pn2)
b2[ia] ≈ -d1
b2[ia] ≈ -d1_1
dY2 = F \ b2 
norm(dY2[ip] - δZ1)
norm(dY2[ia] - δλ1)
max_violation(pn2, nothing)
pn2.Z̄data .= pn2.Zdata .+ view(dY, ip)
Altro.evaluate_constraints!(pn2)
max_violation(pn2, nothing)

copyto!(pn2.Zdata, pn2.Z̄data)


a = [trues(Np); pn2.active]
Na = sum(pn2.active)
nnz_new = diff(pn2.A.colptr)'a

Aa = pn2.A[a,a]
A = SparseMatrixCSC(Np + Na, Np + Na, Aa.colptr, Aa.rowval, Aa.nzval)

F = QDLDL.qdldl(A)
pn2.g .= 0
dY = QDLDL.solve(F, -pn2.b)
dY[1:Np]
δZ1

D2,d2 = Altro.active_constraints(pn2)
@btime Altro.active_constraints($pn1)
@btime Altro.active_constraints($pn2)
@test D1 ≈ D2
@test d1 ≈ d2

@time Altro._projection_solve!(pn1)
@time Altro._projection_solve!(pn2)
@btime Altro._projection_solve!($pn1) evals=10
@btime Altro._projection_solve!($pn2) evals=10

ρ_primal = pn1.opts.ρ_primal
ρ_chol = pn1.opts.ρ_chol

norm(d2, Inf)
max_violation(pn2, nothing)

copyto!(pn2.Z̄, pn2.Z)
max_violation(pn2)
Np = Altro.num_primals(pn2)
Nd = Altro.num_duals(pn2)
Na = length(d2)
ip = 1:Np
id = Np .+ (1:Nd)
ia = Np .+ (1:Na)
A3 = spzeros(Np+Na, Np+Na)
A3[ip,ip] .= pn2.H + ρ_chol*I
A3[ip,ia] .= D2'
A3[ia,ia] .= -I(Na)*1e-8
b3 = -[pn2.g*0; d2]

F = QDLDL.qdldl(A3)
dY = QDLDL.solve(F, b3)
p = dY[ip]
λ = dY[ia]
pn2.Z̄data .+= p 

function getAb(pn)
    D,d = Altro.active_constraints(pn)
    Np = Altro.num_primals(pn)
    Na = length(d)
    ip = 1:Np
    ia = Np .+ (1:Na)
    A = spzeros(Np+Na, Np+Na)
    A[ip,ip] .= pn2.H + ρ_chol*I
    A[ip,ia] .= D'
    A[ia,ia] .= -I(Na)*1e-8
    b = -[pn2.g; d]
    return A,b
end

function meritfun(pn, α, p)
    pn.Z̄data .= pn.Zdata .+ α .* p
    Altro.evaluate_constraints!(pn, pn.Z̄)
    v = Altro.norm_violation(pn, 1)
    f = cost(pn, pn.Z̄)

end

function linesearch(pn)
    Np = Altro.num_primals(pn)
    ip = 1:Np
    
    viol0 = max_violation(pn, nothing)
    println("viol0 = $viol0")

    A,b = getAb(pn)
    F = QDLDL.qdldl(A)
    # dY = QDLDL.solve(F,b)
    dY = zero(b)
    p = view(dY,ip)  # primal step

    Na = length(b) - Np
    ia = Np .+ (1:Na)
    D = view(A, ip, ia)'
    
    α = 1.0
    for i = 1:10
        dY .= QDLDL.solve(F,b)
        pn.Z̄data .= pn.Zdata .+ α .* p

        Altro.evaluate_constraints!(pn, pn.Z̄)
        # Altro.update_active_set!(pn)
        v = max_violation(pn, nothing)   # don't update active set
        println("  viol at iter $i = $v")
        if v < viol0
            println("Finished with alpha = $α")
            copyto!(pn.Zdata, pn.Z̄data)
            break
        else
            b[ia] .= pn.d[pn.active]
            α *= 0.5
            # # try a second-order correction
            # b[ia] .= D*p - view(pn.d, pn.active)
            # dY_ = QDLDL.solve(F, b)
            # pn.Z̄data .= pn.Zdata .+ α .* dY_[ip] 
            # Altro.evaluate_constraints!(pn, pn.Z̄)
            # Altro.update_active_set!(pn)
            # v = max_violation(pn, nothing)   # don't update active set
            # if v < viol0
            #     println("Finished with SOC and alpha = $α")
            #     copyto!(pn.Zdata, pn.Z̄data)
            #     break
            # else
            #     b[ia] .= pn.d[pn.active]
            #     α *= 0.5
            # end
        end
    end
end
Z0 = copy(pn2.Zdata)

##
copyto!(pn2.Zdata, Z0)
copyto!(pn2.Z̄, pn2.Z)
cost(pn2)
max_violation(pn2)

let pn = pn2
    Altro.evaluate_constraints!(pn)
    Altro.constraint_jacobians!(pn)
    Altro.cost_gradient!(pn)
    Altro.cost_hessian!(pn)
    Altro.update_active_set!(pn)
end
linesearch(pn2)
cost(pn2)
max_violation(pn2)
Altro.cost_gradient!(pn2)

##
copyto!(pn2.Zdata, Z0)
copyto!(pn2.Z̄, pn2.Z)
let pn = pn2
    Altro.evaluate_constraints!(pn)
    Altro.constraint_jacobians!(pn)
    Altro.cost_gradient!(pn)
    Altro.cost_hessian!(pn)
    Altro.update_active_set!(pn)
end
pn2.opts.constraint_tolerance = 1e-12
Altro._projection_solve!(pn2)


# Second-order correction
Altro.evaluate_constraints!(pn2, pn2.Z̄)
max_violation(pn2, nothing)
d2_ = pn2.d[pn2.active]
b3[ia] .= D2*p - d2_
dY = QDLDL.solve(F, b3)
p = dY[ip]
pn2.Z̄data .= pn2.Zdata .+ p 
Altro.evaluate_constraints!(pn2, pn2.Z̄)
max_violation(pn2, nothing)

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
A1 = [pn1.H D1'; D1 spzeros(Na,Na)]
A2 = [pn2.H D2'; D2 spzeros(Na,Na)]
b1 = [pn1.g; d1]
b2 = [pn2.g; d2]
A1 * [δZ1; δλ1] - b1

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