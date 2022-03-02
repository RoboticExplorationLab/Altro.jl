using Test
using Altro
using TrajectoryOptimization
using RobotZoo
using RobotDynamics
using JLD2
using LinearAlgebra
const TO = TrajectoryOptimization
const RD = RobotDynamics

save_results = false 

@testset "Cartpole" begin
# load expected results
res = load(joinpath(@__DIR__,"cartpole.jld2"))

## Set up problem and solver
prob, opts = Problems.Cartpole()
t = TO.gettimes(prob)
U = prob.constraints[1].z_max[end] * 2 * sin.(t .* 2)
U = [[u] for u in U]
initial_controls!(prob, U)
rollout!(prob)

solver2 = Altro.ALTROSolver2(prob, opts, save_S=true)
n,m,N = RD.dims(solver2)
ilqr = Altro.get_ilqr(solver2)
Altro.initialize!(ilqr)
X = states(ilqr)
U = controls(ilqr)
@test X ≈ res["X"] atol=1e-6
@test U ≈ res["U"] atol=1e-6

# State diff Jacobian
Altro.errstate_jacobians!(ilqr.model, ilqr.G, ilqr.Z)
G = ilqr.G
@test all(x->x ≈ Matrix(I,n,n), G[1:N])

# Dynamics Jacobians
Altro.dynamics_expansion!(ilqr)
Altro.error_expansion!(ilqr.model, ilqr.D, ilqr.G)
A = [Matrix(D.A) for D in ilqr.D]
B = [Matrix(D.B) for D in ilqr.D]
@test A ≈ res["A"] atol=1e-6
@test B ≈ res["B"] atol=1e-6

# Unconstrained Cost Expansion
Altro.cost_expansion!(ilqr.obj.obj, ilqr.Efull, ilqr.Z)
Altro.error_expansion!(ilqr.model, ilqr.Eerr, ilqr.Efull, ilqr.G, ilqr.Z)
hess0 = [[E.xx E.ux'; E.ux E.uu] for E in ilqr.Efull]
grad0 = [Vector([E.x; E.u]) for E in ilqr.Efull]
@test hess0 ≈ res["hess0"] atol=1e-6
@test grad0 ≈ res["grad0"] atol=1e-6

# AL Cost Expansion
cost(ilqr, ilqr.Z)
Altro.cost_expansion!(ilqr.obj, ilqr.Efull, ilqr.Z)
Altro.error_expansion!(ilqr.model, ilqr.Eerr, ilqr.Efull, ilqr.G, ilqr.Z)
hess = [Array(E.hess) for E in ilqr.Efull]
grad = [Array(E.grad) for E in ilqr.Efull]
@test hess ≈ res["hess"] atol=1e-6
@test grad ≈ res["grad"] atol=1e-6

# Backward Pass
ΔV = Altro.backwardpass!(ilqr)
K = Matrix.(ilqr.K)
d = Vector.(ilqr.d)
S = [Matrix(S.xx) for S in ilqr.S]
s = [Vector(S.x) for S in ilqr.S]
Qzz = [[E.xx E.ux'; E.ux E.uu] for E in ilqr.Q]
Qz = [Vector([E.x; E.u]) for E in ilqr.Q]
@test K ≈ res["K"] atol=1e-6
@test d ≈ res["d"] atol=1e-6
@test S ≈ res["S"] atol=1e-6
@test s ≈ res["s"] atol=1e-6
@test Qzz ≈ res["Qzz"] atol=1e-6
@test Qz  ≈ res["Qz"] atol=1e-6

# Forward Pass
J = cost(ilqr)
TO.rollout!(ilqr, 1.0)
@test cost(ilqr.obj, ilqr.Z̄) ≈ 42918.1164958687
δx = RD.state_diff(ilqr.model, RD.state(ilqr.Z̄[1]), RD.state(ilqr.Z[1]))
δu = d[1]
RD.setcontrol!(ilqr.Z̄[1], RD.control(ilqr.Z[1]) + δu)
RD.setstate!(ilqr.Z̄[2], RD.discrete_dynamics(ilqr.model, ilqr.Z̄[1]))
RD.discrete_dynamics(ilqr.model, ilqr.Z̄[1])
@test δx ≈ zeros(n)
@test RD.control(ilqr.Z̄[1]) ≈ [-13.987381820157527]
@test RD.state(ilqr.Z̄[2]) ≈ [-0.017484227275196912, 0.034968454550393824, -0.6980029115441456, 1.3840172471758034]

δx = RD.state_diff(ilqr.model, RD.state(ilqr.Z̄[2]), RD.state(ilqr.Z[2]))
δu = d[2] + K[2] * δx
# RD.control(ilqr.Z[2]) + δu
RD.setcontrol!(ilqr.Z̄[2], RD.control(ilqr.Z[2]) + δu)
RD.setstate!(ilqr.Z̄[3], RD.discrete_dynamics(ilqr.model, ilqr.Z̄[2]))
@test δx ≈ res["dx2"]
@test δu ≈ res["du2"]

u2 = RD.control(ilqr.Z̄[2])
x3 = RD.state(ilqr.Z̄[3])
@test u2 ≈ res["u2"]
@test x3 ≈ res["x3"]


J_new = Altro.forwardpass!(ilqr, J)
@test J_new ≈ 3968.9692481
dJ = J - J_new
copyto!(ilqr.Z, ilqr.Z̄)

grad = Altro.gradient!(ilqr)
Altro.record_iteration!(ilqr, J, dJ, grad)
Altro.evaluate_convergence(ilqr)

# Perform the second iteration
J_new = let solver = ilqr, J_prev=J_new
    Altro.errstate_jacobians!(solver.model, solver.G, solver.Z)
    Altro.dynamics_expansion!(solver)
    Altro.error_expansion!(solver.model, solver.D, solver.G)
    Altro.cost_expansion!(solver.obj, solver.Efull, solver.Z)
    Altro.error_expansion!(solver.model, solver.Eerr, solver.Efull, solver.G, solver.Z)
    Altro.backwardpass!(solver)
    Altro.forwardpass!(solver, J_prev)
end
copyto!(ilqr.Z, ilqr.Z̄)
@test J_new ≈ 1643.4087998
K2 = Matrix.(ilqr.K)
d2 = Vector.(ilqr.d)
@test K2 ≈ res["K2"]
@test d2 ≈ res["d2"]

##
J_prev = J_new
Altro.set_tolerances!(solver2.solver_al, 1)
J = 0.0
for i = 3:26
    J = let solver = ilqr
        J_prev = cost(solver)
        Altro.errstate_jacobians!(solver.model, solver.G, solver.Z)
        Altro.dynamics_expansion!(solver)
        Altro.error_expansion!(solver.model, solver.D, solver.G)
        Altro.cost_expansion!(solver.obj, solver.Efull, solver.Z)
        Altro.error_expansion!(solver.model, solver.Eerr, solver.Efull, solver.G, solver.Z)
        Altro.backwardpass!(solver)
        Altro.forwardpass!(solver, J_prev)
    end
    # println("iter = $i, J = $J")
    dJ = J_prev - J
    J_prev = J
    copyto!(ilqr.Z, ilqr.Z̄)
    grad = Altro.gradient!(ilqr)
    Altro.record_iteration!(ilqr, J, dJ, grad)
end

@test Altro.evaluate_convergence(ilqr)
@test J ≈ 1.6782224809

# AL update
al = solver2.solver_al
c_max = max_violation(al)
@test c_max ≈ 0.23915446 atol=1e-6
μ_max = Altro.max_penalty(get_constraints(al))
Altro.record_iteration!(al, J, c_max, μ_max)
@test Altro.evaluate_convergence(al) == false

Altro.dualupdate!(get_constraints(al))
Altro.penaltyupdate!(get_constraints(al))
J_new = cost(al)
@test J_new ≈ 2.2301311286397847

# J = Altro.step!(ilqr, J_new, false)
J = let solver = ilqr
    J_prev = cost(solver)
    Altro.errstate_jacobians!(solver.model, solver.G, solver.Z)
    Altro.dynamics_expansion!(solver)
    Altro.error_expansion!(solver.model, solver.D, solver.G)
    Altro.cost_expansion!(solver.obj, solver.Efull, solver.Z)
    Altro.error_expansion!(solver.model, solver.Eerr, solver.Efull, solver.G, solver.Z)
    Altro.backwardpass!(solver)
    Altro.forwardpass!(solver, J_prev)
end
hess2 = [Array(E.hess) for E in ilqr.Eerr]
grad2 = [Array(E.grad) for E in ilqr.Eerr]
@test hess2 ≈ res["hess2"] atol=1e-6
@test grad2 ≈ res["grad2"] atol=1e-6
@test J ≈ 1.8604261241 atol=1e-6
copyto!(ilqr.Z, ilqr.Z̄)
grad = Altro.gradient!(ilqr)
Altro.record_iteration!(ilqr, J, dJ, grad)

for i = 2:3
    local J = let solver = ilqr
        J_prev = cost(solver)
        Altro.errstate_jacobians!(solver.model, solver.G, solver.Z)
        Altro.dynamics_expansion!(solver)
        Altro.error_expansion!(solver.model, solver.D, solver.G)
        Altro.cost_expansion!(solver.obj, solver.Efull, solver.Z)
        Altro.error_expansion!(solver.model, solver.Eerr, solver.Efull, solver.G, solver.Z)
        Altro.backwardpass!(solver)
        Altro.forwardpass!(solver, J_prev)
    end
    # println("iter = $i, J = $J")
    local dJ = J_prev - J
    local J_prev = J
    copyto!(ilqr.Z, ilqr.Z̄)
    local grad = Altro.gradient!(ilqr)
    Altro.record_iteration!(ilqr, J, dJ, grad)
end
@test Altro.evaluate_convergence(ilqr)
@test max_violation(al) ≈ 0.01458607 atol=1e-6

##
if save_results
    dx2 = δx
    du2 = δu

    @save joinpath(@__DIR__,"cartpole.jld2") X U G A B hess0 grad0 hess grad K d S s Qzz Qz K2 d2 hess2 grad2 dx2 du2 u2 x3
end

## Projected Newton
resfile_pn = joinpath(@__DIR__, "cartpole_pn.jld2")
res_pn = load(resfile_pn)
solver = Altro.ALTROSolver2(Problems.Cartpole()..., verbose=0)
solver.opts.constraint_tolerance = solver.opts.projected_newton_tolerance
solve!(solver.solver_al)
@test iterations(solver) == 39
pn = solver.solver_pn
copyto!(get_trajectory(pn), get_trajectory(solver.solver_al))
pn.Z[end].dt = Inf
copyto!(pn.Z̄data, pn.Zdata)
let pn = pn
    Altro.evaluate_constraints!(pn)
    Altro.constraint_jacobians!(pn)
    Altro.cost_hessian!(pn)
    Altro.update_active_set!(pn)
end

# Matrix Factorization
A = Altro.getKKTMatrix(pn)
@test A ≈ res_pn["A"] atol=1e-6
Np = Altro.num_primals(pn)
@test size(A,1) == sum(pn.active)
@test istriu(A)
@test A[end,end] ≈ -1e-8
b = Altro.update_b!(pn)
@test length(b) == sum(pn.active)
@test b[Np+1:end] ≈ -pn.d[pn.active[Np+1:end]]
viol0 = norm(b,Inf)
@test viol0 ≈ max_violation(pn)
@test viol0 ≈ res_pn["viol0"]

resize!(pn.qdldl, sum(pn.active))
Altro.Cqdldl.eliminationtree!(pn.qdldl, A)
Altro.Cqdldl.factor!(pn.qdldl)
F = Altro.Cqdldl.QDLDLFactorization(pn.qdldl)

dY1 = F \ b
pn.Z̄data .= pn.Zdata .+ dY1[1:Np]
Altro.evaluate_constraints!(pn)
viol1 = max_violation(pn, nothing)
@test viol1 < viol0
copyto!(pn.Zdata, pn.Z̄data)
@test dY1 ≈ res_pn["dY1"] atol=1e-6
@test viol1 ≈ res_pn["viol1"]


## Next Line search
Altro.update_b!(pn)
dY2 = F \ b
pn.Z̄data .= pn.Zdata .+ dY2[1:Np]
Altro.evaluate_constraints!(pn)
viol2 = max_violation(pn, nothing)
cost0 = cost(solver.solver_al)
cost_final = cost(pn)
@test (cost_final - cost0) / cost0 * 100 < 0.1
Zpn = copy(pn.Z̄data)
@test dY2 ≈ res_pn["dY2"] atol=1e-6
@test viol2 ≈ res_pn["viol2"]
@test cost_final ≈ res_pn["cost_final"]
@test Zpn ≈ res_pn["Zpn"]

if save_results
    @save resfile_pn viol0 viol1 viol2 dY1 dY2 Zpn cost_final A
end

end
