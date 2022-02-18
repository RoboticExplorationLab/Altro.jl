using Test
using Altro
using TrajectoryOptimization
using RobotZoo
using RobotDynamics
using FileIO
using JLD2
const TO = TrajectoryOptimization
const RD = RobotDynamics

save_results = false 

# TO.diffmethod(::RobotZoo.Cartpole) = RD.ForwardAD()
# @testset "Cartpole" begin
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
n,m,N = RD.dims(solver)
ilqr = Altro.get_ilqr(solver2)
Altro.initialize!(ilqr)
X = states(ilqr)
U = controls(ilqr)
@test X ≈ res["X"] atol=1e-6
@test U ≈ res["U"] atol=1e-6

# State diff Jacobian
Altro.errstate_jacobians!(ilqr.model, ilqr.G, ilqr.Z)
G = ilqr.G
@test all(x->x == Matrix(I,n,n), G[1:N])

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
@test max_violation(al) ≈ 0.01458607 atol=1e-6

##
if save_results
    dx2 = δx
    du2 = δu

    @save joinpath(@__DIR__,"cartpole.jld2") X U G A B hess0 grad0 hess grad K d S s Qzz Qz K2 d2 hess2 grad2 dx2 du2 u2 x3
end

## Projected Newton
res_pn = load("/home/brian/Code/TrajOpt/cartpole_pn.jld2")
solver = ALTROSolver(Problems.Cartpole()..., verbose=0)
solver.opts.constraint_tolerance = solver.opts.projected_newton_tolerance
solve!(solver.solver_al)
@test iterations(solver) == 39
pn = solver.solver_pn
let solver = pn
    Altro.update_constraints!(solver)
    Altro.copy_constraints!(solver)
    Altro.copy_multipliers!(solver)
    Altro.constraint_jacobian!(solver)
    Altro.copy_jacobians!(solver)
    TO.cost_expansion!(solver)
    Altro.update_active_set!(solver; tol=solver.opts.active_set_tolerance_pn)
    Altro.copy_active_set!(solver)
end

# Projection solve
D,d = let solver = pn
    # Update everything
    Altro.update_constraints!(solver)
    Altro.constraint_jacobian!(solver)
    Altro.update_active_set!(solver; tol=solver.opts.active_set_tolerance_pn)
    TO.cost_expansion!(solver)

    # Copy results from constraint sets to sparse arrays
    copyto!(solver.P, solver.Z)
    Altro.copy_constraints!(solver)
    Altro.copy_jacobians!(solver)
    Altro.copy_active_set!(solver)
    Altro.active_constraints(solver)
end
max_violation(pn)
norm(d,Inf)


#
H = pn.H
HinvD = Diagonal(H)\D'
S = Symmetric(D*HinvD)
Sreg = cholesky(S + solver.opts.ρ_chol * I)

# Line Search
dY,dZ = let solver = pn
    a = solver.active_set
    d = solver.d[a]
    viol0 = norm(d,Inf)

    δλ = Altro.reg_solve(S, d, Sreg, 1e-8, 25)
    δZ = -HinvD*δλ
    δλ, δZ
end

let
    pn.P̄.Z .= pn.P.Z + dZ
    copyto!(pn.Z̄, pn.P̄)
    Altro.update_constraints!(pn, pn.Z̄)
    TO.max_violation!(get_constraints(pn))
    Altro.copy_constraints!(pn)
    d = pn.d[pn.active_set]
    copyto!(pn.P.Z, pn.P̄.Z)
end
norm(pn.d[pn.active_set], Inf)

##
vals = [conval.vals for conval in Altro.get_constraints(pn).convals]
@test vals[1] ≈ res_pn["vals"][1] atol=1e-10
@test vals[2][1:end-1] ≈ res_pn["vals"][2][1:end]
@test vals[3] ≈ res_pn["vals"][3] atol=1e-10
@test vals[4] ≈ res_pn["vals"][4] atol=1e-10

##
pn.P.Z ≈ res_pn["Z1"]
pn.d ≈ res_pn["d1"]
pn.active_set ≈ res_pn["a1"]
pn.d[pn.active_set]
res_pn["d1"][res_pn["a1"]]

## Next Line search
let solver = pn, S=(S,Sreg)
    Z̄ = solver.Z̄
    P̄ = solver.P̄
    P = solver.P
    α = 1.0

    d = solver.d[solver.active_set]
    δλ = Altro.reg_solve(S[1], d, S[2], 1e-8, 25)
    @show norm(δλ)
    δZ = -HinvD*δλ
    P̄.Z .= P.Z + α*δZ

    copyto!(Z̄, P̄)
    Altro.update_constraints!(solver, Z̄)
    TO.max_violation!(get_constraints(solver))
    # viol_ = maximum(conSet.c_max)
    Altro.copy_constraints!(solver)
    d = solver.d[solver.active_set]
    viol = norm(d,Inf)
end

# Save the results
copyto!(pn.Z, pn.P)

# Multiplier projection
res = Altro.multiplier_projection!(pn)
@test pn.λ ≈ res_pn["Y"] atol=1e-6


# end