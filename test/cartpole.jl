using RobotZoo
using RobotDynamics
using FileIO
const RD = RobotDynamics

save_results = true 

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

solver = ALTROSolver(prob, opts, save_S=true)
n,m,N = RD.dims(solver)
ilqr = Altro.get_ilqr(solver)
Altro.initialize!(ilqr)
X = states(ilqr)
U = controls(ilqr)
@test X ≈ res["X"] atol=1e-6
@test U ≈ res["U"] atol=1e-6

# State diff Jacobian
TO.state_diff_jacobian!(ilqr.model, ilqr.G, ilqr.Z)
G = ilqr.G
@test G[1] == zeros(n,n)
@test G[N] == zeros(n,n)
@test G ≈ res["G"] atol=1e-6

# Dynamics Jacobians
TO.dynamics_expansion!(RD.StaticReturn(), RD.ForwardAD(), ilqr.model, ilqr.D, ilqr.Z)
TO.error_expansion!(ilqr.D, ilqr.model, ilqr.G)
A = [Matrix(D.A_) for D in ilqr.D]
B = [Matrix(D.B_) for D in ilqr.D]
@test A ≈ res["A"] atol=1e-6
@test B ≈ res["B"] atol=1e-6

# Unconstrained Cost Expansion
TO.cost_expansion!(ilqr.quad_obj, ilqr.obj.obj, ilqr.Z, init=true, rezero=true)
TO.error_expansion!(ilqr.E, ilqr.quad_obj, ilqr.model, ilqr.Z, ilqr.G)
hess0 = [[E.Q E.H'; E.H E.R] for E in ilqr.E]
grad0 = [Vector([E.q; E.r]) for E in ilqr.E]
@test hess0 ≈ res["hess0"] atol=1e-6
@test grad0 ≈ res["grad0"] atol=1e-6

# AL Cost Expansion
TO.cost_expansion!(ilqr.quad_obj, ilqr.obj, ilqr.Z, init=true, rezero=true)
TO.error_expansion!(ilqr.E, ilqr.quad_obj, ilqr.model, ilqr.Z, ilqr.G)
hess = [Array(E.hess) for E in ilqr.E]
grad = [Array(E.grad) for E in ilqr.E]
@test hess ≈ res["hess"] atol=1e-6
@test grad ≈ res["grad"] atol=1e-6

# Backward Pass
ΔV = Altro.static_backwardpass!(ilqr)
K = Matrix.(ilqr.K)
d = Vector.(ilqr.d)
S = [Matrix(S.Q) for S in ilqr.S]
s = [Vector(S.q) for S in ilqr.S]
Qzz = [[E.Q E.H'; E.H E.R] for E in ilqr.Q]
Qz = [Vector([E.q; E.r]) for E in ilqr.Q]
@test K ≈ res["K"] atol=1e-6
@test d ≈ res["d"] atol=1e-6
@test S ≈ res["S"] atol=1e-6
@test s ≈ res["s"] atol=1e-6
@test Qzz ≈ res["Qzz"] atol=1e-6
@test Qz  ≈ res["Qz"] atol=1e-6

# Forward Pass
J = cost(ilqr)
J_new = Altro.forwardpass!(ilqr, ΔV, J)
@test J_new ≈ 3968.9692481
dJ = J - J_new
Altro.copy_trajectories!(ilqr)

Altro.gradient_todorov!(ilqr)
Altro.record_iteration!(ilqr, J, dJ)
Altro.evaluate_convergence(ilqr)

# Perform the second iteration
J_new = Altro.step!(ilqr, J_new, false)
Altro.copy_trajectories!(ilqr)
@test J_new ≈ 1643.4087998
K2 = Matrix.(ilqr.K)
d2 = Vector.(ilqr.d)
@test K2 ≈ res["K2"]
@test d2 ≈ res["d2"]

J_prev = J_new
Altro.set_tolerances!(solver.solver_al, ilqr, 1)
for i = 3:26
    J = Altro.step!(ilqr, J_prev, false)
    println("iter = $i, J = $J")
    dJ = J_prev - J
    J_prev = J
    Altro.copy_trajectories!(ilqr)
    Altro.gradient_todorov!(ilqr)
    Altro.record_iteration!(ilqr, J, dJ)
end

@test Altro.evaluate_convergence(ilqr)
@test J ≈ 1.6782224809

# AL update
al = solver.solver_al
c_max = max_violation(al)
@test c_max ≈ 0.23915446 atol=1e-6
Altro.record_iteration!(al, J, c_max)
@test Altro.evaluate_convergence(al) == false

Altro.dual_update!(al)
Altro.penalty_update!(al)
J_new = cost(al)
@test J_new ≈ 2.2301311286397847

J = Altro.step!(ilqr, J_new, false)
hess2 = [Array(E.hess) for E in ilqr.E]
grad2 = [Array(E.grad) for E in ilqr.E]
@test hess2 ≈ res["hess2"] atol=1e-6
@test grad2 ≈ res["grad2"] atol=1e-6
@test J ≈ 1.8604261241 atol=1e-6
Altro.copy_trajectories!(ilqr)
Altro.gradient_todorov!(ilqr)
Altro.record_iteration!(ilqr, J, dJ)

for i = 2:3
    J = Altro.step!(ilqr, J_prev, false)
    println("iter = $i, J = $J")
    dJ = J_prev - J
    J_prev = J
    Altro.copy_trajectories!(ilqr)
    Altro.gradient_todorov!(ilqr)
    Altro.record_iteration!(ilqr, J, dJ)
end
@test Altro.evaluate_convergence(ilqr)
@test max_violation(al) ≈ 0.01458607 atol=1e-6


##
if save_results
    @save joinpath(@__DIR__,"cartpole.jld2") X U G A B hess0 grad0 hess grad K d S s Qzz Qz K2 d2 hess2 grad2
end

# end