save_results = false

TO.diffmethod(::RobotZoo.Cartpole) = RD.ForwardAD()
@testset "Cartpole" begin
# load expected results
res = load(joinpath(@__DIR__,"cartpole.jld2"))

# Set up problem and solver
prob, opts = Problems.Cartpole()
t = TO.get_times(prob)
U = prob.constraints[1].z_max[end] * 2 * sin.(t .* 2)
U = [[u] for u in U]
initial_controls!(prob, U)
rollout!(prob)

solver = ALTROSolver(prob, opts, save_S=true)
n,m,N = size(solver)
ilqr = Altro.get_ilqr(solver)
Altro.initialize!(ilqr)
X = states(ilqr)
U = controls(ilqr)
@test X ≈ res["X"] atol=1e-6
@test U ≈ res["U"] atol=1e-6

# State diff Jacobian
TO.state_diff_jacobian!(ilqr.G, ilqr.model, ilqr.Z)
G = ilqr.G
@test G[1] == zeros(n,n)
@test G[N] == zeros(n,n)
@test G ≈ res["G"] atol=1e-6

# Dynamics Jacobians
TO.dynamics_expansion!(TO.integration(ilqr), ilqr.D, ilqr.model, ilqr.Z)
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
hess = [[E.Q E.H'; E.H E.R] for E in ilqr.E]
grad = [Vector([E.q; E.r]) for E in ilqr.E]
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

if save_results
    @save joinpath(@__DIR__,"cartpole.jld2") X U G A B hess0 grad0 hess grad K d S s Qzz Qz
end

end