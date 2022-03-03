save_results = false

@testset "Quadrotor" begin
# load expected results
res = load(joinpath(@__DIR__,"quadrotor.jld2"))

# Set up problem and solver
prob, opts = Problems.Quadrotor()
times = TO.gettimes(prob)
u0 = zeros(prob.model)[2]
U = [u0 + 0.1*SA[sin(t), sin(t), -sin(t), -sin(t)] for t in times] 
initial_controls!(prob, U)
rollout!(prob)
states(prob)[end]

solver = ALTROSolver(prob, opts, save_S=true)
n,m,N = RD.dims(solver)
ilqr = Altro.get_ilqr(solver)
Altro.initialize!(ilqr)
X = states(ilqr)
U = controls(ilqr)
@test X ≈ res["X"] atol=1e-6
@test U ≈ res["U"] atol=1e-6

# State diff Jacobian
Altro.errstate_jacobians!(ilqr.model, ilqr.G, ilqr.Z)
G = ilqr.G
@test G[1] != zeros(n,n)
@test G[N] != zeros(n,n)
@test G[1:N] ≈ res["G"][1:N] atol=1e-6

# Dynamics Jacobians
Altro.dynamics_expansion!(ilqr)
Altro.error_expansion!(ilqr.model, ilqr.D, ilqr.G)
A_ = [Matrix(D.A) for D in ilqr.D]
B_ = [Matrix(D.B) for D in ilqr.D]
A = [Matrix(D.fx) for D in ilqr.D]
B = [Matrix(D.fu) for D in ilqr.D]
@test A_ ≈ res["A_"] atol=1e-6
@test B_ ≈ res["B_"] atol=1e-6
@test A ≈ res["A"] atol=1e-6
@test B ≈ res["B"] atol=1e-6

# Unconstrained Cost Expansion
Altro.cost_expansion!(ilqr.obj.obj, ilqr.Efull, ilqr.Z)
Altro.error_expansion!(ilqr.model, ilqr.Eerr, ilqr.Efull, ilqr.G, ilqr.Z)
hess0 = [Array(E.hess) for E in ilqr.Eerr]
grad0 = [Array(E.grad) for E in ilqr.Eerr]
@test hess0 ≈ res["hess0"] atol=1e-6
@test grad0 ≈ res["grad0"] atol=1e-6

# AL Cost Expansion
cost(ilqr)  # evaluates the constraints
Altro.cost_expansion!(ilqr.obj, ilqr.Efull, ilqr.Z)
Altro.error_expansion!(ilqr.model, ilqr.Eerr, ilqr.Efull, ilqr.G, ilqr.Z)
hess = [Array(E.hess) for E in ilqr.Eerr]
grad = [Array(E.grad) for E in ilqr.Eerr]
@test hess ≈ res["hess"] atol=1e-6
@test grad ≈ res["grad"] atol=1e-6

# Backward Pass
ilqr.reg.ρ = ilqr.opts.bp_reg_min*0
ΔV = Altro.backwardpass!(ilqr)
K = Matrix.(ilqr.K)
d = Vector.(ilqr.d)
S = [Matrix(S.xx) for S in ilqr.S]
s = [Vector(S.x) for S in ilqr.S]
Qzz = [E.hess for E in ilqr.Q]
Qz = [E.grad for E in ilqr.Q]
@test norm(K - res["K"]) < 1e-6
@test norm(d - res["d"]) < 1e-5
@test S ≈ res["S"] atol=1e-6
@test s ≈ res["s"] atol=1e-6
@test Qzz ≈ res["Qzz"] atol=1e-6
@test Qz  ≈ res["Qz"] atol=1e-6
@test ΔV ≈ res["ΔV"] atol=1e-6

if save_results
    @save joinpath(@__DIR__,"quadrotor.jld2") X U G A B A_ B_ hess0 grad0 hess grad K d S s Qzz Qz ΔV
end

end