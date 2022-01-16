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
TO.state_diff_jacobian!(ilqr.model, ilqr.G, ilqr.Z)
G = ilqr.G
@test G[1] != zeros(n,n)
@test G[N] != zeros(n,n)
@test G ≈ res["G"] atol=1e-6

# Dynamics Jacobians
TO.dynamics_expansion!(RD.StaticReturn(), RD.ForwardAD(), ilqr.model, ilqr.D, ilqr.Z)
TO.error_expansion!(ilqr.D, ilqr.model, ilqr.G)
A_ = [Matrix(D.A_) for D in ilqr.D]
B_ = [Matrix(D.B_) for D in ilqr.D]
A = [Matrix(D.A) for D in ilqr.D]
B = [Matrix(D.B) for D in ilqr.D]
@test A_ ≈ res["A_"] atol=1e-6
@test B_ ≈ res["B_"] atol=1e-6
@test A ≈ res["A"] atol=1e-6
@test B ≈ res["B"] atol=1e-6

# Unconstrained Cost Expansion
TO.cost_expansion!(ilqr.quad_obj, ilqr.obj.obj, ilqr.Z, init=true, rezero=true)
TO.error_expansion!(ilqr.E, ilqr.quad_obj, ilqr.model, ilqr.Z, ilqr.G)
hess0 = [Array(E.hess) for E in ilqr.E]
grad0 = [Array(E.grad) for E in ilqr.E]
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
ilqr.ρ[1] = ilqr.opts.bp_reg_min*0
ΔV = Altro.static_backwardpass!(ilqr)
K = Matrix.(ilqr.K)
d = Vector.(ilqr.d)
S = [Matrix(S.Q) for S in ilqr.S]
s = [Vector(S.q) for S in ilqr.S]
Qzz = [[E.Q E.H'; E.H E.R] for E in ilqr.Q]
Qz = [Vector([E.q; E.r]) for E in ilqr.Q]
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