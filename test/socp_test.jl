using Altro
using TrajectoryOptimization
using StaticArrays
using BenchmarkTools
using RobotZoo
using RobotDynamics 
using StaticArrays, LinearAlgebra
using ForwardDiff
using Test
using FiniteDiff
using Random
const TO = TrajectoryOptimization
const RD = RobotDynamics


# Test Functions 
function Πsoc(x)
    v = x[1:end-1]
    s = x[end]
    a = norm(v)
    if a <= -s
        return zero(x)
    elseif a <= s
        return x
    elseif a >= abs(s)
        x̄ = append!(v, a)
        return 0.5*(1 + s/a) * x̄
    end
    throw(ErrorException("Invalid second-order cone"))
end
function in_soc(x)
    v = x[1:end-1]
    s = x[end]
    a = norm(v)
	a <= s
end

function alcost(x,z,μ)
    c = push(x, u_bnd)
    proj = Πsoc(z - μ .* c)
    1/2μ*(proj'proj - z'z)
end

function auglag(f,g, x,z,μ)
    # ceq = h(x)
    L0 = f(x) #+ y'ceq + 0.5*μ*ceq'ceq

    # cones = g(x)
    # p = length(cones)
    for i = 1:length(z)
        L0 += alcost(x[uinds[i]], z[i], μ)
        # cone = cones[i]
		# proj = Πsoc(z[i] - μ*cones[i])
        # Iμ = I * inv(μ)
		# pen = proj'proj - z[i]'z[i]
        # L0 += 1 / 2μ * pen
    end
    return L0
end

function DI_objective(Q,R,Qf,xf,dt)
    g = Float64[]
    h = Float64[]
    c = 0.0
    for k = 1:N-1
        append!(g, -Q*xf)
        append!(g, zeros(m))
        append!(h, Q.diag)
        append!(h, R.diag)
        c += 0.5*xf'Q*xf
	end
	# g .*= dt
	# h .*= dtk
	# c *= dt
    append!(g, -Qf*xf)
    # append!(g, zeros(m))
    append!(h, Qf.diag)
    # append!(h, R.diag)
    c += 0.5*xf'Qf*xf
    H = Diagonal(h)
    return H,g,c
end


function DI_dynamics(x0, N; dt=0.1, gravity::Bool=false)
    n = length(x0)
    m = n ÷ 2
    D = m
    NN = N*(n+m)

    #  Continuous dynamics
    Ac = zeros(n,n)
    Bc = zeros(n,m)
    for i = 1:D
        Ac[i,D+i] = 1
        Bc[D+i,i] = 1
    end

    # Euler integration
    Ad = I + Ac*dt
    Bd = Bc*dt

    iA = 1:n
    jA = iA
    jB = n .+ (1:m)
    jA2 = jA .+ (n+m)
    
    A = zeros(N*n, NN)
    b = zeros(N*n)
    A[iA,jA] .= I(n)
    b[iA] .= -x0
    iA = iA .+ n
    for k = 1:N-1
        A[iA,jA] .= Ad
        A[iA,jB] .= Bd
        A[iA,jA2] .= -I(n)
        if gravity
            b[iA[D]] = -9.81
        end

        # Advance inds
        iA = iA .+ n
        jA = jA .+ (n+m)
        jB = jB .+ (n+m)
        jA2 = jA2 .+ (n+m)
    end
    return A,b
end

function DI_cones(::Val{D},N) where D
    uinds = SVector{D}(1:D) .+ 2D
    qinds = [uinds .+ (k-1)*3D for k = 1:N-1]
end
DI_cones(D,N) = DI_cones(Val(D),N)  # WARNING: type unstable

## Setup
D,N = 2,11
n,m = 2D,D
dt = 0.1
tf = (N-1)*dt
x0 = @SVector fill(0., n) 
xf = [(@SVector fill(1.,D)); (@SVector fill(0, D))]
Q = Diagonal(@SVector fill(1.0, n)) * dt
R = Diagonal(@SVector fill(1e-1, m)) * dt
Qf = (N-1)*Q * 100
qinds = DI_cones(D,N)
u_bnd = 6.0

# Define batch functions
H_obj,g_obj,c_obj = DI_objective(Q,R,Qf,xf,dt)
Adyn,bdyn = DI_dynamics(x0,N, dt=dt)

di_obj(x) = 0.5*x'H_obj*x + g_obj'x + c_obj
di_dyn(x) = Adyn*x + bdyn
di_soc(x) = [push(x[qi],u_bnd) for qi in qinds]

# Define using ALTRO
model = RobotZoo.DoubleIntegrator(D)

obj = LQRObjective(Q, R, Qf, xf, N)
cons = ConstraintList(n,m,N)
cone = TO.SecondOrderCone()
# cone = TO.Inequality()
u_bnd = 6.0
connorm = NormConstraint(n, m, u_bnd, cone, :control)
add_constraint!(cons, connorm, 1:N-1)
prob = Problem(model, obj, x0, tf, xf=xf, constraints=cons, integration=RD.Euler(model))
solver = Altro.ALSolver(prob, use_static=Val(true))
conSet = TO.get_constraints(solver)
alobj = TO.get_objective(solver)


# Initialization 
inds = LinearIndices(zeros(3D,N))
xinds = [SVector{n}(z[1:n]) for z in eachcol(inds)]
uinds = [SVector{m}(z[n+1:end]) for z in eachcol(inds)][1:end-1]
NN = 3D*N-D 
P = (N-1)*m
x0 = rand(NN)
z = rand.(length.(di_soc(x0)))
z0 = deepcopy(z)
μ = 2.4

X0 = [x0[xi] for xi in xinds]
U0 = [x0[ui] for ui in uinds]
Z0 = SampledTrajectory(X0,U0, dt=dt)
initial_trajectory!(solver, Z0)

## Test objective and soc constraints
z = deepcopy(z0)
Altro.reset_duals!(conSet)
@test cost(alobj, Z0) ≈ di_obj(x0)
@test conSet[1].vals ≈ di_soc(x0)

# copy multipliers to ALconSet
cval = conSet[1]
for i = 1:N-1
	cval.λ[i] .= z[i]
	cval.μ[i] .= μ
end

# test augmented Lagrangian value
LA(x) = auglag(di_obj, di_soc, x, z, μ)
@test di_obj(x0) ≈ cost(alobj.obj, Z0)  # unconstrained cost
@test cost(solver) ≈ LA(x0)
@test !(cost(solver) ≈ di_obj(x0))   # make sure the AL cost isn't the same as the normal cost

λbar = z .- μ .* di_soc(x0)
statuses = [TO.cone_status(TO.SecondOrderCone(), λ) for λ in λbar]
@test :below ∈ statuses

# E = TO.QuadraticObjective(n,m,N)
E0 = Altro.CostExpansion{Float64}(n,m,N)
E = Altro.get_ilqr(solver).Efull
let solver = solver.ilqr
    Altro.cost_expansion!(solver.obj, E, solver.Z)
    @test_throws AssertionError Altro.cost_expansion!(solver.obj, E0, solver.Z)
end
# TO.cost_expansion!(E, alobj, Z0)
grad = vcat([e.grad for e in E]...)[1:end-m]
@test grad ≈ FiniteDiff.finite_difference_gradient(LA, x0)

hess_blocks = vcat([[e.xx, e.uu] for e in E]...)
hess = cat(hess_blocks..., dims=(1,2))[1:end-m, 1:end-m]
@test hess ≈ FiniteDiff.finite_difference_hessian(LA, x0) atol=1e-4

@test all(in_soc.(di_soc(x0)))
@test !all(in_soc.(z .- μ*di_soc(x0)))


# Make it go outside the cone
Random.seed!(1)
z = rand.(length.(di_soc(x0)))
z .*= 100
λbar = z .- μ .* di_soc(x0)
statuses = [TO.cone_status(TO.SecondOrderCone(), λ) for λ in λbar]
for i = 1:20
    if :outside in statuses && :in in statuses
        break
    else
        z .= rand.(length.(di_soc(x0)))
        z .*= 100
        λbar .= z .- μ .* di_soc(x0)
        statuses .= [TO.cone_status(TO.SecondOrderCone(), λ) for λ in λbar]
    end
end
@test :outside ∈ statuses
@test :in ∈ statuses
for i = 1:N-1
	cval.λ[i] .= z[i]
	cval.μ[i] .= μ
    cval.μinv[i] .= inv.(μ)
end
LA(x) = auglag(di_obj, di_soc, x, z, μ)

# E = TO.QuadraticObjective(n,m,N)
# E = Altro.CostExpansion{Float64}(n,m,N)
E = Altro.get_ilqr(solver).Efull
@test LA(x0) ≈ cost(solver, Z0)
Altro.cost_expansion!(alobj, E, Z0)
# TO.cost_expansion!(E, alobj, Z0)
alcon = conSet[1]
Altro.getinputinds(alcon)
grad = vcat([e.grad for e in E]...)[1:end-m]
@test grad ≈ FiniteDiff.finite_difference_gradient(LA, x0) atol=1e-6
[grad FiniteDiff.finite_difference_gradient(LA, x0)]

hess_blocks = vcat([[e.xx, e.uu] for e in E]...)
hess = cat(hess_blocks..., dims=(1,2))[1:end-m, 1:end-m]
@test hess ≈ FiniteDiff.finite_difference_hessian(LA, x0) atol=1e-2

## Solve it 
prob = Problem(model, obj, zero(SVector{n}), tf, xf=xf, constraints=cons, integration=RD.Euler(model))
solver = Altro.ALSolver(prob, verbose=0, show_summary=false, 
	projected_newton=true, penalty_initial=100.0, penalty_scaling=50, 
	cost_tolerance_intermediate=1e-1)
initial_controls!(solver, [rand(D) for k = 1:N])
solve!(solver)
@test all(abs.(norm.(controls(solver))[[1,2,N-2,N-1]] .- u_bnd) .< 1e-5)


# prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons)
# solver = ALTROSolver(prob, show_summary=true, verbose=2, projected_newton=false) 
# @test benchmark_solve!(solver).allocs == 0
# benchmark_solve!(solver)
