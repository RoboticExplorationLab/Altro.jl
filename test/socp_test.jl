using Altro
using TrajectoryOptimization
using StaticArrays
using BenchmarkTools
using RobotZoo
using RobotDynamics 
using StaticArrays, LinearAlgebra
using ForwardDiff
using Test
const TO = TrajectoryOptimization
const RD = RobotDynamics


## Test Functions 
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

function auglag(f,g, x,z,μ)
    # ceq = h(x)
    L0 = f(x) #+ y'ceq + 0.5*μ*ceq'ceq

    cones = g(x)
    p = length(cones)
    for i = 1:p
        cone = cones[i]
		proj = Πsoc(z[i] - μ*cones[i])
		pen = proj'proj - z[i]'z[i]
        L0 += 1 / (2μ) * pen
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
	g .*= dt
	h .*= dt
	c *= dt
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
Q = Diagonal(@SVector fill(1.0, n))
R = Diagonal(@SVector fill(1e-1, m))
Qf = (N-1)*Q * 100
qinds = DI_cones(D,N)
u_bnd = 6.0

## Define batch functions
H_obj,g_obj,c_obj = DI_objective(Q,R,Qf,xf,dt)
Adyn,bdyn = DI_dynamics(x0,N, dt=dt)

di_obj(x) = 0.5*x'H_obj*x + g_obj'x + c_obj
di_dyn(x) = Adyn*x + bdyn
di_soc(x) = [push(x[qi],u_bnd) for qi in qinds]

## Define using ALTRO
model = RobotZoo.DoubleIntegrator(D)

obj = LQRObjective(Q, R, Qf, xf, N)
cons = ConstraintList(n,m,N)
cone = TO.SecondOrderCone()
# cone = TO.Inequality()
u_bnd = 6.0
connorm = NormConstraint(n, m, u_bnd, cone, :control)
add_constraint!(cons, connorm, 1:N-1)
prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons, integration=RD.Euler)
solver = Altro.AugmentedLagrangianSolver(prob)
conSet = TO.get_constraints(solver)
alobj = TO.get_objective(solver)

## Initialization 
inds = LinearIndices(zeros(3D,N))
xinds = [SVector{n}(z[1:n]) for z in eachcol(inds)]
uinds = [SVector{m}(z[n+1:end]) for z in eachcol(inds)][1:end-1]
NN = 3D*N-D 
P = (N-1)*m
x0 = rand(NN)
z = rand.(length.(di_soc(x0)))
μ = 2.4

X0 = [x0[xi] for xi in xinds]
U0 = [x0[ui] for ui in uinds]
Z0 = Traj(X0,U0,fill(dt,N))
initial_trajectory!(solver, Z0)

## Test objective and soc constraints
cost(obj, Z0) ≈ di_obj(x0)
conSet.convals[1].vals ≈ di_soc(x0)

# copy multipliers to ALconSet
cval = conSet.convals[1]
for i = 1:N-1
	cval.λ[i] .= z[i]
	cval.μ[i] .= μ
end

# test augmented Lagrangian value
LA(x) = auglag(di_obj, di_soc, x, z, μ)
cost(solver) ≈ LA(x0)
!(cost(solver) ≈ di_obj(x0))   # make sure the AL cost isn't the same as the normal cost

λbar = z .- μ .* di_soc(x0)
statuses = [TO.cone_status(TO.SecondOrderCone(), λ) for λ in λbar]
@test :below ∈ statuses

E = TO.QuadraticObjective(n,m,N)
TO.cost_expansion!(E, alobj, Z0)
grad = vcat([[e.q; e.r] for e in E]...)[1:end-m]
@test grad ≈ ForwardDiff.gradient(LA, x0)

hess_blocks = vcat([[e.Q, e.R] for e in E]...)
hess = cat(hess_blocks..., dims=(1,2))[1:end-m, 1:end-m]
@test hess ≈ ForwardDiff.hessian(LA, x0)

@test all(in_soc.(di_soc(x0)))
@test !all(in_soc.(z .- μ*di_soc(x0)))


# Make it go outside the cone
z = rand.(length.(di_soc(x0)))
z .*= 100
λbar = z .- μ .* di_soc(x0)
statuses = [TO.cone_status(TO.SecondOrderCone(), λ) for λ in λbar]
@test :outside ∈ statuses
@test :in ∈ statuses
for i = 1:N-1
	cval.λ[i] .= z[i]
	cval.μ[i] .= μ
end
LA(x) = auglag(di_obj, di_soc, x, z, μ)

E = TO.QuadraticObjective(n,m,N)
TO.cost_expansion!(E, alobj, Z0)
grad = vcat([[e.q; e.r] for e in E]...)[1:end-m]
@test grad ≈ ForwardDiff.gradient(LA, x0)

hess_blocks = vcat([[e.Q, e.R] for e in E]...)
hess = cat(hess_blocks..., dims=(1,2))[1:end-m, 1:end-m]
@test hess ≈ ForwardDiff.hessian(LA, x0)

## Solve it 
solver = Altro.AugmentedLagrangianSolver(prob, verbose=2, show_summary=true, 
	projected_newton=true, penalty_initial=100.0, penalty_scaling=50, 
	cost_tolerance_intermediate=1e-1)
initial_controls!(solver, [rand(D) for k = 1:N])
solve!(solver)
norm.(controls(solver))
@test all(abs.(norm.(controls(solver))[[1,2,N-2,N-1]] .- u_bnd) .< 1e-6)


## Rocket landing problem
D,N = 3,11
n,m = 2D,D
dt = 0.1
tf = (N-1)*dt
x0 = SA_F64[10,10,100, 0,0,-10] 
xf = @SVector zeros(n)
Q = Diagonal(@SVector fill(1.0, n))
R = Diagonal(@SVector fill(1e-1, m))
Qf = (N-1)*Q * 100

##
model = RobotZoo.DoubleIntegrator(3, gravity=SA[0,0,-9.81])
obj = LQRObjective(Q,R,Qf,xf,N)
cons = ConstraintList(n,m,N)
u_bnd = 400.0
connorm = NormConstraint(n, m, u_bnd, TO.SecondOrderCone(), :control)
add_constraint!(cons, GoalConstraint(xf), N)
add_constraint!(cons, connorm, 1:N-1)

prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons)
solver = ALTROSolver(prob, show_summary=true, verbose=true, projected_newton=false) 
solve!(solver)
unorm = norm.(controls(solver))
println(unorm)
maximum(norm.(controls(solver)))

norm(states(solver)[end] - xf)