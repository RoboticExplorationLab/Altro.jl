using RobotZoo
using StaticArrays, LinearAlgebra

model = RobotZoo.Cartpole()
n,m = size(model)
N = 11
x,u = rand(model)
dt = 0.1
z = KnotPoint(x,u,dt)


#--- Generate some constraints
# Circle Constraint
xc = SA[1,1,1]
yc = SA[1,2,3]
r  = SA[1,1,1]
cir = CircleConstraint(n, xc, yc, r)

# Goal Constraint
xf = @SVector rand(n)
goal = GoalConstraint(xf)

# Linear Constraint
p = 5
A = @SMatrix rand(p,n+m)
b = @SVector rand(p)
lin = LinearConstraint(n,m,A,b, Inequality())

# Bound Constraint
xmin = -@SVector rand(n)
xmax = +@SVector rand(n)
umin = -@SVector rand(m)
umax = +@SVector rand(m)
bnd = BoundConstraint(n,m, x_min=xmin, x_max=xmax, u_min=umin, u_max=umax)

# Dynamics Constraint
dyn = TO.DynamicsConstraint(model, N)

##--- Create the Constraint List
cons = ConstraintList(n,m,N)
add_constraint!(cons, cir, 1:N)
add_constraint!(cons, goal, N)
add_constraint!(cons, lin, 2:N-1)
add_constraint!(cons, bnd, 1:N-1)
add_constraint!(cons, dyn, 1:N-1)

##--- Create an ALConstraintSet
conset = Altro.ALConstraintSet(cons, model)
Altro.reset!(conset, SolverOptions())
@test all(conset.convals .=== conset.errvals)

cval = conset.convals[1]
@test cval isa Altro.ALConVal{typeof(cir)}
@test cval.inds == cons.inds[1]
@test size(cval.jac[1]) == (length(cir), n)
@test size(cval.jac) == (N,1)
cval.jac[1] .= 1
@test cval.jac[1] == ones(length(cir), n)
@test cval.iserr == false

cval = conset.convals[3]
@test cval isa Altro.ALConVal{typeof(lin)}
@test cval.inds == cons.inds[3]
@test size(cval.jac[1]) == (length(lin), n+m)
@test size(cval.jac) == (N-2,1)
cval.jac[2] .= 1
cval.jac[2][:,n+1:end] .= 2
@test cval.jac[2] ≈ [ones(length(lin), n) 2*ones(length(lin),m)]
@test cval.iserr == false

cval = conset.convals[end]
@test cval isa Altro.ALConVal{typeof(dyn)}
@test cval.inds == cons.inds[end]
@test size(cval.jac[1]) == (n, n+m)
@test size(cval.jac) == (N-1,2)
@test cval.iserr == false
@test cval.jac[3,2] == zeros(n,n)

# Test iteration
@test all([con for con in conset] .=== [con for con in cons])
@test length(conset) == length(cons)

##--- Test evaluation
Z = Traj([rand(n) for k = 1:N], [rand(m) for k = 1:N-1], fill(dt,N)) 
TO.evaluate!(conset, Z)
@test conset.errvals[end].vals[1] ≈ discrete_dynamics(RK3, model, Z[1]) - state(Z[2])
@test conset.errvals[3].vals[2] ≈ TO.evaluate(lin, Z[3])
@test conset.errvals[2].vals[1] ≈ TO.evaluate(goal, Z[end])

TO.jacobian!(conset, Z)
∇c = TO.gen_jacobian(dyn)
discrete_jacobian!(RK3, ∇c, model, Z[1])
@test conset.errvals[end].jac[1] ≈ ∇c
@test conset.errvals[end].jac[1,2] ≈ -I(n)
@test conset.errvals[2].jac[1] ≈ I(n)
∇c = TO.gen_jacobian(lin)
TO.jacobian!(∇c, lin, Z[5])
@test conset.errvals[3].jac[5] ≈ ∇c ≈ A

##--- Stats functions
vals = [conval.vals for conval in conset.convals]
viols = map(enumerate(vals)) do (i,val)
    pos(x) = x > 0 ? x : zero(x)
    if TO.sense(cons[i]) == Inequality()
        [pos.(v) for v in val]
    else
        [abs.(v) for v in val]
    end
end
c_max = map(enumerate(vals)) do (i,val)
    if TO.sense(cons[i]) == Inequality()
        maximum(maximum.(val))
    else
        maximum(norm.(val,Inf))
    end
end
@test maximum(c_max) ≈ max_violation(conset)
maximum(c_max)
@test maximum(maximum.([maximum.(viol) for viol in viols])) ≈ max_violation(conset)

p = 2
@test norm([norm(norm.(val,p),p) for val in viols],p) ≈ TO.norm_violation(conset, p)
p = 1
@test norm([norm(norm.(val,p),p) for val in viols],p) ≈ TO.norm_violation(conset, p)
p = Inf
@test norm([norm(norm.(val,p),p) for val in viols],p) ≈ TO.norm_violation(conset, p)
@test !(TO.norm_violation(conset, Inf) ≈ TO.norm_violation(conset, 2))
@test_throws ArgumentError TO.norm_violation(conset, 3)

# TODO: test penalty reset
@test Altro.max_penalty(conset) ≈ 1.0
conset.convals[3].μ[6] = @SVector fill(30, length(cons[3]))
@test Altro.max_penalty(conset) == 30
Altro.reset_penalties!(conset)
@test Altro.max_penalty(conset) ≈ 1.0
conset.convals[1].params.μ0 = 10
Altro.reset_penalties!(conset)
@test Altro.max_penalty(conset) ≈ 10

TO.evaluate!(conset, Z)
conset.convals[2].vals[1][2] = 2*max_violation(conset)
@test TO.findmax_violation(conset) == "GoalConstraint at time step 11 at index 2"
conset.convals[4].vals[2][3] = 2*max_violation(conset)
@test TO.findmax_violation(conset) == "BoundConstraint at time step 2 at x max 3"

#--- AL Updates
Altro.reset_duals!(conset)
Altro.dual_update!(conset)
@test conset.convals[2].λ[1] ≈ conset.convals[2].vals[1]
@test conset.convals[1].λ[3] ≈ clamp.(10*conset.convals[1].vals[3], 0, Inf)
λ0 = copy(conset.convals[1].λ[3])

Altro.dual_update!(conset)
@test conset.convals[1].λ[3] ≈ clamp.(λ0 .+ 10*conset.convals[1].vals[3], 0, Inf)
Altro.reset_duals!(conset)
@test conset.convals[1].λ[3] == zeros(length(cons[1]))

Altro.reset_penalties!(conset)
Altro.penalty_update!(conset)
@test Altro.max_penalty(conset) ≈ 100
@test conset.convals[2].μ[1] ≈ fill(10, length(cons[2]))

#--- Active Set
Altro.dual_update!(conset)
Altro.update_active_set!(conset)
@test conset.convals[2].active[1] == ones(n)
@test conset.convals[end].active == [ones(n) for k = 1:N-1]
for i = 1:length(conset.convals[1].active)
    for j = 1:length(cons[1])
        if conset.convals[1].active[i][j]
            @test (conset.convals[1].vals[i][j] > 0) | (conset.convals[1].λ[i][j] > 0)
        else
            @test (conset.convals[1].vals[i][j] <= 0) & (conset.convals[1].λ[i][j] <= 0)
        end
    end
end

#--- Cost
J = zeros(N)
TO.cost!(J, conset.convals[2]) #, conset.λ[2], conset.μ[2], conset.active[2])
@test J[1:N-1] == zeros(N-1)
@test J[N] ≈ (conset.convals[2].λ[1] + 0.5* conset.convals[2].μ[1] .* conset.convals[2].vals[1])' *
    conset.convals[2].vals[1]
TO.cost!(J, conset.convals[3])#, conset.λ[3], conset.μ[3], conset.active[3])
@test J[2] ≈ (conset.convals[3].λ[1] + 0.5* conset.convals[3].μ[1] .* conset.convals[3].active[1] .* conset.convals[3].vals[1])' *
    conset.convals[3].vals[1]
@test J[1] ≈ 0

E = TO.QuadraticObjective(n,m,N)
TO.cost_expansion!(E, conset.convals[1]) #, conset.λ[1], conset.μ[1], conset.active[1])
cx = conset.convals[1].jac[1]
Iμ = Diagonal(conset.convals[1].active[1] .* conset.convals[1].μ[1])
g = Iμ * conset.convals[1].vals[1] .+ conset.convals[1].λ[1]
@test E[1].Q ≈ cx'Iμ*cx + I(n)
@test E[1].q ≈ cx'g

E = TO.QuadraticObjective(n,m,N)
i = 3
k = 4
TO.cost_expansion!(E, conset.convals[i])#, conset.λ[i], conset.μ[i], conset.active[i])
cx = conset.convals[i].jac[k]
Iμ = Diagonal(conset.convals[i].active[k] .* conset.convals[i].μ[k])
H = cx'Iμ*cx + I
g = cx'*(Iμ * conset.convals[i].vals[k] .+ conset.convals[i].λ[k])
@test E[k+1].Q ≈ H[1:n,1:n]
@test E[k+1].H ≈ H[1:n, n+1:end]'
@test E[k+1].R ≈ H[n+1:end, n+1:end]
@test E[k+1].q ≈ g[1:n]
@test E[k+1].r ≈ g[n+1:end]

#--- Link constraints
cons2 = ConstraintList(n,m,N)
add_constraint!(cons2, cir, 1:N)
add_constraint!(cons2, goal, N)
add_constraint!(cons2, lin, 2:N-1)
add_constraint!(cons2, dyn, 1:N-1)
conset2 = Altro.ALConstraintSet(cons2, model)

max_violation(conset2) ≈ 0
@test TO.findmax_violation(conset2) == "No constraints violated"

@test conset.convals[1].con === conset2.convals[1].con
@test conset.convals[1].vals !== conset2.convals[1].vals
@test conset.convals[1].jac !== conset2.convals[1].jac

@test conset.convals[3].con === conset2.convals[3].con
@test conset.convals[3].vals !== conset2.convals[3].vals
@test conset.convals[3].jac !== conset2.convals[3].jac

@test conset.convals[end].con === conset2.convals[end].con
@test conset.convals[end].vals !== conset2.convals[end].vals
@test conset.convals[end].jac !== conset2.convals[end].jac

Altro.link_constraints!(conset2, conset)
@test conset.convals[1].vals === conset2.convals[1].vals
@test conset.convals[1].jac === conset2.convals[1].jac
@test conset.convals[3].vals === conset2.convals[3].vals
@test conset.convals[3].jac === conset2.convals[3].jac
@test conset.convals[end].vals === conset2.convals[end].vals
@test conset.convals[end].jac === conset2.convals[end].jac
@test length(conset) == 5
@test length(conset2) == 4