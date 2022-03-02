using RobotZoo
using StaticArrays, LinearAlgebra

model = RobotZoo.Cartpole()
n,m = RD.dims(model)
N = 11
x,u = rand(model)
t,dt = 1.1,0.1
z = KnotPoint(x,u,t,dt)


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
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
dyn = TO.DynamicsConstraint(dmodel)

##--- Create the Constraint List
cons = ConstraintList(n,m,N)
add_constraint!(cons, cir, 1:N)
add_constraint!(cons, goal, N)
add_constraint!(cons, lin, 2:N-1)
add_constraint!(cons, bnd, 1:N-1)
# add_constraint!(cons, dyn, 1:N-1)

prob, = Problems.Cartpole(N=N)

##--- Create an ALConstraintSet
ilqr = Altro.iLQRSolver(prob)
conset = Altro.ALConstraintSet{Float64}()
Altro.initialize!(conset, cons, ilqr.Z, ilqr.opts, ilqr.obj.J, ilqr.Efull)
Altro.reset!(conset)

cval = conset[1]
@test cval isa Altro.ALConstraint{Float64, typeof(cir)}
@test cval.inds == cons.inds[1]
@test size(cval.jac[1]) == (RD.output_dim(cir), n)
@test size(cval.jac) == (N,)
cval.jac[1] .= 1
@test cval.jac[1] == ones(RD.output_dim(cir), n)

cval = conset[3]
@test cval isa Altro.ALConstraint{Float64,typeof(lin)}
@test cval.inds == cons.inds[3]
@test size(cval.jac[1]) == (RD.output_dim(lin), n+m)
@test size(cval.jac) == (N-2,)
cval.jac[2] .= 1
cval.jac[2][:,n+1:end] .= 2
@test cval.jac[2] ≈ [ones(RD.output_dim(lin), n) 2*ones(RD.output_dim(lin),m)]

# cval = conset.convals[end]
# @test cval isa Altro.ALConVal{typeof(dyn)}
# @test cval.inds == cons.inds[end]
# @test size(cval.jac[1]) == (n, n+m)
# @test size(cval.jac) == (N,2)
# @test cval.iserr == false
# @test cval.jac[3,2] == zeros(n,n+m)

# Test iteration
@test all([con.con for con in conset] .=== [con for con in cons])
@test length(conset) == length(cons)

##--- Test evaluation
Z = SampledTrajectory{n,m}([rand(n) for k = 1:N], [rand(m) for k = 1:N-1], dt=fill(dt,N-1)) 
Altro.evaluate_constraints!(conset, Z)
# @test conset[end].vals[1] ≈ RD.discrete_dynamics(dmodel, Z[1]) - RD.state(Z[2])
@test conset[3].vals[2] ≈ RD.evaluate(lin, Z[3])
@test conset[2].vals[1] ≈ RD.evaluate(goal, Z[end])

Altro.constraint_jacobians!(conset, Z)
∇c = TO.gen_jacobian(dyn)
xn = zeros(n)
RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), dmodel, ∇c, xn, Z[1])
# @test conset[end].jac[1] ≈ ∇c
# @test conset[end].jac[1,2] ≈ [-I(n) zeros(n,m)]
@test conset[2].jac[1] ≈ I(n)
∇c = TO.gen_jacobian(lin)
c = zeros(RD.output_dim(lin))
RD.jacobian!(lin, ∇c, c, Z[5])
@test conset[3].jac[5] ≈ ∇c ≈ A

##--- Stats functions
TO.evaluate_constraints!(conset)
vals = [conval.vals for conval in conset]
viols = map(enumerate(vals)) do (i,val)
    pos(x) = x > 0 ? x : zero(x)
    if TO.sense(cons[i]) == Inequality()
        [pos.(v) for v in val]
    else
        [abs.(v) for v in val]
    end
end
pos(x) = max(0,x)
c_max = map(enumerate(vals)) do (i,val)
    if TO.sense(cons[i]) == Inequality()
        maximum(maximum.([pos.(c) for c in val]))
    else
        maximum(norm.(val,Inf))
    end
end

@test maximum(c_max) ≈ max_violation(conset)
@test maximum(maximum.([maximum.(viol) for viol in viols])) ≈ max_violation(conset)

p = 2
@test norm([norm(norm.(val,p),p) for val in viols],p) ≈ Altro.normviolation!(conset, p)
p = 1
@test norm([norm(norm.(val,p),p) for val in viols],p) ≈ Altro.normviolation!(conset, p)
p = Inf
@test norm([norm(norm.(val,p),p) for val in viols],p) ≈ Altro.normviolation!(conset, p)
@test !(Altro.normviolation!(conset, Inf) ≈ Altro.normviolation!(conset, 2))

# TODO: test penalty reset
conset[1].opts.usedefault[:penalty_initial] = true
conset[1].opts.penalty_initial = 10
Altro.resetparams!(conset[1])
Altro.reset_penalties!(conset)
@test conset[1].opts.penalty_initial == ilqr.opts.penalty_initial
@test Altro.max_penalty(conset) ≈ 1.0

Altro.setparams!(conset[1], penalty_initial = 10.0)
Altro.resetparams!(conset[1])
@test conset[1].opts.penalty_initial == 10.0 

Altro.reset_penalties!(conset)
@test Altro.max_penalty(conset) ≈ 10.0

conset[3].μ[6] = @SVector fill(30, RD.output_dim(cons[3]))
@test Altro.max_penalty(conset) == 30
Altro.reset_penalties!(conset)
@test Altro.max_penalty(conset) ≈ 10.0

Altro.evaluate_constraints!(conset, Z)
conset[2].vals[1][2] = 2*max_violation(conset)

@test Altro.findmax_violation(conset) == "GoalConstraint at time step 11 at index 2"
Altro.findmax_violation(conset)
conset[4].vals[2][3] = 2*max_violation(conset)
@test Altro.findmax_violation(conset) == "BoundConstraint at time step 2 at x max 3"

#--- AL Updates
Altro.reset_duals!(conset)
Altro.dualupdate!(conset)
@test conset[2].λ[1] ≈ conset[2].vals[1]
@test conset[1].λ[3] ≈ clamp.(10*conset[1].vals[3], 0, Inf)
λ0 = copy(conset[1].λ[3])

Altro.dualupdate!(conset)
@test conset[1].λ[3] ≈ clamp.(λ0 .+ 10*conset[1].vals[3], 0, Inf)
Altro.reset_duals!(conset)
@test conset[1].λ[3] == zeros(RD.output_dim(cons[1]))

Altro.reset_penalties!(conset)
Altro.penaltyupdate!(conset)
@test Altro.max_penalty(conset) ≈ 100
@test conset[2].μ[1] ≈ fill(10, RD.output_dim(cons[2]))
