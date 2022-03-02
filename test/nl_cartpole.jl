RD.@autodiff struct CartpoleCost{T} <: TO.CostFunction 
    Q::Vector{T}
    R::Vector{T}
    function CartpoleCost(Q::AbstractVector, R::AbstractVector)
        T = Base.promote_eltype(Q,R)
        new{T}(Q,R)
    end
end
function RD.evaluate(cost::CartpoleCost, x, u)
    y = x[1]
    θ = x[2]
    ydot = x[3]
    θdot = x[4]
    J = cost.Q[2] * (1 .- sin(θ/2))
    J += 0.5* (cost.Q[1] * y^2 + cost.Q[3] * ydot^2 + cost.Q[4] * θdot^2)
    if !isempty(u)
        J +=  0.5 * cost.R[1] * u[1]^2 
    end
    return J
end
RD.state_dim(::CartpoleCost) = 4
RD.control_dim(::CartpoleCost) = 1 
Base.copy(cost::CartpoleCost) = CartpoleCost(copy(cost.Q), copy(cost.R))
RD.default_diffmethod(::CartpoleCost) = RD.ForwardAD()

@testset "Nonlinear Cartpole" begin
## Initialize model
model = RobotZoo.Cartpole()
N = 101 
tf = 3.0
dt = tf / (N-1)

## Set up cost function
Q = [0.1,0.1,0.1,0.1] * dt
R = [0.1] * dt
Qf = [100,1000,100,100.]
costfun = CartpoleCost(Q,R)
cfg = costfun.gradcfg

costfun_term = CartpoleCost(Qf,R*0)
obj = Objective(costfun, costfun_term, N)

## Define problem
x0 = @SVector zeros(4)
xf = SA[0,pi,0,0]

# Solve w/ iLQR
prob = Problem(model, obj, x0, tf, xf=xf)
ilqr = Altro.iLQRSolver2(prob, 
    cost_tolerance=1e-3, gradient_tolerance=1e-2, use_static=Val(false))
ilqr.opts.verbose = 2
b = benchmark_solve!(ilqr)
err = states(ilqr)[end] - xf
@test err'err < 1e-3
# if VERSION >= v"1.5"
#     @test b.allocs == 0
# end

# Add constraints
cons = ConstraintList(4,1,N)
add_constraint!(cons, GoalConstraint(xf), N)
prob = Problem(model, obj, x0, tf, xf=xf, constraints=cons)
solver = ALTROSolver2(prob, cost_tolerance_intermediate=1e-2, show_summary=false, use_static=Val(false))
pn = solver.solver_pn
solver.opts.ρ_primal = 1e-3
solver.opts.projected_newton = true 
solve!(solver)
A = Altro.getKKTMatrix(pn)
istriu(A[1:5,1:5])
Altro.cost_hessian!(pn)
err = states(ilqr)[end] - xf
@test err'err < 1e-4

end