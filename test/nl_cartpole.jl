struct CartpoleCost{T} <: TO.CostFunction 
    Q::Vector{T}
    R::Vector{T}
end
function TO.stage_cost(cost::CartpoleCost, x, u)
    return TO.stage_cost(cost, x) + 0.5 * cost.R[1] * u[1]^2 
end
function TO.stage_cost(cost::CartpoleCost, x::AbstractVector)
    y = x[1]
    θ = x[2]
    ydot = x[3]
    θdot = x[4]
    J = cost.Q[2] * (1 .- sin(θ/2))
    J += 0.5* (cost.Q[1] * y^2 + cost.Q[3] * ydot^2 + cost.Q[4] * θdot^2)
    return J
end
TO.state_dim(::CartpoleCost) = 4
TO.control_dim(::CartpoleCost) = 1
Base.copy(cost::CartpoleCost) = CartpoleCost(copy(cost.Q), copy(cost.R))

@testset "Nonlinear Cartpole" begin
## Initialize model
model = RobotZoo.Cartpole()
N = 101 
tf = 3.0

## Set up cost function
Q = [0.1,0.1,0.1,0.1]
R = [0.1]
Qf = [100,1000,100,100.]
costfun = CartpoleCost(Q,R)
costfun_term = CartpoleCost(Qf,R*0)
obj = Objective(costfun, costfun_term, N)

## Define problem
x0 = zeros(4)
xf = [0,pi,0,0]

# Solve w/ iLQR
prob = Problem(model, obj, xf, tf, x0=x0)
ilqr = Altro.iLQRSolver(prob, 
    cost_tolerance=1e-3, gradient_tolerance=1e-2)
b = benchmark_solve!(ilqr)
err = states(ilqr)[end] - xf
@test err'err < 1e-3
@test b.allocs == 0

# Add constraints
cons = ConstraintList(4,1,N)
add_constraint!(cons, GoalConstraint(xf), N)
prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons)
solver = ALTROSolver(prob, cost_tolerance_intermediate=1e-2, show_summary=false)
solver.opts.ρ_primal = 1e-3
solve!(solver)
err = states(ilqr)[end] - xf
@test err'err < 1e-4

end