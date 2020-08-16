using Test

stats = SolverStats(parent=:ALTRO)
N = 10
Altro.reset!(stats, N)
@test length(stats.cost) == N
@test length(stats.iteration) == N 
@test sum(stats.cost) == 0
@test sum(stats.iteration) == 0

Altro.record_iteration!(stats, cost=10, c_max=100)
@test stats.cost[1] == 10
@test stats.c_max[1] == 100
@test stats.gradient[1] == Inf
@test stats.penalty_max[1] == Inf
@test stats.iteration[1] == 1
@test stats.iteration_outer[1] == 0
@test stats.iteration_pn[1] == false 
@test stats.cost[2] == 0
@test stats.c_max[2] == 0

stats.iterations_outer += 1
Altro.record_iteration!(stats, cost=5, penalty_max=1e4)
@test stats.cost[2] == 5
@test stats.c_max[2] == 100
@test stats.gradient[2] == Inf
@test stats.penalty_max[2] ≈ 1e4
@test stats.iteration[2] == 2
@test stats.iteration_outer[2] == 1
@test stats.iteration_pn[2] == false

Altro.record_iteration!(stats, c_max=2, gradient=1e-1, is_pn=true) 
@test stats.cost[3] == 5
@test stats.c_max[3] == 2
@test stats.gradient[3] == 1e-1
@test stats.penalty_max[3] == 1e4
@test stats.iteration[3] == 3
@test stats.iteration_outer[3] == 1
@test stats.iteration_pn[3] == true
@test stats.iterations_pn == 1
@test stats.dJ_zero_counter == 0

Altro.trim!(stats)
@test length(stats.cost) == 3
@test stats.c_max[end] == 2
@test stats.gradient[end] == 1e-1

@test (@ballocated Altro.reset!(stats, 10_000) samples=10 evals=10) == 0
@test (@ballocated Altro.record_iteration!($stats, 
    c_max=2, gradient=1e-1, is_pn=true) samples=10 evals=10) == 0


solver = ALTROSolver(Problems.DoubleIntegrator()...)
set_options!(solver, verbose=0, projected_newton=true)
solve!(solver)

# Check if final stats match output
Z0 = copy(get_trajectory(solver))
iters = iterations(solver)
c_max = solver.stats.c_max[end]
J = solver.stats.cost[end]
@test c_max ≈ max_violation(solver)
@test J ≈ cost(solver)

# Test stats to dictionary
@test Dict(solver.stats) isa Dict{Symbol,Vector{<:Any}}

# Make sure ALTRO resets properly
initial_trajectory!(solver, Z0)
solve!(solver)
@test iterations(solver) == iters
@test cost(solver) ≈ J
@test max_violation(solver) ≈ c_max

# Make sure AL solver resets properly
solver = Altro.AugmentedLagrangianSolver(Problems.DoubleIntegrator()...)
Z0 = copy(get_trajectory(solver))
solve!(solver)
J = cost(solver)
iters = iterations(solver)
initial_trajectory!(solver, Z0)
Altro.initialize!(solver)
solve!(solver)
@test J ≈ cost(solver)
@test iters == iterations(solver)