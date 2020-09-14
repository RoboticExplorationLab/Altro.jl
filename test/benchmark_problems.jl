using Altro
using TrajectoryOptimization
using Test
if !isdefined(Main,:TEST_TIME)
    TEST_TIME = true 
end

# Double Integrator
solver = ALTROSolver(Problems.DoubleIntegrator()...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 1 
@test max_violation(solver) < 1e-6
@test iterations(solver) == 8 # 8
@test solver.stats.gradient[end] < 1e-9
@test status(solver) == Altro.SOLVE_SUCCEEDED 

# Pendulum
solver = ALTROSolver(Problems.Pendulum()...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 2
@test max_violation(solver) < 1e-6
@test iterations(solver) == 19 # 19
@test solver.stats.gradient[end] < 1e-3
@test status(solver) == Altro.SOLVE_SUCCEEDED 

# Cartpole
solver = ALTROSolver(Problems.Cartpole()...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 <  10 
@test max_violation(solver) < 1e-6
@test iterations(solver) == 40 # 40
@test solver.stats.gradient[end] < 1e-2
@test status(solver) == Altro.SOLVE_SUCCEEDED 

solver = ALTROSolver(Problems.Cartpole()..., projected_newton=false)
if !Sys.iswindows()
    @test b = benchmark_solve!(solver).allocs == 0
end

solver = ALTROSolver(Problems.Cartpole()..., projected_newton=false, static_bp=false)
if !Sys.iswindows()
    @test b = benchmark_solve!(solver).allocs == 0
end

# Acrobot
solver = ALTROSolver(Problems.Acrobot()...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 10
@test max_violation(solver) < 1e-6
@test iterations(solver) == 50 # 50
@test solver.stats.gradient[end] < 1e-2
@test status(solver) == Altro.SOLVE_SUCCEEDED 

# Parallel Park
solver = ALTROSolver(Problems.DubinsCar(:parallel_park)...)
b =  benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time /1e6 < 7 
@test max_violation(solver) < 1e-6
@test iterations(solver) == 13 # 13
@test solver.stats.gradient[end] < 1e-3
@test status(solver) == Altro.SOLVE_SUCCEEDED 

# Three Obstacles
solver = ALTROSolver(Problems.DubinsCar(:three_obstacles)...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time /1e6 < 6 
@test max_violation(solver) < 1e-6
@test iterations(solver) == 20 # 20
@test solver.stats.gradient[end] < 1e-2
@test status(solver) == Altro.SOLVE_SUCCEEDED 

solver = ALTROSolver(Problems.DubinsCar(:three_obstacles)..., projected_newton=false)
@test solver.opts.projected_newton == false 
@test solver.stats.gradient[end] < 1e-1
if !Sys.iswindows()   # not sure why this fails on Windows?
    @test benchmark_solve!(solver).allocs == 0
    @test status(solver) == Altro.SOLVE_SUCCEEDED 
end

# Escape
solver = ALTROSolver(Problems.DubinsCar(:escape)..., infeasible=true, R_inf=0.1)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 20
@test max_violation(solver) < 1e-6
@test iterations(solver) == 13 # 13
@test solver.stats.gradient[end] < 1e-3
@test status(solver) == Altro.SOLVE_SUCCEEDED 

# Zig-zag
solver = ALTROSolver(Problems.Quadrotor(:zigzag)...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 60
@test max_violation(solver) < 1e-6
@test iterations(solver) == 15 # 16
@test solver.stats.gradient[end] < 2e-2
@test status(solver) == Altro.SOLVE_SUCCEEDED 

solver = ALTROSolver(Problems.Quadrotor(:zigzag)..., projected_newton=false)
@test solver.stats.gradient[end] < 0.3
if !Sys.iswindows()   # not sure why this fails on Windows?
    @test benchmark_solve!(solver).allocs == 0
    @test status(solver) == Altro.SOLVE_SUCCEEDED 
end

# Barrell Roll
solver = ALTROSolver(Problems.YakProblems()...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 100 
@test max_violation(solver) < 1e-6
@test iterations(solver) == 18 # 18
@test solver.stats.gradient[end] < 1e-3
@test status(solver) == Altro.SOLVE_SUCCEEDED 
