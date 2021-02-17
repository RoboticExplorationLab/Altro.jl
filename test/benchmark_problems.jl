using Altro
using TrajectoryOptimization
using Test
const TO = TrajectoryOptimization
if !isdefined(Main,:TEST_TIME)
    TEST_TIME = true 
end
v = true

## Double Integrator
v && println("Double Integrator")
solver = ALTROSolver(Problems.DoubleIntegrator()...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 1 
@test max_violation(solver) < 1e-6
@test iterations(solver) == 8 # 8
@test solver.stats.gradient[end] < 1e-9
@test status(solver) == Altro.SOLVE_SUCCEEDED 

## Pendulum
v && println("Pendulum")
solver = ALTROSolver(Problems.Pendulum()...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 2
@test max_violation(solver) < 1e-6
@test iterations(solver) == 19 # 19
@test solver.stats.gradient[end] < 1e-3
@test status(solver) == Altro.SOLVE_SUCCEEDED 

## Cartpole
v && println("Cartpole")
solver = ALTROSolver(Problems.Cartpole()..., save_S=true, verbose=2)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 <  10 
@test max_violation(solver) < 1e-6
@test iterations(solver) == 40 # 40
@test solver.stats.gradient[end] < 1e-2
@test status(solver) == Altro.SOLVE_SUCCEEDED 

##
solver = ALTROSolver(Problems.Cartpole()..., projected_newton=false)
b = benchmark_solve!(solver)
iters = iterations(solver)
J = cost(solver)
if !Sys.iswindows() && VERSION < v"1.5"
    @test b.allocs == 0
end

solver = ALTROSolver(Problems.Cartpole()..., projected_newton=false, static_bp=false)
b = benchmark_solve!(solver)
@test iterations(solver) == iters
@test cost(solver) ≈ J
# Allocations for Julia <v1.5 in backward pass (transpose on StaticArrays)
# if !Sys.iswindows() && VERSION < v"1.5"
#     @test b.allocs == 0
# end

solver = Altro.iLQRSolver(Problems.Cartpole()...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time /1e6 < 10 
@test b.allocs == 0
@test iterations(solver) == 47
@test status(solver) == Altro.SOLVE_SUCCEEDED

## Acrobot
v && println("Acrobot")
solver = ALTROSolver(Problems.Acrobot()...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 15
@test max_violation(solver) < 1e-6
@test iterations(solver) == 50 # 50
@test solver.stats.gradient[end] < 1e-2
@test status(solver) == Altro.SOLVE_SUCCEEDED 

## Parallel Park
v && println("Parallel Park")
solver = ALTROSolver(Problems.DubinsCar(:parallel_park)...)
b =  benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time /1e6 < 8 
@test max_violation(solver) < 1e-6
@test iterations(solver) == 13 # 13
@test solver.stats.gradient[end] < 1e-3
@test status(solver) == Altro.SOLVE_SUCCEEDED 

## Three Obstacles
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
b = benchmark_solve!(solver)
if !Sys.iswindows() && VERSION < v"1.5"   # not sure why this fails on Windows?
    @test b.allocs == 0
    @test status(solver) == Altro.SOLVE_SUCCEEDED 
end

## Escape
v && println("Escape")
solver = ALTROSolver(Problems.DubinsCar(:escape)..., infeasible=true, R_inf=0.1)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 25
@test max_violation(solver) < 1e-5
@test iterations(solver) == 14 # 13
@test solver.stats.gradient[end] < 1e-3
@test status(solver) == Altro.SOLVE_SUCCEEDED 

## Zig-zag
v && println("Quadrotor")
solver = ALTROSolver(Problems.Quadrotor(:zigzag)...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 60
@test max_violation(solver) < 1e-6
@test iterations(solver) == 15 # 16
@test solver.stats.gradient[end] < 2e-2
@test status(solver) == Altro.SOLVE_SUCCEEDED 

solver = ALTROSolver(Problems.Quadrotor(:zigzag)..., projected_newton=false)
b = benchmark_solve!(solver)
@test iterations(solver) - 60 <= 2 # 60
@test status(solver) == Altro.SOLVE_SUCCEEDED
@test solver.stats.gradient[end] < 0.3
if !Sys.iswindows() && VERSION < v"1.5"  # not sure why this fails on Windows?
    b.allocs == 0
end

# Test infeasible Quadrotor (note that this allocates for the static-bp)
solver = ALTROSolver(Problems.Quadrotor(:zigzag)..., projected_newton=false, 
    infeasible=true, static_bp=false, constraint_tolerance=1e-4)
b = benchmark_solve!(solver, samples=2, evals=2)
# if !Sys.iswindows() && VERSION < v"1.5"
#     @test b.allocs == 0
# end
if VERSION < v"1.5"
    @test iterations(solver) == 19 # 20
end
@test status(solver) == Altro.SOLVE_SUCCEEDED

# Barrell Roll
v && println("Barrell Roll")
solver = ALTROSolver(Problems.YakProblems(costfun=:QuatLQR, termcon=:quatvec)...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 100 
@test max_violation(solver) < 1e-6
@test iterations(solver) == 17 # 18
@test solver.stats.gradient[end] < 2e-3  # 1e-3
@test status(solver) == Altro.SOLVE_SUCCEEDED 

solver = ALTROSolver(Problems.YakProblems(costfun=:QuatLQR, termcon=:quatvec)...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 100 
@test iterations(solver) == 17
@test solver.stats.gradient[end] < 2e-3
@test status(solver) == Altro.SOLVE_SUCCEEDED

solver = ALTROSolver(Problems.YakProblems(costfun=:ErrorQuadratic, termcon=:quatvec)..., 
    projected_newton=false, constraint_tolerance=1e-5)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 1000 
@test iterations(solver) == 61 #54
@test solver.stats.gradient[end] < 1e-4
@test status(solver) == Altro.SOLVE_SUCCEEDED

# Check allocations and cache type
ilqr = Altro.iLQRSolver(Problems.YakProblems(costfun=:QuatLQR, termcon=:quatvec)...)
b = benchmark_solve!(ilqr)
@test b.allocs == 0
@test ilqr.exp_cache isa NTuple{4,Nothing}
U = controls(ilqr)

# Make sure FiniteDiff methods don't allocate
ilqr = Altro.iLQRSolver(Problems.YakProblems(costfun=:ErrorQuadratic, termcon=:quatvec)...)
initial_controls!(ilqr, U)
b = benchmark_solve!(ilqr)
@test b.allocs == 0
@test !(ilqr.exp_cache isa NTuple{4,Nothing})