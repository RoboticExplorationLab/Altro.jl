using Altro
using TrajectoryOptimization
using Test
using Altro: ALTROSolver2
using RobotDynamics
const RD = RobotDynamics
const TO = TrajectoryOptimization
if !isdefined(Main,:TEST_TIME)
    TEST_TIME = true 
end
TEST_TIME = false
v = true
ci = haskey(ENV, "CI")

## Double Integrator
v && println("Double Integrator")
solver = ALTROSolver2(Problems.DoubleIntegrator()...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 1 
@test max_violation(solver) < 1e-6
@test iterations(solver) ∈ [8,9] # 8
@test solver.stats.gradient[end] < 1e-9
@test status(solver) == Altro.SOLVE_SUCCEEDED 

## Pendulum
v && println("Pendulum")
solver = ALTROSolver2(Problems.Pendulum()..., verbose=2)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 2
@test max_violation(solver) < 1e-6
@test iterations(solver) == 18 # 17
@test solver.stats.gradient[end] < 1e-3
@test status(solver) == Altro.SOLVE_SUCCEEDED 

## Cartpole
v && println("Cartpole")
solver = ALTROSolver2(Problems.Cartpole()..., save_S=true, verbose=2, use_static=Val(true))
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 <  10 
@test max_violation(solver) < 1e-6
@test iterations(solver) == 41 # 40
@test solver.stats.gradient[end] < 1e-2
@test status(solver) == Altro.SOLVE_SUCCEEDED 

##
solver = ALTROSolver2(Problems.Cartpole()..., projected_newton=false)
b = benchmark_solve!(solver)
iters = iterations(solver)
J = cost(solver)
if !Sys.iswindows() && VERSION > v"1.5"
    @test b.allocs == 0
end

# Use Static arrays
solver = ALTROSolver2(Problems.Cartpole()..., projected_newton=false, use_static=Val(true))
b = benchmark_solve!(solver)
@test iterations(solver) == iters
@test cost(solver) ≈ J
if !Sys.iswindows() && VERSION > v"1.5"
    @test b.allocs == 0
end

solver = Altro.iLQRSolver2(Problems.Cartpole()...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time /1e6 < 10 
!Sys.iswindows() && (@test b.allocs == 0)
@test iterations(solver) == 47
@test status(solver) == Altro.SOLVE_SUCCEEDED

## Acrobot
if !ci
    v && println("Acrobot")
    solver = ALTROSolver2(Problems.Acrobot()...)
    b = benchmark_solve!(solver)
    TEST_TIME && @test minimum(b).time / 1e6 < 15
    @test max_violation(solver) < 1e-6
    @test iterations(solver) == 50 # 50
    @test solver.stats.gradient[end] < 1e-2
    @test status(solver) == Altro.SOLVE_SUCCEEDED 
end

## Parallel Park
if !ci
    v && println("Parallel Park")
    solver = ALTROSolver2(Problems.DubinsCar(:parallel_park)...)
    b =  benchmark_solve!(solver)
    TEST_TIME && @test minimum(b).time /1e6 < 10 
    @test max_violation(solver) < 1e-6
    @test iterations(solver) == 13 # 13
    @test solver.stats.gradient[end] < 1e-3
    @test status(solver) == Altro.SOLVE_SUCCEEDED 

    ## Three Obstacles
    solver = ALTROSolver2(Problems.DubinsCar(:three_obstacles)...)
    b = benchmark_solve!(solver)
    TEST_TIME && @test minimum(b).time /1e6 < 6 
    @test max_violation(solver) < 1e-6
    @test iterations(solver) == 20 # 20
    @test solver.stats.gradient[end] < 1e-1  # 1e-2
    @test status(solver) == Altro.SOLVE_SUCCEEDED 

    solver = ALTROSolver2(Problems.DubinsCar(:three_obstacles)..., projected_newton=false)
    @test solver.opts.projected_newton == false 
    @test solver.stats.gradient[end] < 1e-1
    b = benchmark_solve!(solver)
    if !Sys.iswindows() && VERSION < v"1.5"   # not sure why this fails on Windows?
        @test b.allocs == 0
        @test status(solver) == Altro.SOLVE_SUCCEEDED 
    end
end

## Escape
v && println("Escape")
solver = ALTROSolver2(Problems.DubinsCar(:escape)..., infeasible=true, R_inf=0.1)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 35  # was 25
@test max_violation(solver) < 1e-5
@test iterations(solver) == 14 # 13
@test solver.stats.gradient[end] < 1e-3
@test status(solver) == Altro.SOLVE_SUCCEEDED 

## Zig-zag
v && println("Quadrotor")
solver = ALTROSolver2(Problems.Quadrotor(:zigzag)...)
b = benchmark_solve!(solver)
TEST_TIME && @test minimum(b).time / 1e6 < 60
@test max_violation(solver) < 1e-6
@test iterations(solver) == 14 # 16
@test solver.stats.gradient[end] < 2e-2
@test status(solver) == Altro.SOLVE_SUCCEEDED 

if !ci
    v && println("Quadrotor (no pn)")
    solver = ALTROSolver2(Problems.Quadrotor(:zigzag)..., projected_newton=false)
    b = benchmark_solve!(solver)
    @test iterations(solver) - 60 <= 2 # 60
    @test status(solver) == Altro.SOLVE_SUCCEEDED
    @test solver.stats.gradient[end] < 0.3
    if !Sys.iswindows() && VERSION > v"1.5"  # not sure why this fails on Windows?
        b.allocs == 0
    end

    # # Test infeasible Quadrotor (note that this allocates for the static-bp)
    # v && println("Quadrotor (infeasible)")
    # solver = ALTROSolver(Problems.Quadrotor(:zigzag)..., projected_newton=false, 
    #     infeasible=true, static_bp=false, constraint_tolerance=1e-4, verbose=2)
    # b = benchmark_solve!(solver)
    # @test iterations(solver) == 25
    # @test max_violation(solver) < 1e-4
    # @test solver.stats.gradient[end] < 1e-4
end


# Barrell Roll
# TODO: figure out what's wrong with Yak example
if !ci
    v && println("Barrell Roll")
    solver = ALTROSolver2(Problems.YakProblems(costfun=:QuatLQR, termcon=:quatvec)..., use_static=Val(true))
    b = benchmark_solve!(solver)
    TEST_TIME && @test minimum(b).time / 1e6 < 100 
    @test max_violation(solver) < 1e-6
    @test iterations(solver) == 20 # 20
    @test solver.stats.gradient[end] < 2e-3  # 1e-3
    @test status(solver) == Altro.SOLVE_SUCCEEDED 
end
