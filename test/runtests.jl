using Test
using Altro
using BenchmarkTools
using TrajectoryOptimization
using RobotDynamics
using RobotZoo
using StaticArrays, LinearAlgebra
using JLD2
using FileIO
using FiniteDiff
const RD = RobotDynamics
const TO = TrajectoryOptimization

TEST_TIME = false

@testset "Benchmark Problems" begin
    if !haskey(ENV, "CI")
        include("benchmark_problems.jl")
    end
    include("nl_cartpole.jl")
    include("cartpole.jl")
    include("quadrotor.jl")
    include(joinpath(@__DIR__,"..","examples","quickstart.jl"))
end

@testset "Solvers" begin
    include("constructors.jl")
    include("augmented_lagrangian_tests.jl")
    include("solve_tests.jl")
    include("finite_diff.jl")
    include("socp_test.jl")
end

@testset "Solver Options" begin
    include("solver_opts.jl")
    include("solver_stats.jl")
end