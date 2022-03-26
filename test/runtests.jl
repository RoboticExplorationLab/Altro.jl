using Test
using Altro
using BenchmarkTools
using TrajectoryOptimization
using RobotDynamics
using RobotZoo
using StaticArrays, LinearAlgebra
using JLD2
using ForwardDiff
using FiniteDiff
using Random
using SparseArrays
const RD = RobotDynamics
const TO = TrajectoryOptimization

TEST_TIME = false
Random.seed!(1)

##
# @testset "Initialization" begin
#     include("large_init.jl")
# end

if TEST_TIME
    @testset "Benchmark Suite" begin
        include("benchmark_problems.jl")
    end
end

@testset "Benchmark Problems" begin
    include("nl_cartpole.jl")
    include("cartpole.jl")
    include("quadrotor.jl")
    include("escape_solve.jl")
    include("rocket_test.jl")
    include(joinpath(@__DIR__,"..","examples","quickstart.jl"))
end

@testset "Solvers" begin
    # include("ilqr_test.jl")
    include("constructors.jl")
    include("augmented_lagrangian_tests.jl")
    include("alilqr_test.jl")
    include("ilqr_test.jl")
    include("alcon_test.jl")
    include("alconset_test.jl")
    include("solve_tests.jl")
    include("socp_test.jl")
    include("projected_newton_test.jl")
    include("expansion_test.jl")
    include("infeasible_problem.jl")
end

@testset "Solver Options" begin
    include("solver_opts_test.jl")
    include("solver_stats_test.jl")
end

@testset "Utils" begin
    @testset "QDLDL" begin
        include("qdldl_test.jl")
    end
    @testset "Sparseblocks" begin
        include("sparseblocks_test.jl")
    end
end
