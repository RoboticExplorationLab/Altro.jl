using Test
using Altro
using BenchmarkTools
using TrajectoryOptimization

TEST_TIME = false

@testset "Benchmark Problems" begin
    include("benchmark_problems.jl")
end

@testset "Solvers" begin
    include("constructors.jl")
end

@testset "Solver Options" begin
    include("solver_opts.jl")
    include("solver_stats.jl")
end