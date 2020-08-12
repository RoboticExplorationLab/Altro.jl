using Test
using Altro

TEST_TIME = false

@testset "Benchmark Problems" begin
    include("benchmark_problems.jl")
end

@testset "Solver Options" begin
    include("solver_opts.jl")
end