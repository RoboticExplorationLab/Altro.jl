using Test
using ALTRO

TEST_TIME = false

@testset "Benchmark Problems" begin
    include("benchmark_problems.jl")
end
