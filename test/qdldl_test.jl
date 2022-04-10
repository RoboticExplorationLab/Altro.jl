# include(joinpath(@__DIR__, "..", "src", "qdldl.jl"))
using Altro.Cqdldl
# using Main.Cqdldl
using SparseArrays
using LinearAlgebra
using Test
using Random

function testallocs(solver::Cqdldl.QDLDLSolver, U, b)
    A = Symmetric(U, :U)
    allocs = @allocated resize!(solver, size(U,1))
    allocs += @allocated Cqdldl.eliminationtree!(solver, U)
    allocs += @allocated Cqdldl.factor!(solver)
    x = copy(b)
    allocs += @allocated Cqdldl.solve!(solver, x)
    @test A*x ≈ b
    @test allocs == 0
end

Random.seed!(1)
n, m = 5, 5
H = Diagonal(rand(n)) 
D = sprand(m, n, 0.5)
U = [H D'; spzeros(m ,n) Diagonal(fill(-1e-8, m))]
A = Symmetric(U, :U)
b = randn(n + m)

nnzA = nnz(U)
solver = Cqdldl.QDLDLSolver{Float64}(n+m, 2nnzA, 2*nnzA)
testallocs(solver, U, b)

# Change the number nonzero elements
U2 = copy(U)
U2[8,9] = 0.4
@test nnz(U2) > nnz(U)
testallocs(solver, U2, b)

# Change the size of the matrix
a = trues(n + m)
a[8] = false
U3 = U[a,a]
n3 = sum(a)
b3 = randn(size(U3,2))
@test istriu(U3)
testallocs(solver, U3, b3)

F = Cqdldl.QDLDLFactorization(solver)
x3 = F \ b3
@test Symmetric(U3, :U) * x3 ≈ b3