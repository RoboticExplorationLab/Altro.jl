using Altro
using Altro: SparseBlocks, BlockIndices, addblock!, initialize!, SparseBlockIndex
using SparseArrays
using LinearAlgebra
# include(joinpath(@__DIR__,"../src/direct/sparseblocks.jl"))
using Test


## Test
block1 = BlockIndices(1:2, 3:4, false)
block2 = BlockIndices(1:2, 3:4, true)
block3 = BlockIndices(1:2, 3:5, false)
block1 == block2
d = Dict(block1=>1)
@test d[block2] == 1
@test_throws KeyError d[block3]

A = spzeros(10,10)
blocks = [(2:3,1:2), (3:4,8:10), (5:7,2:2), (2:5,7:9), (8:10,1:4)]

sb = SparseBlocks(10,10)
for block in blocks
    addblock!(sb, block...)
end
initialize!(sb, A)

@test_throws MethodError sb[1]
@test_throws KeyError sb[1:3,1:2]
idx = sb[2:3,1:2] 
@test idx isa SparseBlockIndex
@test size(idx.nzinds) == length.(blocks[1]) 
@test idx.block.i1 == 2:3
@test idx.block.i2 == 1:2

@test_throws DimensionMismatch initialize!(sb, spzeros(10,12)) 
A = spzeros(10,10)
initialize!(sb, A)
@test nnz(A) == sum([prod(length.(block)) for block in blocks]) - 4  # 4 overlapping
@test A.rowval[1] == 2
@test A.rowval[2] == 3
@test A.rowval[3] == 8
@test A.rowval[4] == 9
@test A.rowval[5] == 10 
@test A.rowval[6] == 2 

v = A[idx]
@test size(v) == (2,2)
v .= reshape(1:4, 2,2)
@test A[2,1] == 1
@test A[3,1] == 2 
@test A[2,2] == 3 
@test A[3,2] == 4 

# Assign transpose
vt = v'
vt[2,1] = 10
@test A[2,2] == 10

# Upper/Lower Triangular
data = [1 2; 3 4.0]
v .= 0
v .= UpperTriangular(data)
@test A[2:3, 1:2] ≈ UpperTriangular(data)
v .= 0
v .= LowerTriangular(data)
@test A[2:3, 1:2] ≈ LowerTriangular(data)


# Copy from a diagonal block
D = Diagonal(randn(2))
# NOTE: this doesn't set the off-diagonal elements to zero! 
A[3,1] = 4
v .= D
@test A[2,1] == D[1,1]
@test A[3,2] == D[2,2]

function testallocs(sb::SparseBlocks, A)
    allocs = @allocated idx = sb[2:3,1:2] 
    allocs += @allocated v = A[idx] 
    data = randn(size(v))
    allocs += @allocated A[idx] .= data
    allocs += @allocated A[idx] .+= data
    D = Diagonal(randn(size(v,1)))
    allocs += @allocated A[idx] .= D
    allocs += @allocated A[idx] .+= D
end

@test testallocs(sb, A) == 0
nnz0 = nnz(A)


## Diagonal block
using LinearAlgebra
addblock!(sb, 8:10, 8:10, true)
A = spzeros(10,10)
initialize!(sb, A)
@test nnz(A) == nnz0 + 3 
idx = sb[8:10,8:10]
@test idx.block.isdiag
@test size(idx.nzinds) == (3,1)
@test idx.nzinds[1] ∈ nzrange(A, 8)
@test idx.nzinds[2] ∈ nzrange(A, 9)
@test idx.nzinds[3] ∈ nzrange(A, 10)

function testallocs_diag(sb, A)
    allocs = @allocated idx = sb[8:10,8:10]
    @test idx.block.isdiag
    allocs += @allocated v = A[idx]
    D = Diagonal(randn(length(v)))
    allocs += @allocated v .= D
    allocs += @allocated v .+= D
    allocs
end
@test testallocs_diag(sb, A) == 0


# Performance comp
run_benchmarks = false
if run_benchmarks
using BenchmarkTools
A = spzeros(100,100)
blocks = [(2:30,1:20), (3:40,80:100), (50:70,20:20), (20:50,70:90)]

sb = SparseBlocks(size(A)...)
for block in blocks
    addblock!(sb, block...)
end
initialize!(sb, A)

inds = [sb[block...] for block in blocks]
data = [randn(length.(block)) for block in blocks]
@btime for (d,block) in zip($data,$blocks)
    rows, cols = block
    $A[$sb[rows,cols]] .+= d
end
@btime for (d,idx) in zip($data,$inds)
    $A[idx] .+= d
end
vblocks = [A[idx] for idx in inds]
@btime for (d,block) in zip($data,$vblocks)
    block .+= d
end
@btime for (d,block) in zip($data,$blocks)
    rows, cols = block
    v = view($A, rows, cols)
    v .+= d
end

addblock!(sb, 50:100, 50:100, true)
A = spzeros(100,100)
initialize!(sb, A)

D = Diagonal(randn(51))
ind = sb[50:100, 50:100]
v = A[ind]
@btime $A[$ind] .= $D
@btime $v .= $D
@btime $A[50:100,50:100] .= $D

@btime $A[$ind] .+= $D
@btime $v .+= $D
@btime $A[50:100,50:100] .+= $D
@btime view($A,50:100,50:100) .+= $D
end