using Altro
using SparseArrays
using Altro: SparseBlocks
using Test


## Test
using BenchmarkTools
A = spzeros(10,10)
blocks = [(2:3,1:2), (3:4,8:10), (5:7,2:2), (4:5,7:9)]

# A = spzeros(100,100)
# blocks = [(2:30,1:20), (3:40,80:100), (50:70,20:20), (20:50,70:90)]

sb = SparseBlocks(10,10)
for block in blocks
    Altro.addblock!(sb, block...)
end
Altro.initialize!(sb, A)

@test_throws MethodError sb[1]
@test_throws KeyError sb[1:3,1:2]
idx = sb[2:3,1:2] 
@test idx isa Altro.SparseBlockIndex
@test size(idx.nzinds) == length.(blocks[1]) 
@test idx.block.i1 == 2:3
@test idx.block.i2 == 1:2

@test_throws DimensionMismatch Altro.initialize!(sb, spzeros(10,12)) 
Altro.initialize!(sb, A)
@test nnz(A) == sum([prod(length.(block)) for block in blocks]) - 2  # 2 overlapping
@test A.rowval[1] == 2
@test A.rowval[2] == 3
@test A.rowval[3] == 2
@test A.rowval[4] == 3
@test A.rowval[5] == 5

v = A[idx]
@test size(v) == (2,2)
@test v[1] == A[2,1]
@test v[2] == A[3,1]
@test v[1,2] == A[2,2]

function testallocs(sb::SparseBlocks, A)
    @allocated idx = sb[2:3,1:2] 
    @allocated v = A[idx] 
    data = randn(size(v))
    @allocated A[idx] .= data
    @allocated A[idx] .+= data
end

@test testallocs(sb, A) == 0


# Performance comp
A = spzeros(100,100)
blocks = [(2:30,1:20), (3:40,80:100), (50:70,20:20), (20:50,70:90)]

sb = SparseBlocks(size(A)...)
for block in blocks
    Altro.addblock!(sb, block...)
end
Altro.initialize!(sb, A)

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
