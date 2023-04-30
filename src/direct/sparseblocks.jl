
struct BlockIndices
    i1::UnitRange{Int}
    i2::UnitRange{Int}
    isdiag::Bool
end

# Remove isdiag field from comparison for dict
Base.isequal(b1::BlockIndices, b2::BlockIndices) = b1.i1 == b2.i1 && b1.i2 == b2.i2
Base.hash(b::BlockIndices, h::UInt) = hash((b.i1, b.i2), h)

"""
    SparseBlocks

A type for speeding up expressions like the following:

    A[rows, cols] .= data
    A[rows, cols] .+= data

where `A` is a `SparseMatrixCSC` and `rows` and `cols` are `UnitRange`s. The 
`data` is usually a `Matrix` but will work with any `AbstractArray` or 
collection that supports broadcasting.

The key assumption is that the sparsity in `A` is known and fixed, specified by 
a list of "blocks," or a chunk of consecutive rows and columns, addressable by 
two `UnitRange` objects.

Usage consists of two phases: 1) initialization, where the sparsity structure is 
specified, and 2) usage, where the precomputed data in this struct is used to 
speed up indexing operations.

## Initialization
To specify the sparsity structure, use [`addblock!`](@ref). For example, we could
do the following:

    sb = SparseBlocks(10,10)  # blocks for a 10x10 sparse matrix
    addblock!(sb, 1:2, 1:2)   # nonzeros block in top left corner
    addblock!(sb, 1:3, 8:10)  # nonzeros block in top right corner

After adding all of our blocks, we initialize the sparsity structure via 
[`initialize!`](@ref):

    A = spzeros(10,10)
    initialize!(sb, A)

After calling `initialize!`, the matrix `A` will have the correct number of nonzero 
entries. The speed of the indexing operations is given by using cached conversions 
from the entries of each block to the storage location in the nonzeros vector of `A`,
so it's critical that the sparsity structure of `A` does not change!

## Usage
Once initialized, we can use the cached information in this object to speed up 
writing to our sparse matrix `A`. Basic usage is as simple as wrapping our indices
with an indexing call to our `SparseBlocks` object:

    A[sb[1:2, 1:2]] .= randn(2,2)

Note that these indices must exactly match one of the argument pairs passed to 
[`addblock`](@ref). For best performance, you can cache the [`SparseBlockIndex`](@ref)
returned from this object:

    inds = sb[1:2, 1:2]  # cache this somewhere for future indexing calls
    A[inds] .= randn(2,2)

All of these methods avoid dynamic memory allocations and are about 5-20x faster 
indexing directly into the sparse matrix.
"""
struct SparseBlocks
    n::Int
    m::Int
    inds::Dict{BlockIndices,Matrix{Int}}
    SparseBlocks(n,m) = new(n,m,Dict{BlockIndices,Matrix{Int}}())
end

"""
    addblock!(blocks, rows, cols)

Add a nonzero block for the sparse matrix, spanning the block defined by
   consecutive `rows` and `cols` ranges. All calls to `addblock!` should be 
   made before calling [`initialize!`](@ref).
"""
function addblock!(blocks::SparseBlocks, i1::UnitRange, i2::UnitRange, isdiag::Bool=false)
    if i1.start < 1 || i1.stop > blocks.n || i2.start < 1 || i2.stop > blocks.m
        throw(BoundsError(spzeros(blocks.n, blocks.m), (i1, i2)))
    end
    block = BlockIndices(i1, i2, isdiag)
    if isdiag
        blocks.inds[block] = zeros(Int, min(length(i1), length(i2)), 1)
    else
        blocks.inds[block] = zeros(Int, length(i1), length(i2))
    end
    blocks
end

"""
    initialize!(blocks::SparseBlocks, A::SparseMatrixCSC)

Initialize the sparsity structure specified by `blocks` (via previous calls
to [`addblock!`](@ref)). Writes epsilon values to the nonzero blocks in `A` 
to initialize the data storage for the sparse matrix. The indices into this
storage is computed and stored in `blocks`.
"""
function initialize!(blocks::SparseBlocks, A::SparseMatrixCSC{T}) where T
    n,m = blocks.n, blocks.m
    if size(A) != (n,m)
        throw(DimensionMismatch("Dimension of sparse matrix doesn't match expected size of ($n,$m)."))
    end
    @assert size(A) == (blocks.n, blocks.m)
    for block in keys(blocks.inds) 
        if block.isdiag
            for (i,j) in zip(block.i1, block.i2)
                A[i,j] = eps(T)
            end
        else
            A[block.i1, block.i2] .= eps(T)
        end
    end
    inds = blocks.inds
    for (block,inds) in pairs(blocks.inds)
        rows = block.i1
        cols = block.i2
        n = length(rows)
        if block.isdiag
            for (j,(r,c)) in enumerate(zip(rows, cols))
                nzind = getnzind(A, r, c)
                if !isempty(nzind)
                    inds[j] = nzind[1]
                end
            end
        else
            for (j,col) in enumerate(cols)
                nzind = getnzind(A, rows[1], col)
                if !isempty(nzind)
                    inds[:,j] .= nzind[1] - 1 .+ (1:n)
                end
            end
        end
    end
    blocks
end

# Get the index into the nonzero vector for the given row and column
function getnzind(A,row,col)
    rows = view(A.rowval, nzrange(A,col))
    istart = searchsorted(rows, row) .+ A.colptr[col] .- 1
    return istart
end

"""
    SparseBlockIndex

A custom index for sparse arrays with a fixed sparsity structure, especially for 
    those whose nonzeros appear as dense blocks. This is created by indexing 
    into a [`SparseBlocks`] object, using indices to one of the cached blocks.

This object can then be used as a index for those same indices, creating a 
    [`SparseBlockView`](@ref) object that efficiently maps cartesian indices 
    into the nonzeros vector of the sparse matrix.
"""
struct SparseBlockIndex
    block::BlockIndices
    nzinds::Matrix{Int}
end

function Base.getindex(blocks::SparseBlocks, i1::UnitRange, i2::UnitRange)
    block = BlockIndices(i1, i2, false)
    haskey(blocks.inds, block) || throw(KeyError((i1, i2)))
    SparseBlockIndex(block, blocks.inds[block])
end

"""
    SparseBlockView

A custom view into a sparse matrix with known sparsity, and whose nonzero entries 
appears as dense blocks. Can be used a normal array, where operations on the elements
write and read directly from the nonzeros vector of the sparse matrix.
"""
struct SparseBlockView{Tv,Ti} <: AbstractMatrix{Tv}
    data::SparseMatrixCSC{Tv,Ti}
    block::SparseBlockIndex
end

# Needed to support expressions like A[B] .= data
Broadcast.dotview(A::AbstractSparseMatrix, block::SparseBlockIndex) = SparseBlockView(A, block)

# Define this type when indexing into a SparseMatrixCSC with a SparseBlockIndex
Base.getindex(A::AbstractSparseMatrix, block::SparseBlockIndex) = SparseBlockView(A, block)

# Array interface
Base.size(B::SparseBlockView) = size(B.block.nzinds)
Base.getindex(B::SparseBlockView, index) = getindex(B.data.nzval, B.block.nzinds[index])
Base.getindex(B::SparseBlockView, I::Vararg{Int,2}) = getindex(B.data.nzval, B.block.nzinds[I...])
Base.setindex!(B::SparseBlockView, v, index) = setindex!(B.data.nzval, v, B.block.nzinds[index])
Base.setindex!(B::SparseBlockView, v, I::Vararg{Int,2}) = setindex!(B.data.nzval, v, B.block.nzinds[I...])
Base.IndexStyle(::SparseBlockView) = IndexLinear()

# Broadcasting interface
struct SparseViewStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:SparseBlockView}) = SparseViewStyle()

# Always use the new broadcasting style over others
Base.BroadcastStyle(::SparseViewStyle, ::Broadcast.BroadcastStyle) = SparseViewStyle()

# Handle generic expressions like B .= f.(arg)
function Broadcast.materialize!(B::SparseBlockView, bc::Broadcast.Broadcasted{Broadcast.DefaultMatrixStyle})
    for i in eachindex(bc)
        B[i] = bc[i]
    end
    B
end

# Handle generic expressions like B .= f.(B, g.(args...))
function Broadcast.materialize!(B::SparseBlockView, bc::Broadcast.Broadcasted{SparseViewStyle})
    for i in eachindex(bc)
        B[i] = bc.f(B[i], bc[i])
    end
    B
end

# Specialization of B .= data::AbstractMatrix
function Broadcast.materialize!(B::SparseBlockView, bc::Broadcast.Broadcasted{Broadcast.DefaultMatrixStyle,Nothing,typeof(identity),<:Tuple{<:AbstractMatrix}})
    data = bc.args[1]
    for i in eachindex(B)
        B[i] = data[i] 
    end
    B
end

# Specialization of B .= f.(B, data::AbstractMatrix)
#   which includes B .+= data::AbstractMatrix and variants
function Broadcast.materialize!(B::SparseBlockView, bc::Broadcast.Broadcasted{SparseViewStyle,Nothing,<:Any,<:Tuple{<:SparseBlockView,<:AbstractArray}})
    data = bc.args[2]
    for i in eachindex(B)
        B[i] = bc.f(B[i], data[i])
    end
    B
end

# Specialization of B .= D::Diagonal
function Broadcast.materialize!(B::SparseBlockView, bc::Broadcast.Broadcasted{<:LinearAlgebra.StructuredMatrixStyle{<:Diagonal}, Nothing, typeof(identity), <:Tuple{<:Diagonal}})
    D = bc.args[1]
    if B.block.block.isdiag
        for i in eachindex(B)
            B[i] = D.diag[i]
        end
    else
        for i = 1:minimum(size(B))
            B[i,i] = D.diag[i]
        end
    end
end

function Broadcast.materialize!(B::SparseBlockView, bc::Broadcast.Broadcasted{SparseViewStyle,Nothing,<:Any,<:Tuple{<:SparseBlockView,<:Diagonal}})
    D = bc.args[2]
    f = bc.f
    if B.block.block.isdiag
        for i in eachindex(B)
            B[i] = f(B[i], D.diag[i])
        end
    else
        for i = 1:minimum(size(B))
            B[i,i] = f(B[i], D.diag[i])
        end
    end
end

# Specialize B' .= bc
function Broadcast.materialize!(B::Adjoint{T, <:SparseBlockView{T}}, bc::Broadcast.Broadcasted{Broadcast.DefaultMatrixStyle,Nothing,typeof(identity),<:Tuple{<:AbstractMatrix}}) where T
    data = bc.args[1]
    for i in eachindex(bc)
        it = CartesianIndex(i[2], i[1])
        B.parent[it] = data[i]
    end
    B
end

# Specialize B = UpperTriangular(data)
function Broadcast.materialize!(B::SparseBlockView, bc::Broadcast.Broadcasted{<:LinearAlgebra.StructuredMatrixStyle{<:UpperTriangular}})
    for r = 1:size(B,1)
        for c = r:size(B,2)
            i = CartesianIndex(r,c)
            B[r,c] = bc[i]
        end
    end
end

# Specialize B = LowerTriangular(data)
function Broadcast.materialize!(B::SparseBlockView, bc::Broadcast.Broadcasted{<:LinearAlgebra.StructuredMatrixStyle{<:LowerTriangular}})
    for c = 1:size(B,2)
        for r = c:size(B,1)
            i = CartesianIndex(r,c)
            B[r,c] = bc[i]
        end
    end
end
