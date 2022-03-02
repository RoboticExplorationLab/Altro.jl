using LinearAlgebra
using LoopVectorization
using BenchmarkTools
using StaticArrays
using RecursiveFactorization
function cholfact!(A)
    # fills in lower triangular portion
    # total flops:
    #   2*n*(n+1)*(n-1)/6 + n + n*(n+1)/2
    # total assignments
    #   n*(n+1)*(n-1)/6 + (n+1)/2
    n = size(A,1)
    for j = 1:n

        # (n+1)*(n-1)/6
        for k = 1:j-1  # loop over columns
            for i = j:n
                @inbounds A[i,j] -= A[i,k] * A[j,k]
            end
        end
        ajj = sqrt(A[j,j])  # 1 flop

        # (n+1)/2 flops
        for i = j:n
            @inbounds A[i,j] /= ajj
        end
    end
    return A
end

function avxchol!(A)
    n = size(A,1)
    for j = 1:n

        # (n+1)*(n-1)/6
        for k = 1:j-1  # loop over columns
            @turbo for i = j:n
                A[i,j] -= A[i,k] * A[j,k]
            end
        end
        ajj = sqrt(A[j,j])  # 1 flop

        # (n+1)/2 flops
        @turbo for i = j:n
            A[i,j] /= ajj
        end
    end
end

function _chol_expr(j,n)
    quote
        for k = 1:$j-1  # loop over columns
            @turbo for i = $j:$n
                A[i,$j] -= A[i,k] * A[$j,k]
            end
        end
        ajj = sqrt(A[$j,$j])

        @turbo for i = $j:$n
            A[i,$j] /= ajj
        end
    end
end

@generated function avxchol_unrolled!(A, ::Val{n}) where n
    expr = [_chol_expr(j,n) for j = 1:n]
    quote
        $(Expr(:block, expr...))
    end
end

n = 50
A = randn(n,n); A = A'A;
C = zero(A)
println("Raw Loop")
@btime begin
    $C .= $A
    cholfact!($C)
end

C .= A
avxchol!(C)
LowerTriangular(C) ≈ cholesky(A).L
println("avxchol")
@btime begin
    $C .= $A
    cholfact!($C)
end

println("LAPACK")
@btime begin
    $C .= $A
    LAPACK.potrf!('L', $C)
end

println("Static Arrays")
SM = similar_type(SMatrix{n,n,Float64})
@btime cholesky($(SM(A)))
@btime begin
    Cs = $SM($A) 
    $C .= cholesky(Cs).L
end

p = zeros(Int,n)
lu = copy(A)
RecursiveFactorization.lu!(lu, p)
println("RF lu")
@btime RecursiveFactorization.lu!($lu, $p)