export
	benchmark_solve!,
	shift_fill!

function interp_rows(N::Int,tf::Float64,X::AbstractMatrix)::Matrix
    n,N1 = size(X)
    t1 = range(0,stop=tf,length=N1)
    t2 = collect(range(0,stop=tf,length=N))
    X2 = zeros(n,N)
    for i = 1:n
        interp_cubic = CubicSplineInterpolation(t1, X[i,:])
        X2[i,:] = interp_cubic(t2)
    end
    return X2
end

function convertInf!(A::VecOrMat{Float64},infbnd=1.1e20)
    infs = isinf.(A)
    A[infs] = sign.(A[infs])*infbnd
    return nothing
end

function benchmark_solve!(solver; samples=10, evals=10)
    Z0 = deepcopy(get_trajectory(solver))
    v0 = solver.opts.verbose
    s0 = solver.opts.show_summary
    solver.opts.verbose = 0
    solver.opts.show_summary = false
    b = @benchmark begin
        TO.initial_trajectory!($solver,$Z0)
        solve!($solver)
    end samples=samples evals=evals
    solver.opts.verbose = v0 
    solver.opts.show_summary = s0 
    return b
end

function benchmark_solve!(solver, data::Dict; samples=10, evals=10)
	b = benchmark_solve!(solver, samples=samples, evals=evals)

   # Run stats
   push!(data[:time], time(median(b))*1e-6)  # ms
   push!(data[:iterations], iterations(solver))
   push!(data[:cost], cost(solver))
   return b
end

function shift_fill!(A::Vector, n=1, mode=:copy)
    N = length(A)
    N <= 1 && return nothing
	@inbounds for k = n+1:N
		A[k-n] = A[k]
    end
    if mode == :copy
        a_last = A[N-n]
        @inbounds for k = N-n:N
            A[k] = copy(a_last)
        end
    else mode == :zero
        @inbounds for k = N-n:N
            A[k] .= zero(A[k])
        end
    end
	return nothing
end

"""
    triukkt

Form an upper-triangular KKT matrix with regularization on the dual variables and 
an active set `a`. Takes a matrix `A`, a sparse matrix of size `(Np, Np + Nd)` where `Np` 
is the number of primal variables and `Nd` is the number of dual variables. The vector 
`a` is a vector of Booleans of size `(Np + Nd,)` denoting which columns to remove from `A`.

The CSC data `colptr`, `rowval`, and `nzval` need to be large enough to store the 
resulting information, which should always be the case if `length(colptr) >= n + 1`, 
`length(rowval) >= nnz(A) + Nd` and `length(nzval) >= nnz(A) + Nd`. These vectors 
are updated in-place.

The function returns `nnz_new`, the new number of non-zero entries. A new sparse matrix can 
be created from the updated data:

    n_new = sum(a)
    SparseMatrixCSC(n_new, n_new, colptr[1:n_new+1], rowval[1:nnz_new], nzval[1:nnz_new])

To do this without allocations, you can use the `resize!` command.
"""
function triukkt!(A::SparseMatrixCSC{Tv,Ti}, a::AbstractVector{Bool},
                    colptr, rowval, nzval; reg=zero(Tv)) where {Tv,Ti}
    n,m = size(A)
    m_new = sum(a)
    nnz_new = 0
    for j = 1:m
        nvals = A.colptr[j+1] - A.colptr[j]
        nnz_new += nvals * a[j]
    end

    colptr[1] = 1
    col = 1
    nzcount = 0
    for j = 1:size(A,2)
        nvals = A.colptr[j+1] - A.colptr[j]  # number of elements in the column

        # Keep column if a[j] is true
        if a[j]  
            # Assign next column pointer 
            colptr[col+1] = colptr[col] + nvals

            # Loop over elements in the column, copying into new data
            for i = 1:nvals
                rowval[nzcount + i] = A.rowval[A.colptr[j] + i - 1]
                nzval[nzcount +  i] = A.nzval[A.colptr[j] + i - 1]
            end
            if col > n
                rowval[nzcount + nvals + 1] = col
                nzval[nzcount + nvals + 1] = -reg
                colptr[col+1] += 1
                nzcount += 1
            end
            nzcount += nvals
            col += 1
        end
    end
    m2 = col - 1 - n
    nnz_new += m2
    return nnz_new
end