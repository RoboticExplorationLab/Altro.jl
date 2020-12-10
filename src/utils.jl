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

function set_logger()
    if !(global_logger() isa SolverLogger)
        global_logger(SolverLogging.default_logger(true))
    end
end

function benchmark_solve!(solver; samples=10, evals=10)
    Z0 = deepcopy(get_trajectory(solver))
    if is_constrained(solver)
        λ0 = deepcopy(get_duals(solver))
    else
        λ0 = nothing
    end
    v0 = solver.opts.verbose
    s0 = solver.opts.show_summary
    solver.opts.verbose = 0
    solver.opts.show_summary = false
    b = @benchmark begin
        TO.initial_trajectory!($solver,$Z0)
        # set_duals!($solver, $λ0)
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
