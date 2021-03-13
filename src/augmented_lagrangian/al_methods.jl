
function solve!(solver::AugmentedLagrangianSolver{T,S}) where {T,S}
	initialize!(solver)
    c_max::T = Inf

	conSet = get_constraints(solver)
	solver_uncon = solver.solver_uncon::S

	# Calculate cost
    J_ = TO.get_J(get_objective(solver))
    J = sum(J_)

    # Terminal tolerances
    cost_tol = solver.opts.cost_tolerance
    grad_tol = solver.opts.gradient_tolerance

    for i = 1:solver.opts.iterations_outer
		set_tolerances!(solver, solver_uncon, i, cost_tol, grad_tol)

        # Solve the unconstrained problem
        solve!(solver.solver_uncon)

        # Check solver status
        status(solver) > SOLVE_SUCCEEDED && break

        # Record the updated information
        J = sum(J_)
        TO.max_violation!(conSet)
        c_max = maximum(conSet.c_max)
        record_iteration!(solver, J, c_max)

        # Check for convergence before doing the outer loop udpate
        converged = evaluate_convergence(solver)
        if converged
            break
        end

        # Outer loop update
        dual_update!(solver)
        penalty_update!(solver)

        # Reset verbosity level after it's modified
        set_verbosity!(solver)

        reset!(solver_uncon)

        if i == solver.opts.iterations_outer
            solver.stats.status = MAX_ITERATIONS
        end
    end
    solver.opts.cost_tolerance = cost_tol
    solver.opts.gradient_tolerance = grad_tol
    terminate!(solver)
    return solver
end

function initialize!(solver::AugmentedLagrangianSolver)
	set_verbosity!(solver)
	clear_cache!(solver)

	# Reset solver
    reset!(solver)

	# Calculate cost
    TO.cost!(get_objective(solver), get_trajectory(solver))
end

function step!(solver::AugmentedLagrangianSolver)

    # Solve the unconstrained problem
    solve!(solver.solver_uncon)

    # Outer loop update
    dual_update!(solver)
    penalty_update!(solver)
    TO.max_violation!(get_constraints(solver))

	# Reset verbosity level after it's modified
	set_verbosity!(solver)
end

function record_iteration!(solver::AugmentedLagrangianSolver{T,S}, J::T, c_max::T) where {T,S}

	conSet = get_constraints(solver)
	max_penalty!(conSet)
    max_penalty = maximum(conSet.μ_max)
    J_prev = solver.stats.cost[solver.stats.iterations]
    dJ = J_prev - J
    
    # Just update constraint violation and max penalty
    record_iteration!(solver.stats, c_max=c_max, penalty_max=max_penalty, is_outer=true)
    j = solver.stats.iterations_outer::Int

	if is_verbose(solver) 
        @logmsg OuterLoop :iter value=j
        @logmsg OuterLoop :total value=solver.stats.iterations
        @logmsg OuterLoop :cost value=J
        @logmsg OuterLoop :c_max value=c_max
        @logmsg OuterLoop :penalty value=max_penalty
		print_level(OuterLoop, global_logger())
	end
end

function set_tolerances!(solver::AugmentedLagrangianSolver{T},
        solver_uncon::AbstractSolver{T}, i::Int, 
        cost_tol=solver.opts.cost_tolerance, 
        grad_tol=solver.opts.gradient_tolerance
    ) where T
    if i != solver.opts.iterations_outer
        solver_uncon.opts.cost_tolerance = solver.opts.cost_tolerance_intermediate
        solver_uncon.opts.gradient_tolerance = solver.opts.gradient_tolerance_intermediate
    else
        solver_uncon.opts.cost_tolerance = cost_tol 
        solver_uncon.opts.gradient_tolerance = grad_tol 
    end

    return nothing
end

function evaluate_convergence(solver::AugmentedLagrangianSolver)
	i = solver.stats.iterations
    solver.stats.c_max[i] < solver.opts.constraint_tolerance ||
		solver.stats.penalty_max[i] >= solver.opts.penalty_max
end

"General Dual Update"
function dual_update!(solver::AugmentedLagrangianSolver) where {T,Q,N,M,NM}
    conSet = get_constraints(solver)
	dual_update!(conSet)
end

"General Penalty Update"
function penalty_update!(solver::AugmentedLagrangianSolver)
    conSet = get_constraints(solver)
	penalty_update!(conSet)
end
