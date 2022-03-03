"""
Sets the tolerance for the iLQR solver to the intermediate tolerances, until the last 
outer loop iteration.
"""
function set_tolerances!(solver::ALSolver{T}, i::Int, cost_tol=solver.opts.cost_tolerance, 
                         grad_tol=solver.opts.gradient_tolerance) where T
    opts = options(solver)
    if i != solver.opts.iterations_outer
        opts.cost_tolerance = opts.cost_tolerance_intermediate
        opts.gradient_tolerance = opts.gradient_tolerance_intermediate
    else
        opts.cost_tolerance = cost_tol 
        opts.gradient_tolerance = grad_tol 
    end

    return nothing
end

function solve!(solver::ALSolver)
    reset!(solver)

    conset = get_constraints(solver)
    if !is_constrained(solver)
        ilqr = get_ilqr(solver)
        solve!(ilqr)
        terminate!(solver)
        return solver
    end

    # Terminal tolerances
    cost_tol = solver.opts.cost_tolerance
    grad_tol = solver.opts.gradient_tolerance

    Z̄ = solver.ilqr.Z̄
    for al_iter = 1:solver.opts.iterations_outer
        # Set potentially looser tolerances for inner iLQR solve 
        set_tolerances!(solver, al_iter, cost_tol, grad_tol)

        # Solve problem with iLQR
        solve!(solver.ilqr)

        # Check solver status
        status(solver) > SOLVE_SUCCEEDED && break

        # Evaluate the constraints and the cost for the current trajectory 
        J = TO.cost(solver, Z̄)  # NOTE: this evaluates the constraints
        c_max = max_violation(conset)
        μ_max = max_penalty(conset)
        record_iteration!(solver, J, c_max, μ_max)

        # Check if it's converged
        isconverged = evaluate_convergence(solver)
        if isconverged
            break
        end

        # Outer loop updates
        dualupdate!(conset)
        penaltyupdate!(conset)

        # Reset iLQR solver
        # TODO: is this necessary? it gets reset at the beginning of the iLQR solve method 
        reset!(solver.ilqr)
    end
    # Reset tolerances to their original values
    solver.opts.cost_tolerance = cost_tol
    solver.opts.gradient_tolerance = grad_tol
    terminate!(solver)
    return solver
end

function record_iteration!(solver::ALSolver, J, c_max, μ_max)
    stats = solver.stats 
    record_iteration!(stats, c_max=c_max, penalty_max=μ_max, is_outer=true)
    lg = getlogger(solver)
    @log lg "iter" stats.iterations
    @log lg "AL iter" stats.iterations_outer
    @log lg "cost" J
    @log lg "||v||" c_max
    @log lg μ_max
end

function evaluate_convergence(solver::ALSolver)
    lg = getlogger(solver)
    iter = solver.stats.iterations
    isconverged = false
    if solver.stats.c_max[iter] < solver.opts.constraint_tolerance
        @log lg "info" "Constraint tolerance met."
        solver.stats.status = SOLVE_SUCCEEDED
        isconverged = true
    end
    if solver.opts.kickout_max_penalty && solver.stats.penalty_max[i] >= solver.opts.penalty_max
        @log lg "info" "Hit max penalty."
        isconverged = true
    end
    if iter >= solver.opts.iterations
        @log lg "info" "Hit max iterations."
        solver.stats.status = MAX_ITERATIONS
        isconverged = true
    end
    if solver.stats.iterations_outer >= solver.opts.iterations_outer
        @log lg "info" "Hit max AL iterations."
        solver.stats.status = MAX_ITERATIONS_OUTER
        isconverged = true
    end
    return isconverged
end