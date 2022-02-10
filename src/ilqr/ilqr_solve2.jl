"""
    initialize!(solver::iLQRSolver)

Resets the solver statistics, regularization, and performs the initial 
rollout.

The initial rollout uses the feedback gains store in `solver.K`. These default to 
zero, but if `solve!` is called again it will use the previously cached gains 
to provide better stability.

To reset the gains to zero, you can use [`reset_gains!`](@ref).
"""
function initialize!(solver::iLQRSolver2)
    reset!(solver)  # resets the stats

    # Reset regularization
    solver.reg.ρ = solver.opts.bp_reg_initial
    solver.reg.dρ = 0.0

    # Set the verbosity level
    SolverLogging.setlevel!(solver.logger, solver.opts.verbose)

    # Initial rollout
    # Use a closed-loop rollout, using feedback gains only
    # If the solver was just initialized, this is equivalent to a forward simulation
    # without feedback since the gains are all zero
    rollout!(solver, 0.0)

end

function solve!(solver::iLQRSolver2)
    initialize!(solver)
    lg = solver.logger
    for iter = 1:solver.opts.iterations
        # Calculate the cost
        J_prev = TO.cost(solver)

        # Calculate expansions
        # TODO: do this in parallel
        errstate_jacobians!(solver.model, solver.G, solver.Z)
        dynamics_expansion!(solver)
        error_expansion!(solver.model, solver.D, solver.G)
        cost_expansion!(solver.obj, solver.Efull, solver.Z)
        error_expansion!(solver.model, solver.Eerr, solver.Efull, solver.G, solver.Z)

        # Get next iterate
        backwardpass!(solver)
        Jnew = forwardpass!(solver, J_prev)

        # Accept the step and update the current trajectory
        # This is kept out of the forward pass function to make it easier to 
        # benchmark the forward pass
        copyto!(solver.Z, solver.Z̄)

        # Calculate the gradient of the new trajectory
        dJ = J_prev - Jnew
        grad = gradient!(solver)

        # Record the iteration
        record_iteration!(solver, Jnew, dJ, grad) 

        # Check convergence
        exit = evaluate_convergence(solver)

        # Print log
        if solver.opts.verbose >= 3 
            printlog(lg)
            @log lg "info" ""  # clear the info field
        end

        # Exit
        exit && break
    end
    terminate!(solver)
    return solver
end

function gradient!(solver::iLQRSolver2, Z=solver.Z)
    m = RD.control_dim(solver)
    avggrad = 0.0
    for k in eachindex(solver.d)
        umax = -Inf
        d = solver.d[k]
        u = control(Z[k])
        for i = 1:m
            umax = max(umax, abs(d[i]) / (abs(u[i]) + 1))
        end
        solver.grad[k] = umax
        avggrad += umax
    end
    return avggrad / length(solver.d)
end

function record_iteration!(solver::iLQRSolver2, J, dJ, grad)
    lg = solver.logger
    record_iteration!(solver.stats, cost=J, dJ=dJ, gradient=grad)
    iter = solver.stats.iterations
    if dJ ≈ 0
        solver.stats.dJ_zero_counter += 1
    else
        solver.stats.dJ_zero_counter = 0
    end
    @log lg "cost" J
    @log lg iter
    @log lg dJ
    @log lg grad
    @log lg "dJ_zero" solver.stats.dJ_zero_counter
    @log lg "ρ" solver.reg.ρ
    return nothing
end

function addlogs(solver::iLQRSolver2)
    lg = solver.logger
    iter = -10
    @log lg "cost" 10
    @log lg iter 
    @log lg "grad" -0.02
    @log lg "α" 0.5
    SolverLogging.resetcount!(lg)
    printlog(lg)
end

function evaluate_convergence(solver::iLQRSolver2)
    lg = solver.logger

    # Get current iterations
    i = solver.stats.iterations
    grad = solver.stats.gradient[i]
    dJ = solver.stats.dJ[i]
    J = solver.stats.cost[i]

    # Check for cost convergence
    # must satisfy both 
    if (0.0 <= dJ < solver.opts.cost_tolerance) && (grad < solver.opts.gradient_tolerance) && !solver.stats.ls_failed
        # @logmsg InnerLoop "Cost criteria satisfied."
        @log lg "info" "Cost criteria satisfied" :append
        solver.stats.status = SOLVE_SUCCEEDED
        return true
    end

    # Check total iterations
    if i >= solver.opts.iterations
        # @logmsg InnerLoop "Hit max iterations. Terminating."
        @log lg "info" "Hit max iteration. Terminating" :append
        solver.stats.status = MAX_ITERATIONS
        return true
    end

    # Outer loop update if forward pass is repeatedly unsuccessful
    if solver.stats.dJ_zero_counter > solver.opts.dJ_counter_limit
        # @logmsg InnerLoop "dJ Counter hit max. Terminating."
        @log lg "info" "dJ Counter hit max. Terminating" :append
        solver.stats.status = NO_PROGRESS
        return true
    end

    if J > solver.opts.max_cost_value
        @log lg "info" "Hit maximum cost. Terminating" :append
        solver.stats.status = MAXIMUM_COST
        return true
    end

    return false
end