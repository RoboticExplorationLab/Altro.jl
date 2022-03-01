
function solve!(solver::ALTROSolver2)
    reset!(solver)
    conSet = get_constraints(solver)

    # Shortcut unconstrained solves
    if isempty(conSet) 
        ilqr = get_ilqr(solver) 
        solve!(ilqr)
        terminate!(solver)
        return solver
    end

    # Set terminal condition if using projected newton
    opts = solver.opts
    ϵ_con = solver.opts.constraint_tolerance
    if opts.projected_newton
        opts_al = solver.solver_al.opts
        if opts.projected_newton_tolerance >= 0
            opts_al.constraint_tolerance = opts.projected_newton_tolerance
        else
            opts_al.constraint_tolerance = 0
            opts_al.kickout_max_penalty = true
        end
    end

    # Solve with AL
    solve!(solver.solver_al)

    if status(solver) <= SOLVE_SUCCEEDED || opts.force_pn  # TODO: should this be status < SOLVE_SUCCEEDED?
        # Check convergence
        i = solver.solver_al.stats.iterations
        if i > 1
            c_max = solver.solver_al.stats.c_max[i]
        else
            c_max = TO.max_violation(solver.solver_al)
        end

        opts.constraint_tolerance = ϵ_con
        if (opts.projected_newton && c_max > opts.constraint_tolerance && 
                (status(solver) <= SOLVE_SUCCEEDED || status(solver) == MAX_ITERATIONS_OUTER)) ||
                opts.force_pn
            tstart = time_ns()
            copyto!(get_trajectory(solver.solver_pn), get_trajectory(solver.solver_al))
            solve!(solver.solver_pn)
            copyto!(get_trajectory(solver.solver_al), get_trajectory(solver.solver_pn))
            tpn = (time_ns() - tstart) / 1e6
            # println("PN took $tpn ms")
        end

        # Back-up check
        if status(solver) <= SOLVE_SUCCEEDED 
            # TODO: improve this check
            if TO.max_violation(solver.solver_al) < solver.opts.constraint_tolerance
                solver.stats.status = SOLVE_SUCCEEDED
            end
        end
    end

    terminate!(solver)
    solver
end