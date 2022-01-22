
function rollout!(solver::iLQRSolver2, α)
    N = solver.N
    Z = solver.Z; Z̄ = solver.Z̄
    K = solver.K; d = solver.d;
    δx = solver.dx; δu = solver.du

    RD.setstate!(Z̄[1], solver.x0)
    sig = solver.opts.dynamics_funsig
    for k = 1:N-1
        RD.state_diff!(solver.model, δx[k], state(Z̄[k]), state(Z[k]))
        δu[k] .= d[k] .* α .+ control(Z[k])
        mul!(δu[k], K[k], δx[k], 1.0, 1.0)
        RD.setcontrol!(Z̄[k], δu[k])
        RD.propagate_dynamics!(sig, solver.model, Z̄[k+1], Z̄[k])

        max_x = norm(state(Z̄[k+1]),Inf)
        if max_x > solver.opts.max_state_value || isnan(max_x)
            solver.stats.status = STATE_LIMIT
            return false
        end
        max_u = norm(control(Z̄[k+1]),Inf)
        if max_u > solver.opts.max_control_value || isnan(max_u)
            solver.stats.status = CONTROL_LIMIT 
            return false
        end
    end
    solver.stats.status = UNSOLVED
    return true
end

function forwardpass!(solver::iLQRSolver2, J_prev) 
    Z = solver.Z; Z̄ = solver.Z̄
    ΔV = solver.ΔV
    ϕ = solver.opts.line_search_decrease_factor
    z_lb = solver.opts.line_search_lower_bound
    z_ub = solver.opts.line_search_upper_bound

    α = 1.0
    J = Inf
    z = Inf
    
    J_ = TO.get_J(solver.obj)
    solver.stats.ls_failed = false
    max_iters = solver.opts.iterations_linesearch
    for i = 1:max_iters
        # Forward simulate with the current line search length
        isrolloutgood = rollout!(solver, α)

        # Check if forward simulation fails
        if !isrolloutgood
            α *= ϕ 
            continue
        end

        # Calculate the cost for the new trajectory
        TO.cost!(solver.obj, Z̄)
        J = sum(J_)

        # TODO: Add logging for inner iterations

        expected = -α*(ΔV[1] + α*ΔV[2])
        # Finish if the expected decrease is super small
        if 0.0 < expected < solver.opts.expected_decrease_tolerance
            # Don't take a step at all, since it's likely to have 
            # numerical issues if the expected decrease is infinitessimal
            α = 0.0
            z = Inf
            copyto!(Z̄, Z)
            J = J_prev
            # TODO: Add log message

            # Increase regularization
            increaseregularization!(solver)
            break 
        elseif expected > 0.0
            z = (J_prev - J) / expected
        else
            z = -1.0
        end

        # Check for acceptance criteria
        if (z_lb ≤ z ≤ z_ub)
            break
        end

        # Check max iterations
        if i == max_iters
            # Don't take a step
            α = 0.0
            copyto!(Z̄, Z)
            J = J_prev

            # TODO: Add log message
            increaseregularization!(solver)
            solver.reg.ρ += solver.opts.bp_reg_fp
            break
        end

        # Decrease line search parameter
        α *= ϕ
    end
    if J > J_prev
        # TODO: Add log messge
        solver.stats.status = COST_INCREASE
        return NaN
    end
    # TODO: Add logging
    return J
end