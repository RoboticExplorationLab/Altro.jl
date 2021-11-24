
function rollout!(solver::iLQRSolver{T,Q,n}, α) where {T,Q,n}
    Z = solver.Z; Z̄ = solver.Z̄
    K = solver.K; d = solver.d;

    Z̄[1].z = [solver.x0; control(Z[1])]

    temp = 0.0
	δx = solver.S[end].q
	δu = solver.S[end].r

    for k = 1:solver.N-1
        δx .= RobotDynamics.state_diff(solver.model, state(Z̄[k]), state(Z[k]))
		δu .= d[k] .* α
		mul!(δu, K[k], δx, 1.0, 1.0)
        ū = control(Z[k]) + δu
        RobotDynamics.set_control!(Z̄[k], ū)

        # Z̄[k].z = [state(Z̄[k]); control(Z[k]) + δu]
        Z̄[k+1].z = [RobotDynamics.discrete_dynamics(Q, solver.model, Z̄[k]);
            control(Z[k+1])]

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

"Simulate the forward the dynamics open-loop"
function rollout!(solver::iLQRSolver)
    # TODO: Take the signature from the solver options
    rollout!(RD.StaticReturn(), solver.model, solver.Z, SVector(solver.x0))
    for k in eachindex(solver.Z)
        solver.Z̄[k].t = solver.Z[k].t
    end
end
