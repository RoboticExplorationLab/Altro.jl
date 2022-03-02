
function rollout!(solver::iLQRSolver{T,Q,n}, α) where {T,Q,n}
    Z = solver.Z; Z̄ = solver.Z̄
    K = solver.K; d = solver.d;

    Z̄[1].z = [solver.x0; control(Z[1])]

    temp = 0.0
    xdot = solver.xdot
	δx = solver.S[end].q
	δu = solver.S[end].r

    for k = 1:solver.N-1
        δx .= RobotDynamics.state_diff(solver.model, state(Z̄[k]), state(Z[k]))
		δu .= d[k] .* α
		mul!(get_data(δu), get_data(K[k]), get_data(δx), 1.0, 1.0)
        ū = control(Z[k]) + δu
        RobotDynamics.setcontrol!(Z̄[k], ū)

        Z̄[k].z = [state(Z̄[k]); control(Z[k]) + δu]
        if solver.opts.dynamics_funsig == StaticReturn()
            Z̄[k+1].z = [RD.discrete_dynamics(solver.model, Z̄[k]);
                control(Z[k+1])]
        else
            RD.discrete_dynamics!(solver.model, xdot, Z̄[k])
            RD.setstate!(Z̄[k+1], xdot)
        end

        max_x = norm(state(Z̄[k+1]),Inf)
        if max_x > solver.opts.max_state_value || isnan(max_x)
            solver.stats.status = STATE_LIMIT
            return false
        end
        max_u = norm(control(Z̄[k]),Inf)
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
    # rollout!(RD.StaticReturn(), solver.model, solver.Z, SVector(solver.x0))
    Z = solver.Z
    xdot = solver.xdot
    RD.setstate!(Z[1], solver.x0)
    for k in 1:solver.N-1 
        if solver.opts.dynamics_funsig == StaticReturn()
            RD.propagate_dynamics!(StaticReturn(), solver.model, Z[k+1], Z[k])
            # Z[k+1].z = [RD.discrete_dynamics(solver.model, Z[k]);
            #     control(Z[k+1])]
        else
            RD.discrete_dynamics!(solver.model, xdot, Z[k])
            RD.setstate!(Z[k+1], xdot)
        end
    end
end
