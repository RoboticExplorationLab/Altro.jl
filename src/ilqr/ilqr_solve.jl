function initialize!(solver::iLQRSolver)
	reset!(solver)
    set_verbosity!(solver)
    clear_cache!(solver)

    solver.ρ[1] = solver.opts.bp_reg_initial
    solver.dρ[1] = 0.0

    # Initial rollout
    rollout!(solver)
    TO.cost!(solver.obj, solver.Z)
end

function mysolve(solver)
    Altro.initialize!(solver)
end

# Generic solve methods
"iLQR solve method (non-allocating)"
function solve!(solver::iLQRSolver)
    initialize!(solver)

    Z = solver.Z; Z̄ = solver.Z̄;

    n,m,N = dims(solver)
    J = Inf
    _J = TO.get_J(solver.obj)
    J_prev = sum(_J)

    grad_only = false
    for i = 1:solver.opts.iterations
        J = step!(solver, J_prev, grad_only)

        # check for a change in solver status
        status(solver) > SOLVE_SUCCEEDED && break

        copy_trajectories!(solver)

        dJ = abs(J - J_prev)
        J_prev = copy(J)
        gradient_todorov!(solver)

        # Record iteration and evaluate convergence
        record_iteration!(solver, J, dJ)
        exit = evaluate_convergence(solver)

        # check if the cost function is quadratic
        if !grad_only && solver.opts.reuse_jacobians && TO.is_quadratic(solver.E)
            @logmsg InnerLoop "Gradient-only LQR"
            grad_only = true  # skip updating feedback gain matrix (feedforward only)
        end

        # check for cost blow up

        # Print iteration
        if is_verbose(solver) 
            print_level(InnerLoop, global_logger())
        end
        exit && break

        if J > solver.opts.max_cost_value
            # @warn "Cost exceeded maximum cost"
            solver.stats.status = MAXIMUM_COST
            break
        end
    end
    terminate!(solver)
    return solver
end

# function step!(solver::iLQRSolver, J)
#     TO.state_diff_jacobian!(solver.G, solver.model, solver.Z)
# 	TO.dynamics_expansion!(integration(solver), solver.D, solver.model, solver.Z)
# 	TO.error_expansion!(solver.D, solver.model, solver.G)
#     TO.cost_expansion!(solver.quad_obj, solver.obj, solver.Z, true, true)
#     TO.error_expansion!(solver.E, solver.quad_obj, solver.model, solver.Z, solver.G)
# 	if solver.opts.static_bp
#     	ΔV = static_backwardpass!(solver)
# 	else
# 		ΔV = backwardpass!(solver)
#     end
#     forwardpass!(solver, ΔV, J)
# end

function step!(solver::iLQRSolver{<:Any,<:Any,L}, J, grad_only::Bool=false) where L
    to = solver.stats.to
    init = !solver.opts.reuse_jacobians  # force recalculation if not reusing
    @timeit_debug to "diff jac"     TO.state_diff_jacobian!(solver.model, solver.G, solver.Z)
    if !solver.opts.reuse_jacobians || !(L <: RD.LinearModel) || !grad_only
        @timeit_debug to "dynamics jac" TO.dynamics_expansion!(
            RD.StaticReturn(), RD.ForwardAD(), solver.model, solver.D, solver.Z
        )
    end
	@timeit_debug to "err jac"      TO.error_expansion!(solver.D, solver.model, 
                                                        solver.G)
    @timeit_debug to "cost exp"     TO.cost_expansion!(solver.quad_obj, solver.obj, 
                                                       solver.Z, 
                                                       init=init, rezero=true)
    @timeit_debug to "cost err"     TO.error_expansion!(solver.E, solver.quad_obj, 
                                                        solver.model, solver.Z, 
                                                        solver.G)
	@timeit_debug to "backward pass" if solver.opts.static_bp
    	ΔV = static_backwardpass!(solver, grad_only)
	else
		ΔV = backwardpass!(solver)
    end
    @timeit_debug to "forward pass" forwardpass!(solver, ΔV, J)
end

"""
$(SIGNATURES)
Simulate the system forward using the optimal feedback gains from the backward pass,
projecting the system on the dynamically feasible subspace. Performs a line search to ensure
adequate progress on the nonlinear problem.
"""
function forwardpass!(solver::iLQRSolver, ΔV, J_prev)
    Z = solver.Z; Z̄ = solver.Z̄
    obj = solver.obj

    _J = TO.get_J(obj)
    J::Float64 = Inf
    α = 1.0
    iter = 0
    z = -1.0
    expected = 0.0
    flag = true

    solver.stats.ls_failed = false

    while (z ≤ solver.opts.line_search_lower_bound || z > solver.opts.line_search_upper_bound) && J >= J_prev

        # Check that maximum number of line search decrements has not occured
        if iter > solver.opts.iterations_linesearch
            for k in eachindex(Z)
                Z̄[k].z = Z[k].z
            end
            TO.cost!(obj, Z̄)
            J = sum(_J)

            z = 0
            α = 0.0
            expected = 0.0
            @logmsg InnerLoop "Max Line Search Iterations."
            solver.stats.ls_failed = true

            # solver.stats.status = LINESEARCH_FAIL
            regularization_update!(solver, :increase)
            solver.ρ[1] += solver.opts.bp_reg_fp
            break
        end


        # Otherwise, rollout a new trajectory for current alpha
        flag = rollout!(solver, α)

        # Check if rollout completed
        if ~flag
            # Reduce step size if rollout returns non-finite values (NaN or Inf)
            # @logmsg InnerIters "Non-finite values in rollout"
            iter += 1
            α /= 2.0
            continue
        end

        # Calcuate cost
        TO.cost!(obj, Z̄)
        J = sum(_J)

        expected::Float64 = -α*(ΔV[1] + α*ΔV[2])
        if expected > 0.0
            z::Float64  = (J_prev - J)/expected
        else
            z = -1.0
        end

        iter += 1
        α /= 2.0
    end

    if J > J_prev
        # error("Error: Cost increased during Forward Pass")
        solver.stats.status = COST_INCREASE
        return NaN
    end

    @logmsg InnerLoop :expected value=expected
    @logmsg InnerLoop :z value=z
    @logmsg InnerLoop :α value=2*α
    @logmsg InnerLoop :ρ value=solver.ρ[1]
    @logmsg InnerLoop :dJ_zero value=solver.opts.dJ_counter_limit

    return J

end

function copy_trajectories!(solver::iLQRSolver)
    for k = 1:solver.N
        solver.Z[k].z = solver.Z̄[k].z
    end
end

"""
Stash iteration statistics
"""
function record_iteration!(solver::iLQRSolver, J, dJ)
    gradient = mean(solver.grad)
    record_iteration!(solver.stats, cost=J, dJ=dJ, gradient=gradient)
    i = solver.stats.iterations::Int
    
    if dJ ≈ 0
        solver.stats.dJ_zero_counter += 1
    else
        solver.stats.dJ_zero_counter = 0
    end

    @logmsg InnerLoop :iter value=i
    @logmsg InnerLoop :cost value=J
    @logmsg InnerLoop :dJ   value=dJ
    @logmsg InnerLoop :grad value=gradient
    # @logmsg InnerLoop :zero_count value=solver.stats[:dJ_zero_counter][end]
    return nothing
end

"""
$(SIGNATURES)
    Calculate the problem gradient using heuristic from iLQG (Todorov) solver
"""
function gradient_todorov!(solver::iLQRSolver)
	tmp = solver.S[end].r
    for k in eachindex(solver.d)
		tmp .= abs.(solver.d[k])
		u = abs.(control(solver.Z[k])) .+ 1
		tmp ./= u
		solver.grad[k] = maximum(tmp)
    end
end


"""
$(SIGNATURES)
Check convergence conditions for iLQR
"""
function evaluate_convergence(solver::iLQRSolver)
    # Get current iterations
    i = solver.stats.iterations
    grad = solver.stats.gradient[i]
    dJ = solver.stats.dJ[i]

    # Check for cost convergence
    # must satisfy both 
    if (0.0 <= dJ < solver.opts.cost_tolerance) && (grad < solver.opts.gradient_tolerance) && !solver.stats.ls_failed
        @logmsg InnerLoop "Cost criteria satisfied."
        solver.stats.status = SOLVE_SUCCEEDED
        return true
    end

    # Check total iterations
    if i >= solver.opts.iterations
        @logmsg InnerLoop "Hit max iterations. Terminating."
        solver.stats.status = MAX_ITERATIONS
        return true
    end

    # Outer loop update if forward pass is repeatedly unsuccessful
    if solver.stats.dJ_zero_counter > solver.opts.dJ_counter_limit
        @logmsg InnerLoop "dJ Counter hit max. Terminating."
        solver.stats.status = NO_PROGRESS
        return true
    end

    return false
end

"""
$(SIGNATURES)
Update the regularzation for the iLQR backward pass
"""
function regularization_update!(solver::iLQRSolver,status::Symbol=:increase)
    # println("reg $(status)")
    if status == :increase # increase regularization
        # @logmsg InnerLoop "Regularization Increased"
        solver.dρ[1] = max(solver.dρ[1]*solver.opts.bp_reg_increase_factor, solver.opts.bp_reg_increase_factor)
        solver.ρ[1] = max(solver.ρ[1]*solver.dρ[1], solver.opts.bp_reg_min)
        # if solver.ρ[1] > solver.opts.bp_reg_max
        #     @warn "Max regularization exceeded"
        # end
    elseif status == :decrease # decrease regularization
        # TODO: Avoid divides by storing the decrease factor (divides are 10x slower)
        solver.dρ[1] = min(solver.dρ[1]/solver.opts.bp_reg_increase_factor, 1.0/solver.opts.bp_reg_increase_factor)
        # solver.ρ[1] = solver.ρ[1]*solver.dρ[1]*(solver.ρ[1]*solver.dρ[1]>solver.opts.bp_reg_min)
        solver.ρ[1] = max(solver.opts.bp_reg_min, solver.ρ[1]*solver.dρ[1])
    end
end
