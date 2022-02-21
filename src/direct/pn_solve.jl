function solve!(pn::ProjectedNewtonSolver2)
    evaluate_constraints!(pn)
    constraint_jacobians!(pn)
    cost_gradient!(pn)
    cost_hessian!(pn)
    update_active_set!(pn)

    projection_solve!(pn)
end

function projection_solve!(pn::ProjectedNewtonSolver2)
    ϵ_feas = pn.opts.constraint_tolerance
    viol = TO.max_violation(pn, nothing)
    max_projection_iters = pn.opts.n_steps
    count = 0
    while count <= max_projection_iters && viol > ϵ_feas
        _projection_solve!(pn)
        viol = max_violation(pn)  # calculate again, updating the active set
        if (pn.opts.multiplier_projection)
            res = multiplier_projection!(pn)
        else
            res = Inf
        end
        count += 1
        record_iteration!(pn, viol, res)
    end
    return viol
end

function record_iteration!(pn::ProjectedNewtonSolver2, viol, res)
    J = TO.cost(pn)
    J_prev = pn.stats.cost[pn.stats.iterations]
    record_iteration!(pn.stats, cost=J, c_max=viol, is_pn=true, dJ=J_prev-J, gradient=res, 
        penalty_max=NaN)
end

function _projection_solve!(pn::ProjectedNewtonSolver2)
    Z = pn.Z
    a = pn.active
    max_refinements = 10  # TODO: make this a solver option
    convergence_rate_threshold = pn.opts.r_threshold

    # Regularization
    ρ_chol = pn.opts.ρ_chol
    ρ_primal = pn.opts.ρ_primal

    # Assume constant, diagonal cost Hessian
    # TODO: change this!
    H = pn.H

    # Update everything
    evaluate_constraints!(pn)
    constraint_jacobians!(pn)
    cost_gradient!(pn)
    cost_hessian!(pn)
    update_active_set!(pn)

    # Get active constraints
    D,d = active_constraints(pn)

    viol0 = norm(d, Inf)
    if ρ_primal > 0.0
        Np = num_primals(pn)
        for i = 1:Np
            H[i,i] += ρ_primal
        end
    end
    if isdiag(H)
        HinvD = Diagonal(H) \ D'
    else
        HinvD = H \ Matrix(D')  # TODO: find a better way to do this
    end

    S = Symmetric(D*HinvD)
    Sreg = cholesky(S + ρ_chol*I, check=false)
    if !issuccess(Sreg)  # TODO: handle this better (increase regularization)
        throw(PosDefException(0))
    end
    viol_prev = viol0
    count = 0
    while count < max_refinements
        viol = _projection_linesearch!(pn, (S,Sreg), HinvD)
        convergence_rate = log10(viol) / log10(viol_prev)
        viol_prev = viol
        count += 1

        if convergence_rate < convergence_rate_threshold || viol < pn.opts.constraint_tolerance
            break
        end
    end
    return viol_prev
end

function _projection_linesearch!(pn::ProjectedNewtonSolver2, S, HinvD)
    solve_tol = 1e-8
    refinement_iters = 25
    α = 1.0
    ϕ = 0.5
    count = 1
    pn.Z̄data .= pn.Zdata
    viol = Inf
    d = pn.d[pn.active]
    viol0 = norm(d, Inf)
    while true
        # Solve Schur compliment
        δλ = reg_solve(S[1], d, S[2], solve_tol, refinement_iters)
        δZ = -HinvD*δλ
        pn.Z̄data .= pn.Zdata .+ α .* δZ
        
        evaluate_constraints!(pn, pn.Z̄)
        viol = TO.max_violation(pn, nothing)
        if viol < viol0
            pn.Zdata .= pn.Z̄data
            break
        elseif count > 10
            break
        else
            count += 1
            α *= ϕ
        end
        # constraint_jacobians!(pn)
        # cost_gradient!(pn)
        # cost_hessian!(pn)
    end
    return viol
end

function multiplier_projection!(pn::ProjectedNewtonSolver2)
    λ = pn.Ydata[pn.active]
    D,d = active_constraints(pn)
    g = pn.g
    res0 = g + D'λ
    A = D*D'
    Areg = A + I*pn.opts.ρ_primal
    b = D*res0
    δλ = -reg_solve(A, b, Areg)
    λ += δλ
    res = g + D'λ  # dual feasibility
    pn.Ydata[pn.active] .= λ
    return norm(res)
end