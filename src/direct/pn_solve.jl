function solve!(pn::ProjectedNewtonSolver2)
    evaluate_constraints!(pn)
    constraint_jacobians!(pn)
    cost_gradient!(pn)
    cost_hessian!(pn)
end

function projection_solve!(pn::ProjectedNewtonSolver2)
    ϵ_feas = pn.opts.constraint_tolerance
    viol = norm(pn.d[solver.active], Inf)
    max_projection_iters = pn.opts.n_steps
    count = 0
    while count <= max_projection_iters && viol > ϵ_feas
        viol = _projection_solve!(pn)
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
    ρ_chol = solver.opts.ρ_chol
    ρ_primal = solver.opts.ρ_primal

    # Assume constant, diagonal cost Hessian
    # TODO: change this!
    H = solver.H

    # Update everything
    evaluate_constraints!(pn)
    constraint_jacobians!(pn)
    cost_gradient!(pn)
    cost_hessian!(pn)

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
    solve_tol = 1e-6
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
        pn.Z̄data .+= α * δZ
        
        evaluate_constraints!(pn, pn.Z̄)
        viol = max_violation(pn, nothing)
        if viol < viol0 || count > 10
            break
        else
            count += 1
            α *= ϕ
        end
        # constraint_jacobians!(pn)
        # cost_gradient!(pn)
        # cost_hessian!(pn)
    end
    pn.Zdata .= pn.Z̄data
    return viol
end
