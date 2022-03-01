import QDLDL

function solve!(pn::ProjectedNewtonSolver2)
    copyto!(pn.Z̄data, pn.Zdata)
    evaluate_constraints!(pn)
    constraint_jacobians!(pn)
    # cost_gradient!(pn)
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
        # println("PN Iter = ", count + 1)
        # println("  v = ", viol)
        # _projection_solve!(pn)
        _qdldl_solve!(pn)
        viol = max_violation(pn)  # calculate again, updating the active set
        if (pn.opts.multiplier_projection)
            # res = multiplier_projection!(pn)
            res = Inf
        else
            res = Inf
        end
        count += 1
        # record_iteration!(pn, viol, res)
    end
    return viol
end

function record_iteration!(pn::ProjectedNewtonSolver2, viol, res)
    J = TO.cost(pn)
    J_prev = pn.stats.cost[pn.stats.iterations]
    record_iteration!(pn.stats, cost=J, c_max=viol, is_pn=true, dJ=J_prev-J, gradient=res, 
        penalty_max=NaN)
end

function update_b!(pn::ProjectedNewtonSolver2)
    Np = num_primals(pn)
    Nd = num_duals(pn)
    Na = sum(pn.active) - Np
    b = pn.b
    b .= 0
    resize!(pn.b, Na + Np)
    nzcount = 1
    for i = 1:Nd
        if pn.active[Np + i]
            b[Np + nzcount] = -pn.d[i]
            nzcount += 1
        end
    end
    return b
end

function _qdldl_solve!(pn::ProjectedNewtonSolver2)
    Np = num_primals(pn)
    Nd = num_duals(pn)
    Na = sum(pn.active) - Np

    max_refinements = 10
    convergence_rate_threshold = pn.opts.r_threshold

    # Get A,b
    nnz0 = nnz(pn.Atop) + Nd
    resize!(pn.colptr, Np + Nd + 1)
    resize!(pn.rowval, nnz0)
    resize!(pn.nzval, nnz0)
    nnz_new = triukkt!(pn.Atop, pn.active, pn.colptr, pn.rowval, pn.nzval, reg=1e-8)
    colptr = resize!(pn.colptr, Np + Na + 1)
    rowval = resize!(pn.rowval, nnz_new) 
    nzval = resize!(pn.nzval, nnz_new) 
    A = SparseMatrixCSC(Np + Na, Np + Na, colptr, rowval, nzval)

    # Copy the active constraints to the b vector and resize
    b = update_b!(pn)
    resize!(pn.dY, Np + Na)

    # Factorize the matrix
    resize!(pn.qdldl, Np + Na)
    Cqdldl.eliminationtree!(pn.qdldl, A)
    Cqdldl.factor!(pn.qdldl)
    F = Cqdldl.QDLDLFactorization(pn.qdldl)
    
    # Line search
    viol_prev = max_violation(pn, nothing)
    count = 1
    while count < max_refinements
        viol = _qdldl_linesearch(pn, F, b)
        convergence_rate = log10(viol) / log10(viol_prev)
        # println("  r iter = $(count + 1), v = $viol, r = $convergence_rate")
        viol_prev = viol
        count += 1
        if viol < pn.opts.constraint_tolerance
            break
        elseif convergence_rate < convergence_rate_threshold
            break
        end
    end
    return
end

function _qdldl_linesearch(pn, F, b)
    Np = num_primals(pn)
    Na = length(b) - Np
    ia = Np .+ (1:Na)
    viol0 = max_violation(pn, nothing)
    dY = pn.dY
    # update_b!(pn)
    p = view(dY, 1:Np)
    α = 1.0
    for i = 1:10 
        dY .= b
        ldiv!(F, dY)
        # QDLDL.solve!(F, dY)
        pn.Z̄data .= pn.Zdata .+ α .* p

        evaluate_constraints!(pn, pn.Z̄)
        update_b!(pn)
        v = max_violation(pn, nothing)   # don't update active set
        if v < viol0
            copyto!(pn.Zdata, pn.Z̄data)
            return v
        else
            # b[ia] .= pn.d[pn.active]
            α /= 2
        end
    end
    return NaN
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
        # println("Refinement $count")
        viol = _projection_linesearch!(pn, (S,Sreg), HinvD)
        convergence_rate = log10(viol) / log10(viol_prev)
        # println("  r = $convergence_rate")
        # println("  v = $viol")
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
        d .= pn.d[pn.active]

        if viol < viol0
            # println("  Finished with α = $α")
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
