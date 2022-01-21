function chol!(A)
    C, info = LinearAlgebra._chol!(A, LowerTriangular)
    Cholesky(C.data, 'L', info)
end

function backwardpass!(solver::iLQRSolver2)
    # Extract Variables
    _,m,N = RD.dims(solver)
    D = solver.D     # dynamics expansion
    E = solver.Eerr  # cost expansion
    S = solver.S     # quadratic cost-to-go
    Q = solver.Q     # action-value expansion
    K = solver.K
    d = solver.d
    Quu_reg = solver.Quu_reg
    Qux_reg = solver.Qux_reg
    ΔV = solver.ΔV
    Qtmp = solver.Qtmp

    # Terminal cost-to-go
    S[N].xx .= E[N].xx
    S[N].x .= E[N].x

    k = N-1
    ΔV .= 0
    while k > 0
        A = D[k].fx
        B = D[k].fu

        # Action-value expansion
        mul!(Q[k].x, A', S[k+1].x)
        Q[k].x .+= E[k].x

        mul!(Q[k].u, B', S[k+1].x)
        Q[k].u .+= E[k].u

        mul!(Qtmp.xx, S[k+1].xx, A)
        mul!(Q[k].xx, A', Qtmp.xx)
        Q[k].xx .+= E[k].xx

        mul!(Qtmp.ux, B', S[k+1].xx)
        mul!(Q[k].uu, Qtmp.ux, B)
        Q[k].uu .+= E[k].uu

        mul!(Qtmp.xx, S[k+1].xx, A)
        mul!(Q[k].ux, B', Qtmp.xx)
        Q[k].ux .+= E[k].ux

        # Regularization
        ρ = solver.reg.ρ
        if solver.opts.bp_reg_type == :state
            Quu_reg .= Q[k].uu
            mul!(Quu_reg, B', B, ρ, 1.0)
            Qux_reg .= Qux
            mul!(Qux_reg, A', A, ρ, 1.0)
        elseif solver.opts.bp_reg_type == :control
            Quu_reg .= Q[k].uu
            for i = 1:m
                Quu_reg[i,i] += ρ
            end
            Qux_reg .= Q[k].ux
        end

        # Solve for gains
        K[k] .= Qux_reg
        d[k] .= Q[k].u
        Quu_fact = chol!(Quu_reg)::Cholesky
        if !isposdef(Quu_fact)  # this is a super cheap check
            # TODO: add log message
            @warn "Backwardpass cholesky failed at time step $k"
            increaseregularization!(solver)
            k = N-1
            ΔV .= 0
            continue
        end
        # Save time by solving for K and d at the same time (1 BLAS call)
        ldiv!(Quu_fact, solver.gains[k])
        solver.gains[k] .*= -1

        # Update Cost-to-go
	    # S.x[k]  =  Qx + K[k]'*Quu*d[k] + K[k]'* Qu + Qux'd[k]
        S[k].x .= Q[k].x
        mul!(Qtmp.u, Q[k].uu, d[k])
        mul!(S[k].x, K[k]', Qtmp.u, 1.0, 1.0)
        mul!(S[k].x, K[k]', Q[k].u, 1.0, 1.0)
        mul!(S[k].x, Q[k].ux', d[k], 1.0, 1.0)

	    # S.xx[k] = Qxx + K[k]'*Quu*K[k] + K[k]'*Qux + Qux'K[k]
        S[k].xx .= Q[k].xx
        mul!(Qtmp.ux, Q[k].uu, K[k])
        mul!(S[k].xx, K[k]', Qtmp.ux, 1.0, 1.0)
        mul!(S[k].xx, K[k]', Q[k].ux, 1.0, 1.0)
        mul!(S[k].xx, Q[k].ux', K[k], 1.0, 1.0)

        ΔV[1] += dot(d[k], Q[k].u) 
        ΔV[2] += 0.5 * dot(d[k], Q[k].uu, d[k])

        k -= 1
    end
    decreaseregularization!(solver)
    return ΔV
end