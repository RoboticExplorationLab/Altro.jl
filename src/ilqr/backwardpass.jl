function chol!(A)
    C, info = LinearAlgebra._chol!(A, LowerTriangular)
    Cholesky(C.data, 'L', info)
end

function foo(C,A,B)
    matmul!(C,A,B,1.0, 2.0)
    C .+= 1
    C
end

function backwardpass!(solver::iLQRSolver)
    # Extract Variables
    _,nu,N = RD.dims(solver)
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
        matmul!(Q[k].x, A', S[k+1].x)
        Q[k].x .+= E[k].x

        matmul!(Q[k].u, B', S[k+1].x)
        Q[k].u .+= E[k].u

        matmul!(Qtmp.xx, S[k+1].xx, A)
        matmul!(Q[k].xx, A', Qtmp.xx)
        Q[k].xx .+= E[k].xx

        matmul!(Qtmp.ux, B', S[k+1].xx)
        matmul!(Q[k].uu, Qtmp.ux, B)
        Q[k].uu .+= E[k].uu

        matmul!(Qtmp.xx, S[k+1].xx, A)
        matmul!(Q[k].ux, B', Qtmp.xx)
        Q[k].ux .+= E[k].ux


        # Regularization
        ρ = solver.reg.ρ
        if solver.opts.bp_reg_type == :state
            Quu_reg .= Q[k].uu
            matmul!(Quu_reg, B', B, ρ, 1.0)
            Qux_reg .= Qux
            matmul!(Qux_reg, A', A, ρ, 1.0)
        elseif solver.opts.bp_reg_type == :control
            Quu_reg .= Q[k].uu
            for i = 1:nu[k]
                Quu_reg[i,i] += ρ
            end
            Qux_reg .= Q[k].ux
        end

        # Solve for gains
        K[k] .= Qux_reg
        d[k] .= Q[k].u
        # LAPACK.potrf!('L',Quu_reg)
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
        # LAPACK.potrs!('L', Quu_reg, K[k])
        # LAPACK.potrs!('L', Quu_reg, d[k])
        # K[k] .*= -1
        # d[k] .*= -1
        solver.gains[k] .*= -1

        # k == N-5 && break

        # Update Cost-to-go
	    # S.x[k]  =  Qx + K[k]'*Quu*d[k] + K[k]'* Qu + Qux'd[k]
        S[k].x .= Q[k].x
        matmul!(Qtmp.u, Q[k].uu, d[k])
        matmul!(S[k].x, K[k]', Qtmp.u, 1.0, 1.0)
        matmul!(S[k].x, K[k]', Q[k].u, 1.0, 1.0)
        matmul!(S[k].x, Q[k].ux', d[k], 1.0, 1.0)

	    # S.xx[k] = Qxx + K[k]'*Quu*K[k] + K[k]'*Qux + Qux'K[k]
        S[k].xx .= Q[k].xx
        matmul!(Qtmp.ux, Q[k].uu, K[k])
        matmul!(S[k].xx, K[k]', Qtmp.ux, 1.0, 1.0)
        matmul!(S[k].xx, K[k]', Q[k].ux, 1.0, 1.0)
        matmul!(S[k].xx, Q[k].ux', K[k], 1.0, 1.0)

        # Symmeterize Cost-to-go Hessian
        transpose!(Qtmp.xx, S[k].xx)
        S[k].xx .+= Qtmp.xx
        S[k].xx ./= 2

        ΔV[1] += dot(d[k], Q[k].u) 
        ΔV[2] += 0.5 * dot(d[k], Q[k].uu, d[k])
        # println("k = $k, ΔV = ", ΔV)

        k -= 1
    end
    decreaseregularization!(solver)
    return ΔV
end