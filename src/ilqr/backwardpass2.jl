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

# function backwardpass_comp!(s1::iLQRSolver, s2::iLQRSolver2)
#     # Extract Variables from solver1
#     K1 = s1.K; d1 = s1.d;
#     S1 = s1.S
# 	Quu_reg1 = s1.Quu_reg
# 	Qux_reg1 = s1.Qux_reg
#     ΔV1 = @SVector zeros(2)

#     # Extract Variables from solver2
#     _,m,N = RD.dims(s1)
#     D2 = s2.D     # dynamics expansion
#     E2 = s2.Eerr  # cost expansion
#     S2 = s2.S     # quadratic cost-to-go
#     Q2 = s2.Q     # action-value expansion
#     K2 = s2.K
#     d2 = s2.d
#     Quu_reg2 = s2.Quu_reg
#     Qux_reg2 = s2.Qux_reg

#     ΔV2 = s2.ΔV
#     Qtmp2 = s2.Qtmp

#     # Terminal cost-to-go
# 	Q1 = s1.E[N]
#     get_data(S1[N].Q) .= get_data(Q1.Q)
#     get_data(S1[N].q) .= get_data(Q1.q)

#     S2[N].xx .= E2[N].xx
#     S2[N].x .= E2[N].x

#     k = N-1
#     ΔV2 .= 0
#     while k > 0
# 		# Get error state expanions
#         let solver = s1
#             fdx,fdu = TO.error_expansion(solver.D[k], solver.model)
#             cost_exp = solver.E[k]
#             Q = solver.Q_tmp

#             # Calculate action-value expansion
#             _calc_Q!(Q, cost_exp, S1[k+1], fdx, fdu, S1[k])

#             solver.Q[k].xx .= Q.xx
#             solver.Q[k].ux .= Q.ux
#             solver.Q[k].uu .= Q.uu
#             solver.Q[k].x .= Q.x
#             solver.Q[k].u .= Q.u
#         end

#         let solver = s2, D = D2, S=S2, Q = Q2, E = E2
#             A = D[k].fx
#             B = D[k].fu
#             Qtmp = solver.Qtmp
#             _calc_Q!(solver.Q[k], E[k], S[k+1], A, B, solver.Qtmp)

#             # # Action-value expansion
#             # mul!(Q[k].x, A', S[k+1].x)
#             # Q[k].x .+= E[k].x

#             # mul!(Q[k].u, B', S[k+1].x)
#             # Q[k].u .+= E[k].u

#             # mul!(Qtmp.xx, S[k+1].xx, A)
#             # mul!(Q[k].xx, A', Qtmp.xx)
#             # Q[k].xx .+= E[k].xx

#             # mul!(Qtmp.ux, B', S[k+1].xx)
#             # mul!(Q[k].uu, Qtmp.ux, B)
#             # Q[k].uu .+= E[k].uu

#             # mul!(Qtmp.xx, S[k+1].xx, A)
#             # mul!(Q[k].ux, B', Qtmp.xx)
#             # Q[k].ux .+= E[k].ux
#         end

#         # Regularization
#         let solver = s1
#             Q = solver.Q_tmp
#             get_data(solver.Quu_reg) .= get_data(Q.R) #+ solver.ρ[1]*I
#             get_data(solver.Quu_reg) .+= solver.ρ[1]*Diagonal(@SVector ones(m))
#             get_data(solver.Qux_reg) .= get_data(Q.H)
#         end

#         let solver = s2, Q = Q2
#             solver.Quu_reg .= solver.Q[k].uu
#             solver.Quu_reg .+= solver.reg.ρ * Diagonal(@SVector ones(m))
#             solver.Qux_reg .= solver.Q[k].ux
#             # ρ = solver.reg.ρ
#             # Quu_reg = solver.Quu_reg
#             # Qux_reg = solver.Qux_reg
#             # if solver.opts.bp_reg_type == :state
#             #     Quu_reg .= Q[k].uu
#             #     mul!(Quu_reg, B', B, ρ, 1.0)
#             #     Qux_reg .= Qux
#             #     mul!(Qux_reg, A', A, ρ, 1.0)
#             # elseif solver.opts.bp_reg_type == :control
#             #     Quu_reg .= Q[k].uu
#             #     for i = 1:m
#             #         Quu_reg[i,i] += ρ
#             #     end
#             #     Qux_reg .= Q[k].ux
#             # end
#         end

#         # Solve for gains
#         # K2[k] .= Qux_reg2
#         # d2[k] .= Q2[k].u
#         # LAPACK.potrf!('L',Quu_reg2)
#         # LAPACK.potrs!('L', Quu_reg2, K2[k])
#         # LAPACK.potrs!('L', Quu_reg2, d2[k])
#         # K2[k] .*= -1
#         # d2[k] .*= -1

#         LAPACK.potrf!('L',Quu_reg2)
#         K2[k] .= Qux_reg2
#         d2[k] .= Q2[k].u
#         LAPACK.potrs!('L', Quu_reg2, K2[k])
#         LAPACK.potrs!('L', Quu_reg2, d2[k])
#         K2[k] .*= -1
#         d2[k] .*= -1

# 		_calc_gains!(K1[k], d1[k], Quu_reg1, Qux_reg1, s1.Q_tmp.r)
#         # Quu_fact = chol!(Quu_reg)::Cholesky
#         # if !isposdef(Quu_fact)  # this is a super cheap check
#         #     # TODO: add log message
#         #     @warn "Backwardpass cholesky failed at time step $k"
#         #     increaseregularization!(solver)
#         #     k = N-1
#         #     ΔV .= 0
#         #     continue
#         # end
#         # Save time by solving for K and d at the same time (1 BLAS call)
#         # ldiv!(Quu_fact, solver.gains[k])
#         # solver.gains[k] .*= -1

#         # k == N-5 && break

#         # Update Cost-to-go
# 	    # S.x[k]  =  Qx + K[k]'*Quu*d[k] + K[k]'* Qu + Qux'd[k]
#         # let solver = s2, S = S2, K = K2, d = d2, Qtmp=s2.Qtmp, Q=Q2
#         #     S[k].x .= Q[k].x
#         #     mul!(Qtmp.u, Q[k].uu, d[k])
#         #     mul!(S[k].x, K[k]', Qtmp.u, 1.0, 1.0)
#         #     mul!(S[k].x, K[k]', Q[k].u, 1.0, 1.0)
#         #     mul!(S[k].x, Q[k].ux', d[k], 1.0, 1.0)

#         #     # S.xx[k] = Qxx + K[k]'*Quu*K[k] + K[k]'*Qux + Qux'K[k]
#         #     S[k].xx .= Q[k].xx
#         #     mul!(Qtmp.ux, Q[k].uu, K[k])
#         #     mul!(S[k].xx, K[k]', Qtmp.ux, 1.0, 1.0)
#         #     mul!(S[k].xx, K[k]', Q[k].ux, 1.0, 1.0)
#         #     mul!(S[k].xx, Q[k].ux', K[k], 1.0, 1.0)

#         #     ΔV2[1] += dot(d[k], Q[k].u) 
#         #     ΔV2[2] += 0.5 * dot(d[k], Q[k].uu, d[k])
#         # end
#         ΔV2 += _calc_ctg!(S2[k], s2.Q[k], K2[k], d2[k], s2.Qtmp.u, s2.Qtmp.ux)
# 		ΔV1 += _calc_ctg!(S1[k], s1.Q_tmp, K1[k], d1[k])
#         # println("k = $k, ΔV = ", ΔV)

#         k -= 1
#     end
#     # decreaseregularization!(s1)
#     # decreaseregularization!(s2)
#     return ΔV1, ΔV2
# end