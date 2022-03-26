function alilqr_allocs(solver)
    cost_tol = solver.opts.cost_tolerance
    grad_tol = solver.opts.gradient_tolerance
    al_iter = 1

    allocs = @allocated Altro.reset!(solver)
    allocs += @allocated Altro.set_tolerances!(solver, al_iter, cost_tol, grad_tol)

    cons = Altro.get_constraints(solver)
    Altro.settraj!(cons, solver.ilqr.Z̄)
    let solver = solver.ilqr
        allocs += @allocated J_prev = TO.cost(solver, solver.Z̄)
        allocs += @allocated Altro.errstate_jacobians!(solver.model, solver.G, solver.Z̄)
        allocs += @allocated Altro.dynamics_expansion!(solver, solver.Z̄)
        allocs += @allocated Altro.cost_expansion!(solver.obj, solver.Efull, solver.Z̄)
        allocs += @allocated Altro.error_expansion!(solver.model, solver.Eerr, solver.Efull, solver.G, solver.Z̄)
        allocs += @allocated Altro.backwardpass!(solver)
        allocs += @allocated Jnew = Altro.forwardpass!(solver, J_prev)
        allocs += @allocated copyto!(solver.Z, solver.Z̄)
        dJ = J_prev - Jnew
        allocs += @allocated grad = Altro.gradient!(solver)
        Altro.record_iteration!(solver, Jnew, dJ, grad) 
        allocs += @allocated Altro.evaluate_convergence(solver)
    end
    Z̄ = solver.ilqr.Z̄
    conset = get_constraints(solver)
    allocs += @allocated J = TO.cost(solver, Z̄)
    allocs += @allocated c_max = Altro.max_violation(conset)
    allocs += @allocated μ_max = Altro.max_penalty(conset)
    Altro.record_iteration!(solver, J, c_max, μ_max)

    allocs += @allocated Altro.dualupdate!(conset)
    allocs += @allocated Altro.penaltyupdate!(conset)
    allocs += @allocated Altro.reset!(solver.ilqr)
    allocs
end


@testset "AliLQR Basic Tests" begin
prob, opts = Problems.Quadrotor()
al = Altro.ALSolver(prob, copy(opts), use_static=Val(true))
conset = get_constraints(al)
ilqr = Altro.get_ilqr(al)
@test conset[1].E === ilqr.Efull
@test conset[1].Z[1] === ilqr.Z
@test conset[1].opts.solveropts === ilqr.opts === al.opts
@test conset[1].cost === get_objective(ilqr).alcost

@testset "ALiLQR Allocation test" begin
    solver = Altro.ALSolver(Problems.Quadrotor()...)
    solver.opts.verbose = 0
    !Sys.iswindows() && @test alilqr_allocs(solver) == 0
end
end

solver = Altro.ALSolver(Problems.DubinsCar(:three_obstacles)...)
b = benchmark_solve!(solver)
b.allocs
solver = ALTROSolver(Problems.DubinsCar(:three_obstacles)..., projected_newton=false)
b = benchmark_solve!(solver, samples=1, evals=1)
b.allocs
alilqr_allocs(solver)
