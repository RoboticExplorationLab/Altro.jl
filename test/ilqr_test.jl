function ilqrallocs(solver)
    allocs = @allocated J_prev = TO.cost(solver, solver.Z̄)

    # Calculate expansions
    # TODO: do this in parallel
    allocs += @allocated Altro.errstate_jacobians!(solver.model, solver.G, solver.Z̄)
    allocs += @allocated Altro.dynamics_expansion!(solver, solver.Z̄)
    allocs += @allocated Altro.cost_expansion!(solver.obj, solver.Efull, solver.Z̄)
    allocs += @allocated Altro.error_expansion!(solver.model, solver.Eerr, solver.Efull, solver.G, solver.Z̄)

    # Get next iterate
    allocs += @allocated Altro.backwardpass!(solver)
    allocs += @allocated Jnew = Altro.forwardpass!(solver, J_prev)

    # Accept the step and update the current trajectory
    # This is kept out of the forward pass function to make it easier to 
    # benchmark the forward pass
    allocs += @allocated copyto!(solver.Z, solver.Z̄)

    # Calculate the gradient of the new trajectory
    dJ = J_prev - Jnew
    grad = Altro.gradient!(solver)

    # Record the iteration
    Altro.record_iteration!(solver, Jnew, dJ, grad)

    # Check convergence
    exit = Altro.evaluate_convergence(solver)
    return allocs
end

if TEST_ALLOCS
    @testset "iLQR Solver Allocations" begin
        solver = Altro.iLQRSolver(Problems.Cartpole()..., use_static=Val(true))
        ilqrallocs(solver)
        @test ilqrallocs(solver) == 0
        solver = Altro.iLQRSolver(Problems.Cartpole()..., use_static=Val(false))
        ilqrallocs(solver)
        @test ilqrallocs(solver) == 0
    end
end