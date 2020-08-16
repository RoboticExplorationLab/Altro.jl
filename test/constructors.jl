prob, opts = Problems.DoubleIntegrator()

solver = ALTROSolver(prob)
solver = ALTROSolver(prob, opts)
@test solver.opts === solver.solver_al.opts === solver.solver_al.solver_uncon.opts

# Pass in option arguments
solver = ALTROSolver(prob, opts, verbose=2, cost_tolerance=1)
@test solver.opts.verbose == 2
@test solver.opts.cost_tolerance == 1
@test solver.stats.parent == Altro.solvername(solver)


# Test other solvers
stats = solver.stats
al = Altro.AugmentedLagrangianSolver(prob, opts)
@test al.opts === opts
@test al.stats.parent == Altro.solvername(al)
al = Altro.AugmentedLagrangianSolver(prob, opts, stats)
@test al.stats === solver.stats
@test al.stats.parent == Altro.solvername(solver)
@test al.solver_uncon.stats.parent == Altro.solvername(solver)

# Try passing in a bad option
ilqr = Altro.iLQRSolver(prob, opts, something_wrong=false)
@test ilqr.opts === solver.opts

# Solve an unconstrained problem
ilqr = Altro.iLQRSolver(Problems.Cartpole(constrained=false)..., verbose=2)
b0 = benchmark_solve!(ilqr)
solver = ALTROSolver(Problems.Cartpole(constrained=false)..., verbose=2, projected_newton=false)
b1 = benchmark_solve!(solver)
@test iterations(solver) == iterations(ilqr)
@test cost(solver) ≈ cost(ilqr)
@test max_violation(solver) == 0
@test b0.allocs == 0
@test b1.allocs == 0
t0 = minimum(b0).time 
t1 = minimum(b1).time
@test abs(t0-t1)/t1 < 0.1
