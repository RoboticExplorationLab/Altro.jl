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