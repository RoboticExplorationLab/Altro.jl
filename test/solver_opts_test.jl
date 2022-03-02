
prob, opts = Problems.DoubleIntegrator()
solver = Altro.iLQRSolver(prob, verbose=true, cost_tolerance=1e-1)
@test solver.opts.verbose == true
@test solver.opts.cost_tolerance == 1e-1

solver = Altro.ALSolver(prob, verbose=true, cost_tolerance=1e-1, square_root=true)
@test solver.opts.verbose == true
@test solver.ilqr.opts.verbose == true 
@test solver.opts.cost_tolerance == 1e-1
@test solver.ilqr.opts.cost_tolerance == 1e-1
@test solver.ilqr.opts.square_root == true
solver = Altro.ALSolver(prob, verbose=1, cost_tolerance=1e-1, square_root=true)
@test solver.opts.verbose == true
@test solver.ilqr.opts.verbose == 1 
Altro.set_options!(solver, verbose=0)
@test solver.opts.verbose == false 
@test solver.ilqr.opts.verbose == false 

# Test ALTRO solver options
solver = ALTROSolver2(prob, cost_tolerance=1e-1, square_root=true, 
    iterations=210, iterations_outer=60, ρ_chol=1e-1, constraint_tolerance=1e-2)
@test solver.opts.cost_tolerance == 1e-1
@test solver.solver_al.opts.cost_tolerance == 1e-1
@test solver.solver_al.ilqr.opts.cost_tolerance == 1e-1
@test solver.solver_al.ilqr.opts.iterations == 210 
@test solver.solver_al.ilqr.opts.square_root == true 
@test solver.solver_al.opts.iterations_outer == 60
@test solver.solver_pn.opts.ρ_chol == 1e-1
@test solver.solver_pn.opts.constraint_tolerance == 1e-2
@test solver.solver_al.opts.constraint_tolerance == 1e-2
@test solver.opts.constraint_tolerance == 1e-2