
prob, opts = Problems.DoubleIntegrator()
solver = iLQRSolver(prob, verbose=true, cost_tolerance=1e-1)
@test solver.opts.verbose == true
@test solver.opts.cost_tolerance == 1e-1

solver = AugmentedLagrangianSolver(prob, verbose=true, cost_tolerance=1e-1, square_root=true)
@test solver.opts.verbose == true
@test solver.solver_uncon.opts.verbose == true 
@test solver.opts.cost_tolerance == 1e-1
@test solver.solver_uncon.opts.cost_tolerance == 1e-1
@test solver.solver_uncon.opts.square_root == true
solver = AugmentedLagrangianSolver(prob, verbose=1, cost_tolerance=1e-1, square_root=true)
@test solver.opts.verbose == true
@test solver.solver_uncon.opts.verbose == false 
Altro.set_options!(solver, verbose=0)
@test solver.opts.verbose == false 
@test solver.solver_uncon.opts.verbose == false 

# Test verbosity levels
solver = ALTROSolver(prob, verbose=0)
@test solver.opts.verbose == 0
@test solver.solver_al.opts.verbose == false
@test solver.solver_al.solver_uncon.opts.verbose == false
solver = ALTROSolver(prob, verbose=1)
@test solver.opts.verbose == 1
@test solver.solver_al.opts.verbose == true 
@test solver.solver_al.solver_uncon.opts.verbose == false
Altro.set_options!(solver, verbose=2)
@test solver.opts.verbose == 2
@test solver.solver_al.opts.verbose == true 
@test solver.solver_al.solver_uncon.opts.verbose == true 
Altro.set_options!(solver, verbose=false)
@test solver.opts.verbose == 0
@test solver.solver_al.opts.verbose == false
@test solver.solver_al.solver_uncon.opts.verbose == false
Altro.set_options!(solver, verbose=true)
@test solver.opts.verbose == 2
@test solver.solver_al.opts.verbose == true 
@test solver.solver_al.solver_uncon.opts.verbose == true 

# Test ALTRO solver options
solver = ALTROSolver(prob, cost_tolerance=1e-1, square_root=true, 
    iterations=210, iterations_outer=60, ρ_chol=1e-1, constraint_tolerance=1e-2)
@test solver.opts.cost_tolerance == 1e-1
@test solver.solver_al.opts.cost_tolerance == 1e-1
@test solver.solver_al.solver_uncon.opts.cost_tolerance == 1e-1
@test solver.solver_al.solver_uncon.opts.iterations == 210 
@test solver.solver_al.solver_uncon.opts.square_root == true 
@test solver.solver_al.opts.iterations_outer == 60
@test solver.solver_pn.opts.ρ_chol == 1e-1
@test solver.solver_pn.opts.constraint_tolerance == 1e-2
@test solver.solver_al.opts.constraint_tolerance == 1e-2
@test solver.opts.constraint_tolerance == 1e-2

# Test existence
solver = ALTROSolver(prob)
@test Altro.has_option(solver, :square_root) == true
@test Altro.has_option(solver, :constrained) == true
@test Altro.has_option(solver, :solve_type) == true
@test Altro.has_option(solver, :something_odd) == false
solver = iLQRSolver(prob)
@test Altro.has_option(solver, :constraint_tolerance) == false
@test Altro.has_option(solver, :bp_reg)

# Test getters
solver = iLQRSolver(prob)
@test Altro.get_option(solver, :bp_reg) == false
@test Altro.get_option(solver, :square_root) == false
@test_throws ErrorException Altro.get_option(solver, :constraint_tolerance)
solver = ALTROSolver(prob)
@test Altro.get_option(solver, :square_root) == false
@test Altro.get_option(solver, :solve_type) == :feasible
@test Altro.get_option(solver, :constrained) == true
@test_throws ErrorException Altro.get_option(solver, :something_odd)


# Get all options
@test Altro.get_options(solver.solver_pn) isa Dict{Symbol,Any}
@test length(Altro.get_options(solver)) > length(Altro.get_options(solver, false))
Altro.set_options!(solver, verbose=0)
Altro.set_options!(solver, constraint_tolerance=1e-4)
@test_nowarn Altro.get_options(solver.solver_al, true, false)

Altro.set_options!(solver, verbose=1)
@test_logs (:warn, r"Cannot combine") match_mode=:all Altro.get_options(solver.solver_al, true, false)
