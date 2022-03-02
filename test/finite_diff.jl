@testset "Finite Diff Dynamics" begin
## Test with ForwardDiff
RD.diffmethod(::RobotZoo.Cartpole) = RD.ForwardAD()
prob,opts = Problems.Cartpole()
Z0 = copy(prob.Z)
solver = Altro.iLQRSolverOld(prob, opts, show_summary=false)
solve!(solver)
@test solver.cache.fx1 == zeros(size(prob)[1])
@test (@allocated solve!(solver)) == 0
J = cost(solver)

solver = ALTROSolver(prob, opts, show_summary=false)
initial_trajectory!(solver, Z0)
solve!(solver)
initial_trajectory!(solver, Z0)
allocs = @allocated solve!(solver)

## Test with finite differencing
RD.diffmethod(::RobotZoo.Cartpole) = RD.FiniteDifference()
prob,opts = Problems.Cartpole()
solver = Altro.iLQRSolverOld(prob, opts, show_summary=false)
solve!(solver)
@test solver.cache.fx1 != zeros(size(prob)[1])
@test (@allocated solve!(solver)) == 0

# make sure finite diff doesn't add any allocs for the entire solve
solver = ALTROSolver(prob, opts)
initial_trajectory!(solver, Z0)
solve!(solver)
initial_trajectory!(solver, Z0)
@test (@allocated solve!(solver)) == allocs  
end