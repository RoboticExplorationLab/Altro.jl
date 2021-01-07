using FiniteDiff
using RobotDynamics
using RobotZoo

RD.diffmethod(::RobotZoo.Cartpole) = RD.ForwardAD()
prob,opts = Problems.Cartpole()
solver = Altro.iLQRSolver(prob, opts)
solve!(solver)
@test solver.cache.fx1 == zeros(size(prob)[1])
@test (@allocated solve!(solver)) == 0

# Test with finite differencing
RD.diffmethod(::RobotZoo.Cartpole) = RD.FiniteDifference()
prob,opts = Problems.Cartpole()
solver = Altro.iLQRSolver(prob, opts)
solve!(solver)
@test solver.cache.fx1 != zeros(size(prob)[1])
@test (@allocated solve!(solver)) == 0