const TO = TrajectoryOptimization

altro = ALTROSolver(Problems.Cartpole()...)
solver = altro.solver_al.solver_uncon
Altro.initialize!(solver)

Z = solver.Z
TO.state_diff_jacobian!(solver.G, solver.model, Z)
TO.dynamics_expansion!(TO.integration(solver), solver.D, solver.model, solver.Z)
TO.error_expansion!(solver.D, solver.model, solver.G)
TO.cost_expansion!(solver.quad_obj, solver.obj, solver.Z, true, true)
TO.error_expansion!(solver.E, solver.quad_obj, solver.model, Z, solver.G)

ΔV1 = Altro.static_backwardpass!(solver)
ΔV2 = Altro.backwardpass!(solver)
@test ΔV1 ≈ ΔV2

# Make sure the backward pass doesn't modify it's state
ΔV = Altro.static_backwardpass!(solver)
ΔV2 = Altro.backwardpass!(solver)
@test ΔV ≈ ΔV1 ≈ ΔV2

# Test allocations
@test (@ballocated Altro.static_backwardpass!($solver) samples=2 evals=2) == 0
@test (@ballocated Altro.backwardpass!($solver) samples=2 evals=2) == 0