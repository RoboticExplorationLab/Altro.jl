
@testset "iLQR Solver" begin
solver = Altro.iLQRSolver(Problems.Cartpole()...)

@test (@ballocated Altro.initialize!($solver) evals=1 samples=1) == 0
@test (@ballocated TO.state_diff_jacobian!(
    $solver.model, $solver.G, $solver.Z
) evals=1 samples=1) == 0
@test (@ballocated TO.dynamics_expansion!(
    $(solver.opts.dynamics_funsig), $(solver.opts.dynamics_diffmethod), 
    $solver.model, $solver.D, $solver.Z
) evals=1 samples=1) == 0
@test (@ballocated TO.error_expansion!(
    $solver.D, $solver.model, $solver.G
) evals=1 samples=1) == 0
@test (@ballocated TO.error_expansion!(
    $solver.D, $solver.model, $solver.G
) evals=1 samples=1) == 0
@test (@ballocated TO.cost_expansion!(
    $solver.quad_obj, $solver.obj, $solver.Z, init=true, rezero=true
) evals=1 samples=1) == 0
@test (@ballocated TO.error_expansion!(
    $solver.E, $solver.quad_obj, $solver.model, $solver.Z, $solver.G
) evals=1 samples=1) == 0
@test (@ballocated Altro.backwardpass!($solver) evals=1 samples=1) == 0
ΔV = Altro.backwardpass!(solver)
J = cost(solver)
@test (@ballocated Altro.forwardpass!($solver, $ΔV, $J) evals=1 samples=1) == 0

@test (@ballocated Altro.step!($solver, $J) evals=1 samples=1) == 0
end