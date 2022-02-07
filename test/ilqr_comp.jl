using Altro
using TrajectoryOptimization
using RobotDynamics
using BenchmarkTools
using Test
using LinearAlgebra
using SolverLogging
using Logging
const RD = RobotDynamics
const TO = TrajectoryOptimization
# ENV["JULIA_DEBUG"] = SolverLogging

# prob,opts = Problems.Pendulum()
prob,opts = Problems.Quadrotor()
s1 = Altro.iLQRSolver(prob, opts)
s2 = Altro.iLQRSolver2(copy(prob), copy(opts))
# s1.opts.verbose = 2
s2.opts.verbose = 4
s1.opts.save_S = true
RD.dims(s2) == RD.dims(s1)

Altro.initialize!(s1)
Altro.initialize!(s2)
@test SolverLogging.getlevel(s2.logger) == 4

##
@test s1.Z ≈ s2.Z
TO.state_diff_jacobian!(s1.model, s1.G, s1.Z)
Altro.errstate_jacobians!(s2.model, s2.G, s2.Z)
s1.G ≈ s2.G

Altro.dynamics_jacobians!(s1)
Altro.dynamics_expansion!(s2)
J1 = [d.∇f for d in s1.D]
J2 = [d.∇f for d in s2.D]
@test J1 ≈ J2

TO.error_expansion!(s1.D, s1.model, s1.G)
Altro.error_expansion!(s2.model, s2.D, s2.G)
∇e1 = [[d.A d.B] for d in s1.D]
∇e2 = [d.∇e for d in s2.D]
@test ∇e1 ≈ ∇e2

TO.cost_expansion!(s1.quad_obj, s1.obj, s1.Z, init=true, rezero=true)
Altro.cost_expansion!(s2.obj, s2.Efull, s2.Z)
for k = 1:prob.N
    @test s1.quad_obj[k].hess ≈ s2.Efull[k].hess
    @test s1.quad_obj[k].grad ≈ s2.Efull[k].grad
end

TO.error_expansion!(s1.E, s1.quad_obj, s1.model, s1.Z, s1.G)
Altro.error_expansion!(s2.model, s2.Eerr, s2.Efull, s2.G, s2.Z)
for k = 1:prob.N
    @test s1.E[k].hess ≈ s2.Eerr[k].hess
    @test s1.E[k].grad ≈ s2.Eerr[k].grad
end

## Backwardpass
s1.ρ[1] = 1e-8 
s2.reg.ρ = 1e-8 
s1.opts.save_S = true
DV1 = Altro.backwardpass!(s1)
DV2 = Altro.backwardpass!(s2)
@test DV1 ≈ DV2
@test s1.K ≈ s2.K
@test s1.d ≈ s2.d
for k = 1:prob.N-1
    @test s1.S[k].xx ≈ s2.S[k].xx
    @test s1.S[k].x ≈ s2.S[k].x
end

s1.Z ≈ s2.Z
Altro.rollout!(s1, 1.0)
Altro.rollout!(s2, 1.0)
s1.Z ≈ s2.Z
# @test states(s1.Z̄) ≈ states(s2.Z̄)
# @test controls(s1.Z̄) ≈ controls(s2.Z̄)

Jprev1 = cost(s1)
Jprev2 = cost(s2)
@test Jprev1 ≈ Jprev2
Jnew1 = Altro.forwardpass!(s1, DV1, Jprev1)
Jnew2 = Altro.forwardpass!(s2, Jprev2)
@test Jnew1 ≈ Jnew2

Altro.copy_trajectories!(s1)
copyto!(s2.Z, s2.Z̄)
@test s1.Z ≈ s2.Z

Altro.gradient_todorov!(s1)
grad2 = Altro.gradient!(s2)
@test s1.grad ≈ s2.grad
@test grad2 ≈ mean(s1.grad)

Altro.record_iteration!(s1, Jprev1, Jprev1-Jnew1)
Altro.record_iteration!(s2, Jprev2, Jprev2-Jnew2, grad2)
s2.stats.iterations

for field in (:iterations, :iteration, :cost, :dJ, :gradient, :c_max, :dJ_zero_counter)
    @test getfield(s1.stats, field) ≈ getfield(s2.stats, field)
end

exit1 = Altro.evaluate_convergence(s1)
exit2 = Altro.evaluate_convergence(s2)
@test exit1 ≈ exit2
SolverLogging.resetcount!(s2.logger)
printlog(s2.logger)

## Try entire solve
prob,opts = Problems.Pendulum()
prob,opts = Problems.Quadrotor()
prob,opts = Problems.Cartpole()
prob,opts = Problems.DubinsCar(:parallel_park)
s1 = Altro.iLQRSolver(prob, opts)
s2 = Altro.iLQRSolver2(copy(prob), copy(opts))
lg = s2.logger
s1.opts.verbose = 2
s2.opts.verbose = 4
solve!(s1)
solve!(s2)
cost(s1) ≈ cost(s2)
iterations(s1) == iterations(s2)
get_trajectory(s1) ≈ get_trajectory(s2)