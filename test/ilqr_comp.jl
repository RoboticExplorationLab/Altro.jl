using Altro
using TrajectoryOptimization
const TO = TrajectoryOptimization

prob,opts = Problems.Pendulum()
s1 = Altro.iLQRSolver(prob, opts)
s2 = Altro.iLQRSolver2(copy(prob), copy(opts))
RD.dims(s2) == RD.dims(s1)

TO.state_diff_jacobian!(s1.model, s1.G, s1.Z)
Altro.errstate_jacobians!(s2.model, s2.G, s2.Z)
s1.G ≈ s2.G

Altro.dynamics_jacobians!(s1)
Altro.dynamics_expansion!(s2)
J1 = [d.∇f for d in s1.D]
J2 = [d.∇f for d in s2.D]
J1 ≈ J2

TO.error_expansion!(s1.D, s1.model, s1.G)
Altro.error_expansion!(s2.model, s2.D, s2.G)
∇e1 = [[d.A d.B] for d in s1.D]
∇e2 = [d.∇e for d in s2.D]
∇e1 ≈ ∇e2

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
