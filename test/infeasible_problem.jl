@testset "Infeasible Problem" begin
prob, opts = Problems.DubinsCar(:escape)
model_inf = Altro.InfeasibleModel.(prob.model)
@test RD.dims(model_inf) == (fill(3,prob.N), fill(5,prob.N))
Z0 = prob.Z
Z = Altro.infeasible_trajectory(model_inf, Z0)
@test all(z->length(z) == 8, Z)
conSet = TO.change_dimension(get_constraints(prob), 3, 5, 1:3, 1:2)
inf = Altro.InfeasibleConstraint(model_inf[1])
TO.add_constraint!(conSet, inf, 1:prob.N-1)
obj = Altro.infeasible_objective(prob.obj, 1.0)
@test RD.dims(obj) == (fill(3, prob.N), fill(5, prob.N))
prob_inf = Problem(model_inf, obj, conSet, prob.x0, prob.xf, Z, prob.N, prob.t0, prob.tf)
ilqr = Altro.iLQRSolver(prob_inf, opts, use_static=Val(true))
@test Altro.usestatic(ilqr)
ilqr = Altro.iLQRSolver(prob_inf, opts, use_static=Val(false))
@test Altro.usestatic(ilqr) == false
end