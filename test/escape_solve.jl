@testset "Escape Solve" begin
using FileIO
##
res = load(joinpath(@__DIR__, "escape_solve.jld2"))
prob,opts = Problems.DubinsCar(:escape)
solver = ALTROSolver(prob, opts, infeasible=true, R_inf=0.1)

##
ilqr = Altro.get_ilqr(solver)
let solver = ilqr
    TO.state_diff_jacobian!(solver.model, solver.G, solver.Z)
    TO.dynamics_expansion!( RD.StaticReturn(), RD.ForwardAD(), solver.model, solver.D, solver.Z)
	TO.error_expansion!(solver.D, solver.model, solver.G)
    TO.cost_expansion!(solver.quad_obj, solver.obj.obj, solver.Z, init=true, rezero=true)
    TO.error_expansion!(solver.E, solver.quad_obj, solver.model, solver.Z, solver.G)
end

@test Matrix.(ilqr.G) ≈ fill(zeros(3,3), prob.N+1) 
@test [D.∇f for D in ilqr.D] ≈ res["D"]
@test [E.xx for E in ilqr.E] ≈ res["Q0"]
@test [E.uu for E in ilqr.E] ≈ res["R0"]
@test [E.x for E in ilqr.E] ≈ res["q0"]
@test [E.u for E in ilqr.E] ≈ res["r0"]

conset = get_constraints(solver.solver_al)
let solver = ilqr
    RD.evaluate!(conset, solver.Z)
    RD.jacobian!(conset, solver.Z)
    TO.cost_expansion!(solver.quad_obj, solver.obj, solver.Z, init=true, rezero=true)
    TO.error_expansion!(solver.E, solver.quad_obj, solver.model, solver.Z, solver.G)
end
vals = [Vector.(cv.vals) for cv in conset.convals]
jacs = [Matrix.(cv.jac) for cv in conset.convals]
grad = [Vector.(cv.grad) for cv in conset.convals]
hess = [Matrix.(cv.hess) for cv in conset.convals]

@test vals[1] ≈ res["vals"][1]  # circle constraints
@test vals[2] ≈ res["vals"][2]  # bound constraint
@test vals[3] ≈ res["vals"][3]  # goal constraint
@test vals[4] ≈ res["vals"][4]  # infeasible constraint 
@test [jac[:,1:3] for jac in jacs[1]] ≈ res["jacs"][1]    # circle constraint
@test jacs[2] ≈ res["jacs"][2]                            # bound constraint
@test [jac[:,1:3] for jac in jacs[3]] ≈ res["jacs"][3]    # goal constraint
@test [jac[:,4:end] for jac in jacs[4]] ≈ res["jacs"][4]  # infeasible constraint
@test norm([jac[:,1:3] for jac in jacs[4]]) ≈ 0.0 

@test [grad[1:3] for grad in grad[1]] ≈ res["grad"][1]    # circle constraint
@test grad[2] ≈ res["grad"][2]                            # bound constraint
@test [grad[1:3] for grad in grad[3]] ≈ res["grad"][3]    # goal constraint
@test [grad[4:end] for grad in grad[4]] ≈ res["grad"][4]  # infeasible constraint

@test [hess[1:3,1:3] for hess in hess[1]] ≈ res["hess"][1]      # circle constraint
@test hess[2] ≈ res["hess"][2]                                  # bound constraint
@test [hess[1:3,1:3] for hess in hess[3]] ≈ res["hess"][3]      # goal constraint
@test [hess[4:end,4:end] for hess in hess[4]] ≈ res["hess"][4]  # infeasible constraint

@test [E.xx for E in ilqr.E] ≈ res["Q"]
@test [E.uu for E in ilqr.E] ≈ res["R"]
@test [E.x for E in ilqr.E] ≈ res["q"]
@test [E.u for E in ilqr.E] ≈ res["r"]

ΔV = Altro.backwardpass!(ilqr)
@test ΔV ≈ res["dV"]

J = cost(ilqr)
J_new = Altro.forwardpass!(ilqr, ΔV, J)
@test J_new ≈ res["J_new"]
end