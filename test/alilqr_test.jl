
prob, opts = Problems.Quadrotor()
al = Altro.ALSolver(prob, copy(opts), use_static=Val(true))
conset = get_constraints(al)
ilqr = Altro.get_ilqr(al)
@test conset[1].E === ilqr.Efull
@test conset[1].Z[1] === ilqr.Z
@test conset[1].opts.solveropts === ilqr.opts === al.opts
@test conset[1].cost === get_objective(ilqr).cost