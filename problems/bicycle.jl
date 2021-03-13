function BicycleCar(scenario=:parallel_park, ;N=101)
    # Model
    model = RobotZoo.BicycleModel()
    n,m = size(model)

    if scenario == :parallel_park
        opts = SolverOptions(
            penalty_initial = 1e4,
            verbose = 1,
            cost_tolerance_intermediate = 1e-1
        )

        # Scenario
        tf = 3.0
        x0 = SA_F64[0,0,0,0]
        xf = SA[0,2,deg2rad(0),0]

        # Objective
        Q = Diagonal(SA[1,1,1e-2,1e-2])
        R = Diagonal(SA[1e0,1e0])
        Qf = Diagonal(SA_F64[1,1,1,10])
        obj = LQRObjective(Q,R,Qf,xf,N)

        # Constraints
        cons = ConstraintList(n,m,N)
        bnd = [10,6.0,Inf,deg2rad(45)]
        bnd = BoundConstraint(n, m, x_min=-bnd, x_max=bnd)
        
        add_constraint!(cons, bnd, 1:N-1)
        add_constraint!(cons, GoalConstraint(xf), N)

        # Problem
        prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons)
        initial_controls!(prob, SA[-0.1,0.0])
        rollout!(prob)

        return prob, opts
    elseif scenario == :turn90
        opts = SolverOptions(
            penalty_initial=1e3,
        )

        # Scenario
        tf = 3.0
        x0 = SA_F64[0,0,0,0]
        xf = SA[0,0,deg2rad(90),0]

        # Objective
        Q = Diagonal(SA[1,1,1e-2,1e-2])
        R = Diagonal(SA[1e0,1e0])
        Qf = Diagonal(SA_F64[1,1,1,1])*100
        obj = LQRObjective(Q,R,Qf,xf,N)

        # Constraints
        cons = ConstraintList(n,m,N)
        bnd = [1.5,1.5,Inf,deg2rad(45)]
        bnd = BoundConstraint(n, m, x_min=-bnd, x_max=bnd)
        
        add_constraint!(cons, bnd, 1:N-1)
        add_constraint!(cons, GoalConstraint(xf), N)

        # Problem
        prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons)
        initial_controls!(prob, SA[-0.1,0.0])
        rollout!(prob)

        return prob, opts
    end
    throw(ErrorException("$scenario not a known scenario"))
end