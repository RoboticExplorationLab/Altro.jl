
function DubinsCar(scenario=:three_obstacles; N=101)
    if scenario == :three_obstacles
        opts = SolverOptions(
            cost_tolerance_intermediate=1e-2,
            penalty_scaling=10.,
            penalty_initial=10.0,
        )

        #  Car w/ obstacles
        model = RobotZoo.DubinsCar()
        n,m = RD.dims(model)

        N = 101 # number of knot points
        tf = 5.0
        dt = tf/(N-1)

        x0 = @SVector [0., 0., 0.]
        xf = @SVector [3., 3., 0.]

        Q = Diagonal(@SVector [1., 1., 1.])
        R = Diagonal(@SVector [0.5, 0.5])
        Qf = 10.0*Diagonal(@SVector ones(n))
        obj = LQRObjective(Q*dt,R*dt,Qf,xf,N)

        # create obstacle constraints
        r_circle_3obs = 0.25

        circle_x = 3*@SVector [0.25, 0.5, 0.75]
        circle_y = 3*@SVector [0.25, 0.5, 0.75]
        circle_r = @SVector fill(r_circle_3obs+model.radius, 3)

        obs = CircleConstraint(n, circle_x, circle_y, circle_r)
        bnd = BoundConstraint(n,m, u_min=[0,-3],u_max=[3,3])
        goal = GoalConstraint(xf)

        conSet = ConstraintList(n,m,N)
        add_constraint!(conSet, obs, 2:N-1)
        add_constraint!(conSet, bnd, 1:N-1)
        add_constraint!(conSet, goal, N:N)

        # Create problem
        U = [@SVector fill(0.01,m) for k = 1:N-1]
        car_3obs_static = Problem(model, obj, x0, tf, constraints=conSet, xf=xf)
        initial_controls!(car_3obs_static, U)
        rollout!(car_3obs_static)
        return car_3obs_static, opts

    elseif scenario==:turn90
        opts = SolverOptions(
            cost_tolerance_intermediate=1e-3,
            active_set_tolerance_al=1e-4
        )

        # model
        model = RobotZoo.DubinsCar()
        n,m = RD.dims(model)
        tf = 3.
        dt = tf / (N-1)

        # cost
        d = 1.5
        x0 = @SVector [0., 0., 0.]
        xf = @SVector [d, d,  deg2rad(90)]
        Qf = 100.0*Diagonal(@SVector ones(n))
        Q = (1e-2)*Diagonal(@SVector ones(n))
        R = (1e-2)*Diagonal(@SVector ones(m))

        # problem
        U = [@SVector fill(0.1,m) for k = 1:N-1]
        obj = LQRObjective(Q*dt,R*dt,Qf,xf,N)

        # constraints
        cons = ConstraintList(n,m,N)
        add_constraint!(cons, GoalConstraint(xf), N)

        prob = Problem(model, obj, x0, tf, xf=xf, U0=U)
        rollout!(prob)

        return prob, opts

    elseif scenario==:parallel_park
        opts = SolverOptions(
            cost_tolerance_intermediate=1e-3,
            active_set_tolerance_pn=1e-4
        )

        # model
        model = RobotZoo.DubinsCar()
        n,m = RD.dims(model)
        tf = 3.
        dt = tf / (N-1)

        # cost
        d = 1.5
        x0 = @SVector [0., 0., 0.]
        xf = @SVector [0., d,  0.]
        Qf = 100.0*Diagonal(@SVector ones(n))
        Q = (1e-2)*Diagonal(@SVector ones(n))
        R = (1e-2)*Diagonal(@SVector ones(m))

        # constraints
        u_bnd = 2.
        x_min = @SVector [-0.25, -0.001, -Inf]
        x_max = @SVector [0.25, d + 0.001, Inf]
        bnd_x = BoundConstraint(n,m, x_min=x_min, x_max=x_max)
        bnd_u = BoundConstraint(n,m, u_min=-u_bnd, u_max=u_bnd)
        # bnd = BoundConstraint(n,m,x_min=x_min,x_max=x_max,u_min=-u_bnd,u_max=u_bnd)
        goal = GoalConstraint(xf)

        conSet = ConstraintList(n,m,N)
        # add_constraint!(conSet, bnd, 1:N-1)
        add_constraint!(conSet, bnd_u, 1:N-1)
        add_constraint!(conSet, bnd_x, 2:N-1)
        add_constraint!(conSet, goal, N:N)

        # problem
        U = [@SVector fill(0.1,m) for k = 1:N-1]
        obj = LQRObjective(Q*dt,R*dt,Qf,xf,N)

        prob = Problem(model, obj, x0, tf, constraints=conSet, xf=xf, U0=U)
        rollout!(prob)

        return prob, opts

    elseif scenario==:escape
        opts = SolverOptions(
            constraint_tolerance=1e-5,
            cost_tolerance_intermediate=1e-3,
            penalty_scaling=100.,
            penalty_initial=10.,
            active_set_tolerance_pn=1e-4,
            active_set_tolerance_al=1e-4
        )

        # Static Car Escape
        T = Float64;

        # model
        model = RobotZoo.DubinsCar()
        n,m = RD.dims(model)
        x0 = @SVector [2.5,2.5,0.]
        xf = @SVector [7.5,2.5,0.]
        N = 101
        tf = 3.0
        dt = tf / (N-1)

        # cost
        Q = (1e-3)*Diagonal(@SVector ones(n))
        R = (1e-2)*Diagonal(@SVector ones(m))
        Qf = 100.0*Diagonal(@SVector ones(n))
        obj = LQRObjective(Q*dt,R*dt,Qf,xf,N)

        # constraints
        r = 0.5
        s1 = 30; s2 = 50; s3 = 15

        circles_escape = NTuple{3,Float64}[]

        for i in range(0,stop=5,length=s1)
            push!(circles_escape,(0.,i,r))
        end
        for i in range(0,stop=5,length=s1)
            push!(circles_escape,(5.,i,r))
        end
        for i in range(0,stop=5,length=s1)
            push!(circles_escape,(10.,i,r))
        end
        for i in range(0,stop=10,length=s2)
            push!(circles_escape,(i,0.,r))
        end
        for i in range(0,stop=3,length=s3)
            push!(circles_escape,(i,5.,r))
        end
        for i in range(5,stop=8,length=s3)
            push!(circles_escape,(i,5.,r))
        end

        n_circles_escape = 3*s1 + s2 + 2*s3

        circles_escape
        x,y,r = collect(zip(circles_escape...))
        x = SVector{n_circles_escape}(x)
        y = SVector{n_circles_escape}(y)
        r = SVector{n_circles_escape}(r)

        obs = CircleConstraint(n,x,y,r)
        bnd = BoundConstraint(n,m,u_min=-5.,u_max=5.)
        goal = GoalConstraint(xf)

        conSet = ConstraintList(n,m,N)
        add_constraint!(conSet, obs, 2:N-1)
        add_constraint!(conSet, bnd, 1:N-1)
        add_constraint!(conSet, goal, N:N)

        # Build problem
        U0 = [@SVector ones(m) for k = 1:N-1]

        car_escape_static = Problem(model, obj, x0, tf;
            constraints=conSet, xf=xf)
        initial_controls!(car_escape_static, U0);

        X_guess = [2.5 2.5 0.;
                   4. 5. .785;
                   5. 6.25 0.;
                   7.5 6.25 -.261;
                   9 5. -1.57;
                   7.5 2.5 0.]
        X0_escape = Altro.interp_rows(N,tf,Array(X_guess'))
        initial_states!(car_escape_static, X0_escape)

        return car_escape_static, opts
    end
end
