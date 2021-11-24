function Cartpole(method=:none; constrained::Bool=true, N=101, 
        Qv=1e-2, Rv=1e-1, Qfv=1e2, u_bnd=3.0, tf=5.0)

    opts = SolverOptions(
        cost_tolerance_intermediate=1e-2,
        penalty_scaling=10.,
        penalty_initial=1.0
    )

    model = RobotZoo.Cartpole()
    n,m = size(model)
    # tf = 5.
    dt = tf/(N-1)

    Q = Qv*Diagonal(@SVector ones(n)) * dt
    Qf = Qfv*Diagonal(@SVector ones(n))
    R = Rv*Diagonal(@SVector ones(m)) * dt
    x0 = @SVector zeros(n)
    xf = @SVector [0, pi, 0, 0]
    obj = LQRObjective(Q,R,Qf,xf,N)

    u_bnd = u_bnd 
    conSet = ConstraintList(n,m,N)
    bnd = BoundConstraint(n,m, u_min=-u_bnd, u_max=u_bnd)
    goal = GoalConstraint(xf)
    if constrained
        add_constraint!(conSet, bnd, 1:N-1)
        add_constraint!(conSet, goal, N:N)
    end

    X0 = [@SVector fill(NaN,n) for k = 1:N]
    u0 = @SVector fill(0.01,m)
    U0 = [u0 for k = 1:N-1]
    Z = Traj(X0,U0,dt*ones(N))
    prob = Problem(model, obj, conSet, x0, xf, Z, N, 0.0, tf, integration=RD.RK3(model))
    rollout!(prob)

    return prob, opts
end