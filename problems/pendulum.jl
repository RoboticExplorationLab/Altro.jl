
function Pendulum()
    opts = SolverOptions(
        penalty_scaling=100.,
        penalty_initial=0.1,
    )

    model = RobotZoo.Pendulum()
    n,m = RD.dims(model)
    tf = 3.0
    N = 51
    dt = tf / (N-1)

    # cost
    Q = 1e-3*Diagonal(@SVector ones(n))
    R = 1e-3*Diagonal(@SVector ones(m))
    Qf = 1e-0*Diagonal(@SVector ones(n))
    x0 = @SVector zeros(n)
    xf = @SVector [pi, 0.0]  # i.e. swing up
    obj = LQRObjective(Q*dt,R*dt,Qf,xf,N)

    # constraints
    conSet = ConstraintList(n,m,N)
    u_bnd = 3.
    bnd = BoundConstraint(n,m,u_min=-u_bnd,u_max=u_bnd)
    goal = GoalConstraint(xf)
    add_constraint!(conSet, bnd, 1:N-1)
    add_constraint!(conSet, goal, N:N)

    # problem
    U = [@SVector fill(0.1, m) for k = 1:N-1]
    pendulum_static = Problem(model, obj, x0, tf, constraints=conSet, xf=xf)
    initial_controls!(pendulum_static, U)
    rollout!(pendulum_static)
    return pendulum_static, opts
end
