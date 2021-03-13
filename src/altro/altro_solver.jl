
"""$(TYPEDEF)
Augmented Lagrangian Trajectory Optimizer (ALTRO) is a solver developed by the Robotic Exploration Lab at Stanford University.
    The solver is special-cased to solve Markov Decision Processes by leveraging the internal problem structure.

ALTRO consists of two "phases":
1) AL-iLQR: iLQR is used with an Augmented Lagrangian framework to solve the problem quickly to rough constraint satisfaction
2) Projected Newton: A collocation-flavored active-set solver projects the solution from AL-iLQR onto the feasible subspace to achieve machine-precision constraint satisfaction.

# Constructor

    ALTROSolver(prob::Problem, opts::SolverOptions; [infeasible, R_inf, kwarg_opts...])

The `infeasible` keyword is a boolean flag that specifies whether the solver should
augment the controls to make it artificially fully actuated, allowing state initialization.
The `R_inf` is the weight on these augmented controls. Any solver options can be passed
as additional keyword arguments and will be set in the solver.

# Getters
* `get_model`
* `get_objective`
* `get_constraints`
* `Altro.get_ilqr`
* `TO.get_initial_state`

# Other methods
* `Base.size`: returns `(n,m,N)`
* `TO.integration`
* `TO.is_constrained`
"""
struct ALTROSolver{T,S} <: ConstrainedSolver{T}
    opts::SolverOptions{T}
    stats::SolverStats{T}
    solver_al::AugmentedLagrangianSolver{T,S}
    solver_pn::ProjectedNewtonSolver{T}
end

function ALTROSolver(prob::Problem{Q,T}, opts::SolverOptions=SolverOptions();
        infeasible::Bool=false,
        R_inf::Real=1.0,
        solver_uncon=iLQRSolver,
        kwarg_opts...
    ) where {Q,T}
    if infeasible
        # Convert to an infeasible problem
        prob = InfeasibleProblem(prob, prob.Z, R_inf/prob.Z[1].dt)

        # Set infeasible constraint parameters
        # con_inf = get_constraints(prob).constraints[end]
        # con_inf::ConstraintVals{T,Control,<:InfeasibleConstraint}
        # con_inf.params.μ0 = opts.penalty_initial_infeasible
        # con_inf.params.ϕ = opts.penalty_scaling_infeasible
    end
    set_options!(opts; kwarg_opts...)
    stats = SolverStats{T}(parent=solvername(ALTROSolver))
    solver_al = AugmentedLagrangianSolver(prob, opts, stats; solver_uncon=solver_uncon)
    solver_pn = ProjectedNewtonSolver(prob, opts, stats)
    link_constraints!(get_constraints(solver_pn), get_constraints(solver_al))
    S = typeof(solver_al.solver_uncon)
    solver = ALTROSolver{T,S}(opts, stats, solver_al, solver_pn)
    reset!(solver)
    # set_options!(solver; opts...)
    solver
end

# Getters
@inline Base.size(solver::ALTROSolver) = size(solver.solver_pn)
@inline TO.get_trajectory(solver::ALTROSolver)::Traj = get_trajectory(solver.solver_al)
@inline TO.get_objective(solver::ALTROSolver) = get_objective(solver.solver_al)
@inline TO.get_model(solver::ALTROSolver) = get_model(solver.solver_al)
@inline get_initial_state(solver::ALTROSolver) = get_initial_state(solver.solver_al)
solvername(::Type{<:ALTROSolver}) = :ALTRO
TrajectoryOptimization.integration(solver::ALTROSolver) = integration(solver.solver_al)
is_constrained(solver::ALTROSolver) = !isempty(get_constraints(solver.solver_al))
@inline get_ilqr(solver::ALTROSolver) = solver.solver_al.solver_uncon

function TO.get_constraints(solver::ALTROSolver)
    if solver.opts.projected_newton
        get_constraints(solver.solver_pn)
    else
        get_constraints(solver.solver_al)
    end
end



# Solve Methods
function solve!(solver::ALTROSolver)
    reset!(solver)
    conSet = get_constraints(solver.solver_al)

    if isempty(conSet) 
        ilqr = solver.solver_al.solver_uncon
        solve!(ilqr)
        terminate!(solver)
        return solver
    end

    # Set terminal condition if using projected newton
    opts = solver.opts
    ϵ_con = solver.opts.constraint_tolerance
    if opts.projected_newton
        opts_al = solver.solver_al.opts
        if opts.projected_newton_tolerance >= 0
            opts_al.constraint_tolerance = opts.projected_newton_tolerance
        else
            opts_al.constraint_tolerance = 0
            opts_al.kickout_max_penalty = true
        end
    end

    # Solve with AL
    solve!(solver.solver_al)

    if status(solver) <= SOLVE_SUCCEEDED || opts.force_pn
        # Check convergence
        i = solver.solver_al.stats.iterations
        if i > 1
            c_max = solver.solver_al.stats.c_max[i]
        else
            c_max = TO.max_violation(solver.solver_al)
        end

        opts.constraint_tolerance = ϵ_con
        if (opts.projected_newton && c_max > opts.constraint_tolerance && 
                (status(solver) <= SOLVE_SUCCEEDED || status(solver) == MAX_ITERATIONS_OUTER)) ||
                opts.force_pn
            solve!(solver.solver_pn)
        end

        # Back-up check
        if status(solver) <= SOLVE_SUCCEEDED 
            # TODO: improve this check
            if TO.max_violation(solver) < solver.opts.constraint_tolerance
                solver.stats.status = SOLVE_SUCCEEDED
            end
        end
    end

    terminate!(solver)
    solver
end


# Infeasible methods
function InfeasibleProblem(prob::Problem{RK}, Z0::Traj, R_inf::Real) where RK
    @assert !isnan(sum(sum.(states(Z0))))

    n,m,N = size(prob)  # original sizes

    # Create model with augmented controls
    model_inf = InfeasibleModel(prob.model)

    # Get a trajectory that is dynamically feasible for the augmented problem
    #   and matches the states and controls of the original guess
    Z = infeasible_trajectory(model_inf, Z0)

    # Convert constraints so that they accept new dimensions
    conSet = TO.change_dimension(get_constraints(prob), n, m+n, 1:n, 1:m)

    # Constrain additional controls to be zero
    inf = InfeasibleConstraint(model_inf)
    TO.add_constraint!(conSet, inf, 1:N-1)

    # Infeasible Objective
    obj = infeasible_objective(prob.obj, R_inf)

    # Create new problem
    Problem{RK}(model_inf, obj, conSet, prob.x0, prob.xf, Z, N, prob.t0, prob.tf)
end

function infeasible_objective(obj::Objective, regularizer)
    n,m = TO.state_dim(obj.cost[1]), TO.control_dim(obj.cost[1])
    Rd = [@SVector zeros(m); @SVector fill(regularizer,n)]
    R = Diagonal(Rd)
    cost_inf = TO.DiagonalCost(Diagonal(@SVector zeros(n)), R, checks=false)
    costs = map(obj.cost) do cost
        cost_idx = TO.change_dimension(cost, n, n+m, 1:n, 1:m)
        cost_idx + cost_inf
    end
    TO.Objective(costs)
end

