export
    ALTROSolverOptions,
    ALTROSolver


"""$(TYPEDEF) Solver options for the ALTRO solver.
$(FIELDS)
"""
@with_kw mutable struct ALTROSolverOptions{T} <: AbstractSolverOptions{T}

    # General
    "Output verbosity. 0 is none, 1 is AL only, 2 is AL and iLQR."
    verbose::Int = 0
    "finish with a projected newton solve."
    projected_newton::Bool = true
    "Solve problem, satisfying specified constraints. If false, will ignore constraints
        and solve with iLQR."
    constrained::Bool = true

    # Tolerances
    "cost tolerance"
    cost_tolerance::T = 1e-4
    "constraint tolerance"
    constraint_tolerance::T = 1e-4
    "constraint satisfaction tolerance that triggers the projected newton solver.
        If set to a non-positive number it will kick out when the maximum penalty is reached."
    projected_newton_tolerance::T = 1.0e-3

    # Other
    "initial penalty term for infeasible controls."
    penalty_initial_infeasible::T = 1.0
    "penalty update rate for infeasible controls."
    penalty_scaling_infeasible::T = 10.0
end



"""$(TYPEDEF)
Augmented Lagrangian Trajectory Optimizer (ALTRO) is a solver developed by the Robotic Exploration Lab at Stanford University.
    The solver is special-cased to solve Markov Decision Processes by leveraging the internal problem structure.

ALTRO consists of two "phases":
1) AL-iLQR: iLQR is used with an Augmented Lagrangian framework to solve the problem quickly to rough constraint satisfaction
2) Projected Newton: A collocation-flavored active-set solver projects the solution from AL-iLQR onto the feasible subspace to achieve machine-precision constraint satisfaction.
"""
struct ALTROSolver{T,S} <: ConstrainedSolver{T}
    opts::ALTROSolverOptions{T}
    solver_al::AugmentedLagrangianSolver{T,S}
    solver_pn::ProjectedNewtonSolver{T}
end

AbstractSolver(prob::Problem, opts::ALTROSolverOptions) = ALTROSolver(prob, opts)

function ALTROSolver(prob::Problem{Q,T};
        infeasible::Bool=false,
        R_inf::Real=1.0,
        solver_uncon=iLQRSolver, opts...) where {Q,T}
    if infeasible
        # Convert to an infeasible problem
        prob = InfeasibleProblem(prob, prob.Z, R_inf/prob.Z[1].dt)

        # Set infeasible constraint parameters
        # con_inf = get_constraints(prob).constraints[end]
        # con_inf::ConstraintVals{T,Control,<:InfeasibleConstraint}
        # con_inf.params.μ0 = opts.penalty_initial_infeasible
        # con_inf.params.ϕ = opts.penalty_scaling_infeasible
    end
    opts_altro = ALTROSolverOptions()
    solver_al = AugmentedLagrangianSolver(prob; solver_uncon=solver_uncon, opts...)
    solver_pn = ProjectedNewtonSolver(prob)#; opts...)
    TO.link_constraints!(get_constraints(solver_pn), get_constraints(solver_al))
    S = typeof(solver_al.solver_uncon)
    solver = ALTROSolver{T,S}(opts_altro, solver_al, solver_pn)
    set_options!(solver; opts...)
    solver
end

# Getters
@inline Base.size(solver::ALTROSolver) = size(solver.solver_pn)
@inline TO.get_trajectory(solver::ALTROSolver)::Traj = get_trajectory(solver.solver_al)
@inline TO.get_objective(solver::ALTROSolver) = get_objective(solver.solver_al)
@inline TO.get_model(solver::ALTROSolver) = get_model(solver.solver_al)
@inline get_initial_state(solver::ALTROSolver) = get_initial_state(solver.solver_al)

function iterations(solver::ALTROSolver)
    if !solver.opts.projected_newton
        iterations(solver.solver_al)
    else
        iterations(solver.solver_al) + iterations(solver.solver_pn)
    end
end

function TO.get_constraints(solver::ALTROSolver)
    if solver.opts.projected_newton
        get_constraints(solver.solver_pn)
    else
        get_constraints(solver.solver_al)
    end
end


# Options methods
function set_options!(solver::ALTROSolver; d...)
    v = get_verbosity(d)
    set_options!(solver.opts; d..., verbose=v[1])
    set_options!(solver.solver_al; d..., verbose=v[1])
    set_options!(solver.solver_pn; d...)
end

function has_option(solver::ALTROSolver, field::Symbol)
    has_option(solver.solver_al, field) || has_option(solver.solver_pn, field) ||
        has_option(solver.opts, field)
end

function get_option(solver::ALTROSolver, field::Symbol)
    if has_option(solver.opts, field)
        return getfield(solver.opts,field)
    elseif has_option(solver.solver_al, field)
        return get_option(solver.solver_al, field)
    elseif has_option(solver.solver_pn, field)
        return get_option(solver.solver_pn, field)
    else
        throw(ErrorException("$field is not a valid option"))
    end
end

function get_options(solver::ALTROSolver, recursive::Bool=true, group::Bool=false)
    d1 = _get_options(solver) 
    if recursive
        d2 = get_options(solver.solver_al, recursive, group)
        d3 = get_options(solver.solver_pn, recursive, group)
        if group
            return Dict(:ALTRO => d1, :AL => d2[:AL], :iLQR => d2[:iLQR], :PN => d3)
        else
            d = merge(merge_opt, d1, d2, d3)
            check_invalid_merge(d, :ALTRO=>d1, :AL=>d2, :PN=>d3)
            return d
        end
    else
        return d1
    end
end

function get_verbosity(d::AbstractDict)::Tuple{Int,Bool,Bool}
    v0 = (0,false,false)
    v1 = (1,true,false)
    v2 = (2,true,true)
    get_verbosity(v::Bool) = v ? v2 : v0 
    function get_verbosity(v::Int)
        if v == 0
            v0
        elseif v == 1
            v1
        elseif v == 2
            v2
        else
            throw(ArgumentError("verbosity must be between 0-2"))
        end
    end
    if :verbose in keys(d)
        return get_verbosity(d[:verbose])
    else
        return v0 
    end
end


# Solve Methods
function solve!(solver::ALTROSolver)
    conSet = get_constraints(solver)

    # Set terminal condition if using projected newton
    opts = solver.opts
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

    # Check convergence
    i = solver.solver_al.stats.iterations
    c_max = solver.solver_al.stats.c_max[i]

    if opts.projected_newton && c_max > opts.constraint_tolerance
        solve!(solver.solver_pn)
    end

end


# Infeasible methods
function InfeasibleProblem(prob::Problem, Z0::Traj, R_inf::Real)
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
    Problem(model_inf, obj, conSet, prob.x0, prob.xf, Z, N, prob.t0, prob.tf)
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

