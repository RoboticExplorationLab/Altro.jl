"""
    AbstractSolver{T}

A general abstraction for the solvers in Altro. All solvers support the following methods
from TrajectoryOptimization.jl:

    get_model
    get_objective
    get_trajectory
    set_options!
    cost
    max_violation

As well the following methods specific to Altro.jl:

    stats
    options
"""
abstract type AbstractSolver{T} end

# Default getters
@inline stats(solver::AbstractSolver) = solver.stats
iterations(solver::AbstractSolver) = stats(solver).iterations
@inline options(solver::AbstractSolver) = solver.opts
set_options!(solver::AbstractSolver; opts...) = set_options!(options(solver); opts...)
solvername(solver::S) where S <: AbstractSolver = solvername(S)
is_parentsolver(solver::AbstractSolver) = stats(solver).parent == solvername(solver)

resetstats!(solver::AbstractSolver) = reset!(stats(solver), iterations(solver), solvername(solver))

"""
    reset_solver!(solver::AbstractSolver)

Reset solver stats and constraints.
"""
function reset_solver!(solver::AbstractSolver)
    # Reset the stats only if it's the top level solver
    opts = options(solver)::SolverOptions
    reset!(stats(solver), opts.iterations, solvername(solver))

    if is_constrained(solver)
        reset!(get_constraints(solver), opts)
    end
end

"""
    terminate!(solver::AbstractSolver)

Perform any necessary actions after finishing the solve.
"""
function terminate!(solver::AbstractSolver)
    # Time solve
    stat = stats(solver)
    stat.tsolve = (time_ns() - stat.tstart)*1e-6  # in ms

    # Delete extra stats entries, only if terminal solver
    opts = options(solver)
    if opts.trim_stats
        trim!(stats(solver), solvername(solver))
    end

    # if is_parentsolver(solver)
    #     # Reset global solver
    #     Logging.global_logger(ConsoleLogger())
    # end

    # Print solve summary
    if solver.opts.show_summary && is_parentsolver(solver)
        print_summary(solver)
    end
end

"""
    TerminationStatus

* `UNSOLVED`: Initial value. Solve either hasn't been attempted or is in process.. 
* `SOLVE_SUCCEEDED`: Solve met all the required convergence criteria.
* `MAX_ITERATIONS`: Solve was unable to meet the required convergence criteria within the maximum number of iterations.
* `MAX_ITERATIONS_OUTER`: Solve was unable to meet the required constraint satisfaction the maximum number of outer loop iterations.
* `MAXIMUM_COST`: Cost exceeded maximum allowable cost.
* `STATE_LIMIT`: State values exceeded the imposed numerical limits.
* `CONTROL_LIMIT`: Control values exceeded the imposed numerical limits.
* `NO_PROGRESS`: iLQR was unable to make any progress for `dJ_counter_limit` consecutive iterations.
* `COST_INCREASE`: The cost increased during the iLQR forward pass.
"""
@enum(TerminationStatus, UNSOLVED, LINESEARCH_FAIL, SOLVE_SUCCEEDED, MAX_ITERATIONS, MAX_ITERATIONS_OUTER,
    MAXIMUM_COST, STATE_LIMIT, CONTROL_LIMIT, NO_PROGRESS, COST_INCREASE)

@inline status(solver::AbstractSolver) = stats(solver).status

function print_summary(solver::S) where S <: AbstractSolver
    stat = stats(solver)
    col_h1 = crayon"bold green"
    col_h2 = crayon"bold blue"
    col0 = Crayon(reset=true)
    get_color(v::Bool) = v ? crayon"green" : crayon"red" 

    # Info header
    println(col_h1, "\nSOLVE COMPLETED")
    print(col0," solved using the ")
    print(col0, crayon"bold cyan", solvername(solver))
    print(col0, " Solver,\n part of the Altro.jl package developed by the REx Lab at Stanford and Carnegie Mellon Universities\n")

    # Stats
    println(col_h2, "\n  Solve Statistics")
    println(col0, "    Total Iterations: ", iterations(solver))
    println(col0, "    Solve Time: ", stat.tsolve, " (ms)")

    # Convergence
    println(col_h2, "\n  Covergence")
    if iterations(solver) == 0
        println(crayon"red", "    Solver failed to make it through the first iteration.")
    else
        println(col0, "    Terminal Cost: ", stat.cost[end])
        println(col0, "    Terminal dJ: ", get_color(stat.dJ[end] < solver.opts.cost_tolerance), stat.dJ[end])
        println(col0, "    Terminal gradient: ", get_color(stat.gradient[end] < solver.opts.gradient_tolerance), stat.gradient[end])
        if is_constrained(solver)
            println(col0, "    Terminal constraint violation: ", get_color(stat.c_max[end] < solver.opts.constraint_tolerance), stat.c_max[end])
        end
    end
    println(col0, "    Solve Status: ", crayon"bold", get_color(status(solver) == SOLVE_SUCCEEDED), status(solver))
    print(Crayon(reset=true))  # reset output color
end

"""
    UnconstrainedSolver

Unconstrained optimization solver. Will ignore
any constraints in the problem
"""
abstract type UnconstrainedSolver{T} <: AbstractSolver{T} end


"""
    ConstrainedSolver

Abstract solver for constrained trajectory optimization problems

In addition to the methods required for `AbstractSolver`, all `ConstrainedSolver`s
    must define the following method
```julia
get_constraints(::ConstrainedSolver)::ConstrainSet
```
"""
abstract type ConstrainedSolver{T} <: AbstractSolver{T} end

is_constrained(::Type{<:AbstractSolver})::Bool = true
is_constrained(::Type{<:ConstrainedSolver})::Bool = true
is_constrained(::Type{<:UnconstrainedSolver})::Bool = false
is_constrained(solver::AbstractSolver) = is_constrained(typeof(solver)) && !isempty(get_constraints(solver))

function TO.cost(solver::AbstractSolver, Z=get_trajectory(solver))
    obj = get_objective(solver)
    TO.cost(obj, Z)
end

#--- Rollout ---#
function rollout!(solver::AbstractSolver)
    ilqr = get_ilqr(solver)
    rollout!(ilqr)
end

TO.states(solver::AbstractSolver) = [state(z) for z in get_trajectory(solver)]
function TO.controls(solver::AbstractSolver)
    N = RD.dims(solver)[3]
    Z = get_trajectory(solver)
    [control(Z[k]) for k = 1:N-1]
end

TO.set_initial_state!(solver::AbstractSolver, x0) = copyto!(get_initial_state(solver), x0)

@inline TO.initial_states!(solver::AbstractSolver, X0) = RobotDynamics.setstates!(get_trajectory(solver), X0)
@inline TO.initial_controls!(solver::AbstractSolver, U0) = RobotDynamics.setcontrols!(get_trajectory(solver), U0)
function TO.initial_trajectory!(solver::AbstractSolver, Z0::SampledTrajectory)
    Z = get_trajectory(solver)
    for k in eachindex(Z)
        RobotDynamics.setdata!(Z[k], Z0[k].z)
    end
end

# Default getters
@inline RobotDynamics.gettimes(solver::AbstractSolver) = 
    RobotDynamics.gettimes(get_trajectory(solver))

# Constrained solver
TO.num_constraints(solver::AbstractSolver) = num_constraints(get_constraints(solver))

# @inline TO.findmax_violation(solver::ConstrainedSolver) =
#     TO.findmax_violation(get_constraints(solver))