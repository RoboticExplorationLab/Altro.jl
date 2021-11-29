"""
    AbstractSolver{T} <: MathOptInterface.AbstractNLPEvaluator

Abstract solver for trajectory optimization problems

# Interface
Any type that inherits from `AbstractSolver` must define the following methods:
```julia
model = get_model(::AbstractSolver)::AbstractModel
obj   = get_objective(::AbstractSolver)::AbstractObjective
E     = get_cost_expansion(::AbstractSolver)::QuadraticExpansion  # quadratic error state expansion
Z     = get_trajectory(::AbstractSolver)::Traj
n,m,N = Base.size(::AbstractSolver)
x0    = get_initial_state(::AbstractSolver)::StaticVector
solve!(::AbstractSolver)
```

Optional methods for line search and merit function interface. Note that these do not
    have to return `Traj`
```julia
Z     = get_solution(::AbstractSolver)  # current solution (defaults to get_trajectory)
Z     = get_primals(::AbstractSolver)   # current primals estimate used in the line search
dZ    = get_step(::AbstractSolver)      # current step in the primal variables
```

Optional methods
```julia
opts  = options(::AbstractSolver)       # options struct for the solver. Defaults to `solver.opts`
st    = solver_stats(::AbstractSolver)  # dictionary of solver statistics. Defaults to `solver.stats`
iters = iterations(::AbstractSolver)    #
```
"""
abstract type AbstractSolver{T} end

# Default getters
@inline TO.get_model(solver::AbstractSolver) = solver.model
@inline TO.get_objective(solver::AbstractSolver) = solver.obj
@inline TO.get_trajectory(solver::AbstractSolver) = solver.Z.Z_
@inline get_cost_expansion(solver::AbstractSolver) = solver.E
@inline get_cost_expansion_error(solver::AbstractSolver) = solver.E
@inline get_error_state_jacobians(solver::AbstractSolver) = solver.G
@inline get_initial_state(solver::AbstractSolver) = solver.x0

@inline get_solution(solver::AbstractSolver) = get_trajectory(solver)
@inline get_primals(solver::AbstractSolver) = solver.Z̄
@inline get_step(solver::AbstractSolver) = solver.δZ
@inline stats(solver::AbstractSolver) = solver.stats
iterations(solver::AbstractSolver) = stats(solver).iterations
@inline options(solver::AbstractSolver) = solver.opts
set_options!(solver::AbstractSolver; opts...) = set_options!(options(solver); opts...)
reset!(solver::AbstractSolver) = reset_solver!(solver)  # default method
solvername(solver::S) where S <: AbstractSolver = solvername(S)
is_parentsolver(solver::AbstractSolver) = stats(solver).parent == solvername(solver)

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
    trim!(stats(solver), solvername(solver))

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

"$(TYPEDEF) Unconstrained optimization solver. Will ignore
any constraints in the problem"
abstract type UnconstrainedSolver{T} <: AbstractSolver{T} end


"""$(TYPEDEF)
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

@inline get_duals(solver::ConstrainedSolver) = get_duals(get_constraints(solver))
@inline set_duals!(solver::ConstrainedSolver, λ) = set_duals!(get_constraints(solver), λ)
@inline set_duals!(solver::AbstractSolver, λ) = nothing 


function TO.cost(solver::AbstractSolver, Z=get_trajectory(solver))
    obj = get_objective(solver)
    TO.cost(obj, Z)
end

function TO.cost_expansion!(solver::AbstractSolver)
    Z = get_trajectory(solver)
    E, obj = get_cost_expansion(solver), get_objective(solver)
    cost_expansion!(E, obj, Z)
end

TO.error_expansion!(solver::AbstractSolver) = error_expansion_uncon!(solver)

""" $(SIGNATURES)
Calculate all the constraint values given the trajectory `Z`
"""
function update_constraints!(solver::ConstrainedSolver, Z::Traj=get_trajectory(solver))
    conSet = get_constraints(solver)
    RD.evaluate!(conSet, Z)
end

function update_active_set!(solver::ConstrainedSolver, 
        Z=get_trajectory(solver); tol=solver.opts.active_set_tolerance)
    conSet = get_constraints(solver)
    update_active_set!(conSet, Val(tol))
end

""" $(SIGNATURES)
Calculate all the constraint Jacobians given the trajectory `Z`
"""
function constraint_jacobian!(solver::ConstrainedSolver, Z=get_trajectory(solver))
    conSet = get_constraints(solver)
    RD.jacobian!(conSet, Z)
    return nothing
end

#--- Error Expansion ---#
function error_expansion_uncon!(solver::AbstractSolver)
    E  = get_cost_expansion_error(solver)
    E0 = get_cost_expansion(solver)
    G  = get_error_state_jacobians(solver)
    error_expansion!(E, E0, get_model(solver), G)
end

function TO.error_expansion!(solver::ConstrainedSolver)
    error_expansion_uncon!(solver)
    conSet = get_constraints(solver)
    G = get_error_state_jacobians(solver)
    error_expansion!(conSet, get_model(solver), G)
end

#--- Gradient Norm ---#
function norm_grad(solver::UnconstrainedSolver, recalculate::Bool=true)
    if recalculate
        cost_expansion!(solver)
    end
    norm_grad(get_cost_expansion(solver))
end

function norm_grad(solver::ConstrainedSolver, recalculate::Bool=true)
    conSet = get_constraints(solver)
    if recalculate
        cost_expansion!(solver)
        jacobian
    end
    norm_grad(get_cost_expansion(solver))
end

#--- Rollout ---#
function rollout!(solver::AbstractSolver)
    Z = get_trajectory(solver)
    model = get_model(solver)
    x0 = get_initial_state(solver)
    RD.rollout!(StaticReturn(), model, Z, x0)
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
function TO.initial_trajectory!(solver::AbstractSolver, Z0::Traj)
    Z = get_trajectory(solver)
    for k in eachindex(Z)
        RobotDynamics.setdata!(Z[k], Z0[k].z)
    end
end

# Default getters
@inline RobotDynamics.gettimes(solver::AbstractSolver) = 
    RobotDynamics.get_times(get_trajectory(solver))


# Line Search and Merit Functions
"""
    get_primals(solver)
    get_primals(solver, α::Real)

Get the primal variables used during the line search. When called without an `α` it should
return the container where the temporary primal variables are stored. When called with a
step length `α` it returns `z + α⋅dz` where `z = get_solution(solver)` and `dz = get_step(solver)`.
"""
function get_primals(solver::AbstractSolver, α)
    z = get_solution(solver)
    z̄ = get_primals(solver)
    dz = get_step(solver)
    z̄ .= z .+ α*dz
end

# Constrained solver
TO.num_constraints(solver::AbstractSolver) = num_constraints(get_constraints(solver))

function TO.max_violation(solver::ConstrainedSolver, Z::Traj=get_trajectory(solver); recalculate=true)
    conSet = get_constraints(solver)
    if recalculate
        RD.evaluate!(conSet, Z)
    end
    TO.max_violation(conSet)
end

function TO.norm_violation(solver::ConstrainedSolver, Z::Traj=get_trajectory(solver); recalculate=true, p=2)
    conSet = get_constraints(solver)
    if recalculate
        RD.evaluate!(conSet, Z)
    end
    TO.norm_violation(conSet, p)
end

@inline TO.findmax_violation(solver::ConstrainedSolver) =
    TO.findmax_violation(get_constraints(solver))

function second_order_correction!(solver::ConstrainedSolver)
    conSet = get_constraints(solver)
	Z = get_primals(solver)     # get the current value of z + α⋅δz
	RD.evaluate!(conSet, Z)        # update constraints at new step
	throw(ErrorException("second order correction not implemented yet..."))
end

"""
	cost_dgrad(solver, Z, dZ; recalculate)

Return the scalar directional gradient of the cost evaluated at `Z` in the direction of `dZ`,
where `Z` and `dZ` are the types returned by `get_primals(solver)` and `get_step(solver)`,
and must be able to be converted to a vector of `KnotPoint`s via `Traj()`.
"""
function cost_dgrad(solver::AbstractSolver, Z=get_primals(solver), dZ=get_step(solver);
		recalculate=true)
	E = get_cost_expansion(solver)
	if recalculate
		obj = get_objective(solver)
		cost_gradient!(E, obj, Traj(Z))
	end
	TO.dgrad(E, Traj(dZ))
end

"""
	norm_dgrad(solver, Z, dZ; recalculate, p)

Calculate the directional derivative of `norm(c(Z), p)` in the direction of `dZ`, where
`c(Z)` is the vector of constraints evaluated at `Z`.
`Z` and `dZ` are the types returned by `get_primals(solver)` and `get_step(solver)`,
and must be able to be converted to a vector of `KnotPoint`s via `Traj()`.
"""
function norm_dgrad(solver::AbstractSolver, Z=get_primals(solver), dZ=get_step(solver);
		recalculate=true, p=1)
    conSet = get_constraints(solver)
	if recalculate
		Z_ = Traj(Z)
		RD.evaluate!(conSet, Z_)
		RD.jacobian!(conSet, Z_)
	end
    Dc = TO.norm_dgrad(conSet, Traj(dZ), 1)
end


"""
	dhess(solver, Z, dZ; recalculate)

Calculate the scalar 0.5*dZ'G*dZ where G is the hessian of cost, evaluating the cost hessian
at `Z`.
"""
function cost_dhess(solver::AbstractSolver, Z=TO.get_primals(solver),
		dZ=TO.get_step(solver); recalculate=true)
	E = get_cost_expansion(solver)
	if recalculate
		obj = get_objective(solver)
		cost_hessian!(E, obj, Traj(Z))
	end
	TO.dhess(solver.E, Traj(dZ))
end

# Logging
log_level(solver::AbstractSolver) = OuterLoop

is_verbose(solver::AbstractSolver) = 
    log_level(solver) >= LogLevel(-100*options(solver).verbose)

function set_verbosity!(solver::AbstractSolver)
    llevel = log_level(solver)
    if is_verbose(solver)
        set_logger()
        Logging.disable_logging(LogLevel(llevel.level-1))
    else
        Logging.disable_logging(llevel)
    end
end

function clear_cache!(solver::AbstractSolver)
    llevel = log_level(solver)
    if is_verbose(solver)
        SolverLogging.clear_cache!(global_logger().leveldata[llevel])
    end
end
