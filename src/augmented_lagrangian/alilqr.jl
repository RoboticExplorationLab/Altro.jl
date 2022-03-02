@doc raw""" ```julia
struct ALSolver <: ConstrainedSolver{T}
```
Augmented Lagrangian (AL) is a standard tool for constrained optimization. For a trajectory optimization problem of the form:
```math
\begin{aligned}
  \min_{x_{0:N},u_{0:N-1}} \quad & \ell_f(x_N) + \sum_{k=0}^{N-1} \ell_k(x_k, u_k, dt) \\
  \textrm{s.t.}            \quad & x_{k+1} = f(x_k, u_k), \\
                                 & g_k(x_k,u_k) \leq 0, \\
                                 & h_k(x_k,u_k) = 0.
\end{aligned}
```
AL methods form the following augmented Lagrangian function:
```math
\begin{aligned}
    \ell_f(x_N) + &λ_N^T c_N(x_N) + c_N(x_N)^T I_{\mu_N} c_N(x_N) \\
           & + \sum_{k=0}^{N-1} \ell_k(x_k,u_k,dt) + λ_k^T c_k(x_k,u_k) + c_k(x_k,u_k)^T I_{\mu_k} c_k(x_k,u_k)
\end{aligned}
```
This function is then minimized with respect to the primal variables using any unconstrained minimization solver (e.g. iLQR).
    After a local minima is found, the AL method updates the Lagrange multipliers λ and the penalty terms μ and repeats the unconstrained minimization.
    AL methods have superlinear convergence as long as the penalty term μ is updated each iteration.
"""
struct ALSolver{T,S} <: ConstrainedSolver{T}
    opts::SolverOptions{T}
    stats::SolverStats{T}
    ilqr::S
end

function ALSolver(
        prob::Problem{T}, 
        opts::SolverOptions=SolverOptions(), 
        stats::SolverStats=SolverStats(parent=solvername(ALSolver));
        use_static=Val(false), 
        kwarg_opts...
    ) where {T}
    set_options!(opts; kwarg_opts...)

    # Build Augmented Lagrangian Objective
    alobj = ALObjective2{T}(prob.obj)
    prob_al = Problem(prob.model, alobj, ConstraintList(dims(prob)...),
        prob.x0, prob.xf, prob.Z, prob.N, prob.t0, prob.tf)

    
    # Instantiate the iLQR solver
    ilqr = iLQRSolver2(prob_al, opts, stats, use_static=use_static)
    initialize!(alobj.conset, prob.constraints, ilqr.Z, ilqr.opts, alobj.alcost, ilqr.Efull)
    # settraj!(alobj.conset, get_trajectory(ilqr))

    # Build the solver
    solver = ALSolver(opts, stats, ilqr)
    reset!(solver)
    return solver
end

# Getters
for method in (:(RD.dims), :(RD.state_dim), :(RD.errstate_dim), :(RD.control_dim), 
              :(TO.get_trajectory), :(TO.get_objective), :(TO.get_model), 
              :(TO.get_initial_state), :getlogger)
    @eval $method(solver::ALSolver) = $method(solver.ilqr)
end
solvername(::Type{<:ALSolver}) = :AugmentedLagrangian
get_ilqr(solver::ALSolver) = solver.ilqr
TO.get_constraints(solver::ALSolver) = solver.ilqr.obj.conset
stats(solver::ALSolver) = solver.stats
options(solver::ALSolver) = solver.opts

# Methods
function reset!(solver::ALSolver)
    # reset_solver!(solver)
    opts = options(solver)::SolverOptions
    reset!(stats(solver), opts.iterations, solvername(solver))
    reset!(solver.ilqr)

    # Reset constraints
    conset = get_constraints(solver)
    reset!(conset)
end

function TO.max_violation(solver::ALSolver)
    evaluate_constraints!(get_constraints(solver), get_trajectory(solver))
    max_violation(get_constraints(solver))
end