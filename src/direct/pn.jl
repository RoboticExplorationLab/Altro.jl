struct DynamicsVals{T,N,A}
    fVal::Vector{SVector{N,T}}
    xMid::Vector{SVector{N,T}}
    ∇f::Vector{A}
end

function DynamicsVals(dyn_con::DynamicsConstraint)
	DynamicsVals(dyn_con.fVal, dyn_con.xMid, dyn_con.∇f)
end

struct ProblemInfo{T,N}
    model::AbstractModel
    obj::Objective
    conSet::ALConstraintSet{T}
    x0::SVector{N,T}
    xf::SVector{N,T}
end

function ProblemInfo(prob::Problem)
    n = size(prob)[1]
    ProblemInfo(prob.model, prob.obj, ALConstraintSet(prob), SVector{n}(prob.x0), SVector{n}(prob.xf))
end


"""
$(TYPEDEF)
Projected Newton Solver
Direct method developed by the REx Lab at Stanford University
Achieves machine-level constraint satisfaction by projecting onto the feasible subspace.
    It can also take a full Newton step by solving the KKT system.
This solver is to be used exlusively for solutions that are close to the optimal solution.
    It is intended to be used as a "solution polishing" method for augmented Lagrangian methods.
"""
struct ProjectedNewtonSolver{T,N,M,NM} <: ConstrainedSolver{T}
    # Problem Info
    prob::ProblemInfo{T,N}
    Z::Traj{N,M,T,KnotPoint{T,N,M,NM}}
    Z̄::Traj{N,M,T,KnotPoint{T,N,M,NM}}

    opts::SolverOptions{T}
    stats::SolverStats{T}
    P::Primals{T,N,M}
    P̄::Primals{T,N,M}

    H::SparseMatrixCSC{T,Int}
    g::Vector{T}
    # E::Vector{CostExpansion{T,N,N,M}}
    E::QuadraticObjective{N,M,T}

    D::SparseMatrixCSC{T,Int}
    d::Vector{T}
    λ::Vector{T}

    dyn_vals::DynamicsVals{T}
    active_set::Vector{Bool}

    dyn_inds::Vector{SVector{N,Int}}
    con_inds::Vector{<:Vector}
end

function ProjectedNewtonSolver(prob::Problem, 
        opts::SolverOptions=SolverOptions(), stats::SolverStats=SolverStats())
    Z = prob.Z  # grab trajectory before copy to keep associativity
    prob = copy(prob)  # don't modify original problem

    n,m,N = size(prob)
    NN = n*N + m*(N-1)

    # Add dynamics constraints
    TO.add_dynamics_constraints!(prob, integration(prob), idx=1)
    conSet = prob.constraints
    NP = sum(num_constraints(conSet))

    # Trajectory
    prob_info = ProblemInfo(prob)
    Z̄ = copy(prob.Z)

    # Create concatenated primal vars
    P = Primals(n,m,N)
    P̄ = Primals(n,m,N)

    # Allocate Cost Hessian & Gradient
    H = spzeros(NN,NN)
    g = zeros(NN)
    # E = [CostExpansion{Float64}(n,m) for k = 1:N]
    E = QuadraticObjective(n,m,N)

    D = spzeros(NP,NN)
    d = zeros(NP)
    λ = zeros(NP)

    fVal = [@SVector zeros(n) for k = 1:N]
    xMid = [@SVector zeros(n) for k = 1:N-1]
    ∇F = [@SMatrix zeros(n,n+m+1) for k = 1:N]
    dyn_vals = DynamicsVals(fVal, xMid, ∇F)
    active_set = zeros(Bool,NP)

    con_inds = gen_con_inds(conSet)

    # Set constant pieces of the Jacobian
    xinds,uinds = P.xinds, P.uinds

    dyn_inds = SVector{n,Int}[]
    ProjectedNewtonSolver(prob_info, Z, Z̄, opts, stats,
        P, P̄, H, g, E, D, d, λ, dyn_vals, active_set, dyn_inds, con_inds)
end

primals(solver::ProjectedNewtonSolver) = solver.P.Z
primal_partition(solver::ProjectedNewtonSolver) = solver.P.xinds, solver.P.uinds

# AbstractSolver interface
Base.size(solver::ProjectedNewtonSolver{T,n,m}) where {T,n,m} = n,m,length(solver.Z)
TO.get_model(solver::ProjectedNewtonSolver) = solver.prob.model
TO.get_constraints(solver::ProjectedNewtonSolver) = solver.prob.conSet
TO.get_trajectory(solver::ProjectedNewtonSolver) = solver.Z
TO.get_objective(solver::ProjectedNewtonSolver) = solver.prob.obj
iterations(solver::ProjectedNewtonSolver) = solver.stats.iterations_pn
get_active_set(solver::ProjectedNewtonSolver) = solver.active_set
solvername(::Type{<:ProjectedNewtonSolver}) = :ProjectedNewton