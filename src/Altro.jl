module Altro

import TrajectoryOptimization
import RobotDynamics
using StaticArrays
using BenchmarkTools
using Interpolations
using SolverLogging
using Crayons

using SparseArrays
using LinearAlgebra
using Logging
using Statistics
using TimerOutputs
using ForwardDiff
using FiniteDiff
import Octavian

const TO = TrajectoryOptimization
const RD = RobotDynamics

using TrajectoryOptimization:
    num_constraints, get_trajectory

import TrajectoryOptimization: rollout!, get_constraints, get_model, get_objective, 
    evaluate_constraints!, constraint_jacobians!
import RobotDynamics: discrete_dynamics, dynamics, dynamics!, evaluate, evaluate!

using TrajectoryOptimization:
    Problem,
    ConstraintList,
    AbstractObjective, Objective, #QuadraticObjective,
    # SampledTrajectory,
    # DynamicsExpansion, # TODO: Move to ALTRO
    # ALConstraintSet,
    # DynamicsConstraint,
    states, controls,
    Equality, Inequality, SecondOrderCone,
    cost

using RobotDynamics:
    AbstractModel, DiscreteDynamics, DiscreteLieDynamics,
    QuadratureRule, Implicit, Explicit,
    FunctionSignature, InPlace, StaticReturn, 
    DiffMethod, ForwardAD, FiniteDifference, UserDefined,
    AbstractKnotPoint, KnotPoint, StaticKnotPoint,
    state_dim, control_dim, output_dim, dims,
    state, control, SampledTrajectory


# types
export
    ALTROSolverOld,
    ALTROSolver,
    # iLQRSolverOld,
    # AugmentedLagrangianSolver,
    SolverStats,
    SolverOptions

export
    solve!,
    benchmark_solve!,
    iterations,
    set_options!,
    max_violation,
    status

# modules
export
    Problems

const ColonSlice = Base.Slice{Base.OneTo{Int}}
const SparseView{T,I} = SubArray{T, 2, SparseMatrixCSC{T, I}, Tuple{UnitRange{I}, UnitRange{I}}, false}
const VectorView{T,I} = SubArray{T, 1, Vector{T}, Tuple{UnitRange{I}}, true}

# Select the matix multiplication kernel
const USE_OCTAVIAN = parse(Bool, get(ENV, "ALTRO_USE_OCTAVIAN", "false"))
@static if USE_OCTAVIAN
    const matmul! = Octavian.matmul!
else
    const matmul! = mul!
end

# Include the QDLDL wrapper
include("qdldl.jl")
using .Cqdldl

# High level structs
include("utils.jl")
include("infeasible_model.jl")
include("solvers.jl")
include("solver_opts.jl")

# iLQR Solver
include("ilqr/cost_expansion.jl")
include("ilqr/dynamics_expansion.jl")
include("ilqr/ilqr_solver.jl")
include("ilqr/backwardpass.jl")
include("ilqr/forwardpass.jl")
include("ilqr/ilqr_solve.jl")

# Augmented Lagrangian Solver
include("augmented_lagrangian/alcon.jl")
include("augmented_lagrangian/alconset.jl")
include("augmented_lagrangian/al_objective.jl")
include("augmented_lagrangian/al_solver.jl")
include("augmented_lagrangian/al_solve.jl")
include("direct/sparseblocks.jl")

# Projected Newton Solver
include("direct/pncon.jl")
include("direct/pnconset.jl")
include("direct/pn_solver.jl")
include("direct/pn_solve.jl")

# ALTRO Solver
include("altro/altro_solver.jl")
include("altro/altro_solve.jl")

# Example problems submodule
include("problems.jl")

end # module
