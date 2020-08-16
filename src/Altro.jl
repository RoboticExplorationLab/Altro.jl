module Altro

import TrajectoryOptimization
import RobotDynamics
using StaticArrays
using Parameters
using DocStringExtensions
using BenchmarkTools
using Interpolations
using UnsafeArrays
using SolverLogging

using SparseArrays
using LinearAlgebra
using Logging
using Statistics

const TO = TrajectoryOptimization

using TrajectoryOptimization:
    integration, num_constraints, get_trajectory

import TrajectoryOptimization: rollout!, get_constraints, get_model, get_objective
import RobotDynamics: discrete_jacobian!, discrete_dynamics, dynamics

using TrajectoryOptimization:
    Problem,
    ConstraintList,
    AbstractObjective, Objective, QuadraticObjective,
    AbstractTrajectory,
    DynamicsExpansion, # TODO: Move to ALTRO
    ALConstraintSet,
    DynamicsConstraint,
    Traj,
    states, controls

using RobotDynamics:
    AbstractModel,
    QuadratureRule, Implicit, Explicit,
    AbstractKnotPoint, KnotPoint, StaticKnotPoint,
    state, control


# types
export
    ALTROSolver,
    # iLQRSolver,
    # AugmentedLagrangianSolver,
    SolverStats,
    SolverOptions

export
    solve!,
    benchmark_solve!,
    iterations,
    set_options!

# modules
export
    Problems


include("utils.jl")
include("infeasible_model.jl")
include("solvers.jl")
include("solver_opts.jl")

include("ilqr/ilqr.jl")
include("ilqr/ilqr_solve.jl")
include("ilqr/backwardpass.jl")
include("ilqr/rollout.jl")
include("augmented_lagrangian/al_solver.jl")
include("augmented_lagrangian/al_objective.jl")
include("augmented_lagrangian/al_methods.jl")
include("direct/primals.jl")
include("direct/pn.jl")
include("direct/pn_methods.jl")
include("altro/altro_solver.jl")

include("direct/copy_blocks.jl")
include("direct/direct_constraints.jl")

include("problems.jl")
# include("deprecated.jl")

end # module
