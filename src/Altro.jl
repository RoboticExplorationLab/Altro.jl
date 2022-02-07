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
using Crayons

using SparseArrays
using LinearAlgebra
using Logging
using Statistics
using TimerOutputs
using ForwardDiff
using FiniteDiff

const TO = TrajectoryOptimization
const RD = RobotDynamics

using TrajectoryOptimization:
    num_constraints, get_trajectory

import TrajectoryOptimization: rollout!, get_constraints, get_model, get_objective
import RobotDynamics: discrete_dynamics, dynamics, dynamics!, evaluate, evaluate!

using TrajectoryOptimization:
    Problem,
    ConstraintList,
    AbstractObjective, Objective, #QuadraticObjective,
    AbstractTrajectory,
    DynamicsExpansion, # TODO: Move to ALTRO
    # ALConstraintSet,
    DynamicsConstraint,
    Traj,
    states, controls,
    Equality, Inequality, SecondOrderCone

using RobotDynamics:
    AbstractModel, DiscreteDynamics, DiscreteLieDynamics,
    QuadratureRule, Implicit, Explicit,
    FunctionSignature, InPlace, StaticReturn, 
    DiffMethod, ForwardAD, FiniteDifference, UserDefined,
    AbstractKnotPoint, KnotPoint, StaticKnotPoint,
    state_dim, control_dim, output_dim, dims,
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
    set_options!,
    status

# modules
export
    Problems

const ColonSlice = Base.Slice{Base.OneTo{Int}}

include("logging/SolverLogging.jl")
using .SolverLogging_v1

include("linalg.jl")
include("utils.jl")
include("infeasible_model.jl")
include("solvers.jl")
include("solver_opts.jl")

include("ilqr/cost_expansion.jl")
include("ilqr/dynamics_expansion.jl")
include("ilqr/ilqr2.jl")
include("ilqr/backwardpass2.jl")
include("ilqr/forwardpass.jl")
include("ilqr/ilqr_solve2.jl")

include("ilqr/ilqr.jl")
include("ilqr/ilqr_solve.jl")
include("ilqr/backwardpass.jl")
include("ilqr/rollout.jl")
# include("augmented_lagrangian/conic_penalties.jl")
include("augmented_lagrangian/alconval.jl")
include("augmented_lagrangian/ALconset.jl")
include("augmented_lagrangian/alcosts.jl")
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
