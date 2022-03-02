############################################################################################
#                           AUGMENTED LAGRANGIAN OBJECTIVE                                 #
############################################################################################

struct ALObjectiveOld{T,O<:Objective} <: AbstractObjective
    obj::O
    constraints::ALConstraintSet{T}
end

function ALObjectiveOld(obj::Objective, cons::ConstraintList, model::AbstractModel)
    ALObjectiveOld(obj, ALConstraintSet(cons, model))
end
@inline ALObjectiveOld(prob::Problem) = ALObjectiveOld(prob.obj, prob.constraints, prob.model)

@inline TO.get_J(obj::ALObjectiveOld) = obj.obj.J
@inline Base.length(obj::ALObjectiveOld) = length(obj.obj)
@inline RobotDynamics.state_dim(obj::ALObjectiveOld) = RobotDynamics.state_dim(obj.obj)
@inline RobotDynamics.control_dim(obj::ALObjectiveOld) = RobotDynamics.control_dim(obj.obj)
@inline TO.ExpansionCache(obj::ALObjectiveOld) = TO.ExpansionCache(obj.obj)


function Base.copy(obj::ALObjectiveOld)
    ALObjectiveOld(obj.obj, ConstraintSet(copy(obj.constraints.constraints), length(obj.obj)))
end

function TO.cost!(obj::ALObjectiveOld, Z::SampledTrajectory)
    # Calculate unconstrained cost
    TO.cost!(obj.obj, Z)

    # Calculate constrained cost
    RD.evaluate!(obj.constraints, Z)
    # update_active_set!(obj.constraints, Val(0.0))
    TO.cost!(TO.get_J(obj), obj.constraints)
end

function TO.cost(obj::ALObjectiveOld, Z::SampledTrajectory)
    TO.cost!(obj, Z)
    return sum(TO.get_J(obj))
end

function TO.cost_expansion!(E, obj::ALObjectiveOld, Z::SampledTrajectory; 
        init::Bool=false, rezero::Bool=false)
    # Update constraint jacobians
    RD.jacobian!(obj.constraints, Z, init)

    # Calculate expansion of original objective
    TO.cost_expansion!(E, obj.obj, Z, init=true, rezero=rezero)  # needs to be computed every time...

    # Add in expansion of constraints
    TO.cost_expansion!(E, obj.constraints)
    # TO.cost_expansion!(E, obj.constraints, Z, true)
end