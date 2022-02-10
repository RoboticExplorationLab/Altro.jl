struct ALObjective2{T,O<:AbstractObjective} <: TO.AbstractObjective
    obj::O
    conset::ALConstraintSet2{T}
    function ALObjective2{T}(obj::AbstractObjective, cons::ConstraintList) where T
        new{T,typeof(obj)}(obj, ALConstraintSet2{T}(cons))
    end
end

function TO.cost(alobj::ALObjective2, Z::AbstractTrajectory)
    # Calculate unconstrained cost
    J = TO.cost(alobj.obj, Z)

    # Calculate constraints
    evaluate_constraints!(alobj.conset, Z)

    # Calculate AL penalty
    J += alcost(alobj.conset)    

    return J
end

function cost_expansion!(alobj::ALObjective2, E::CostExpansion2, Z)
    # Calculate expansion of original cost
    cost_expansion!(alobj.obj, E, Z)

    # Calculate constraint Jacobians
    constraint_jacobians!(alobj.conset, Z)

    # Calculate expansion of AL penalty
    algrad!(alobj.conset)
    alhess!(alobj.conset)

    # Add to existing expansion
    add_alcost_expansion!(alobj.conset, E)

    return nothing
end
