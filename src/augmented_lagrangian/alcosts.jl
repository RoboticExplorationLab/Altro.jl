# Top-level functions that dispatch to the individual constraints
function TO.cost!(J::Vector{<:Real}, conSet::ALConstraintSet)
    for i in eachindex(conSet.convals)
        TO.cost!(J, conSet.convals[i])
    end
end

function TO.cost_expansion!(E::Objective, conSet::ALConstraintSet)
    for i in eachindex(conSet.convals)
        TO.cost_expansion!(E, conSet.convals[i])
    end
end

# Cost function on a single constraint
function TO.cost!(J, conval::ALConVal)
    cone = TO.sense(conval)
    for (i,k) in enumerate(conval.inds)
        c = SVector(conval.vals[i])
        λ = SVector(conval.λ[i])
        μ = SVector(conval.μ[i])
        J[k] += TO.cost(cone, λ, c, μ) 
    end
end

TO.cost(::TO.Equality, λ, c, μ) = λ'c + 0.5*c'Diagonal(μ)*c

function TO.cost(::TO.Inequality, λ, c, μ) 
    a = @. (c >= 0) | (λ > 0)
	return λ'c + 0.5*c'Diagonal(μ .* a)*c
end

function TO.cost(cone::TO.SecondOrderCone, λ, c, μ)
	λbar = TO.projection(cone, λ - μ .* c)
	Iμ = Diagonal(1 ./ μ)
	return 0.5 * (λbar'Iμ*λbar - λ'Iμ*λ)
end

# Cost expansion on a single constraint across all time steps
function TO.cost_expansion!(E::Objective, conval)
    cone = TO.sense(conval)
    for (i,k) in enumerate(conval.inds)
        TO.cost_expansion!(cone, conval, i)
    end
    # add the values to the existing cost 
    copy_expansion!(E, conval)
end

function TO.cost_expansion!(::Equality, conval, i)
    c = SVector(conval.vals[i])
    ∇c = conval.jac[i]
    λ = SVector(conval.λ[i])
    # μ = SVector(conval.μ[i])
    # Iμ = Diagonal(μ)
    μ = conval.μ[i][1]
    
    λbar = λ + μ * c
    conval.const_hess[i] = true
    # conval.grad[i] = ∇c'λbar
    # conval.hess[i] = μ*∇c'∇c
    mul!(conval.grad[i], Transpose(∇c.data), λbar)
    mul!(conval.hess[i], Transpose(∇c), ∇c)
    conval.hess[i] .*= μ
    return
end

function TO.cost_expansion!(::Inequality, conval, i)
    c = SVector(conval.vals[i])
    ∇c = conval.jac[i]
    tmp = conval.tmp
    λ = SVector(conval.λ[i])
    # μ = SVector(conval.μ[i])
    μ = conval.μ[i][1]
    a = @. (c >= 0) | (λ > 0)
    Iμ = Diagonal(a)
    
    λbar = λ + μ * (a .* c)
    conval.const_hess[i] = false 
    # conval.grad[i] = ∇c'λbar
    # conval.hess[i] = μ*∇c'Iμ*∇c

    mul!(conval.grad[i], Transpose(∇c.data), λbar)
    mul!(tmp.data, Iμ, ∇c.data)  # TODO: do this directly on the SizedArrays
    mul!(conval.hess[i], Transpose(∇c), tmp)
    conval.hess[i] .*= μ
end

function TO.cost_expansion!(cone::SecondOrderCone, conval, i)
    c = SVector(conval.vals[i])
    ∇c = conval.jac[i]
    λ = SVector(conval.λ[i])
    μ = SVector(conval.μ[i])
    Iμ = Diagonal(μ)
    ∇proj = conval.∇proj[i]  # pxp projection jacobian
    ∇²proj = conval.∇²proj[i]

    # The term inside the projection operator
    λbar = λ - μ .* c

    # Evaluate the projection and it's derivatives
    λp = TO.projection(cone, λbar)      # TODO: don't project more than you need to!
    TO.∇projection!(cone, ∇proj, λbar)        # evaluate the Jacobian of the projection operation
    TO.∇²projection!(cone, ∇²proj, λbar, λp)  # evalute the Jacobian of ∇Π(λ - μ*c)'Π(λ - μ*c)
    
    # Apply the chain rule
    μ = μ[1]
	∇cproj = -∇proj*∇c
	∇²cproj = ∇c'*∇²proj*∇c
    
    # Combine and store the result
	conval.grad[i] = ∇cproj'*λp 
    conval.hess[i] = μ*(∇cproj'∇cproj .+ ∇²cproj)
    conval.const_hess[i] = false
end

function copy_expansion!(E::QuadraticObjective{n,m}, conval::ALConVal{<:TO.StateConstraint}) where {n,m}
    ix,iu = SVector{n}(1:n), SVector{m}(1:m) .+ n
    for (i,k) in enumerate(conval.inds)
        E[k].q .+= conval.grad[i]
        E[k].Q .+= conval.hess[i]
        E.const_hess[k] &= conval.is_const[i]
    end
end

function copy_expansion!(E::QuadraticObjective{n,m}, conval::ALConVal{<:TO.ControlConstraint}) where {n,m}
    ix,iu = SVector{n}(1:n), SVector{m}(1:m) .+ n
    for (i,k) in enumerate(conval.inds)
        E[k].r .+= conval.grad[i]
        E[k].R .+= conval.hess[i]
        E.const_hess[k] &= conval.is_const[i]
    end
end

function copy_expansion!(E::QuadraticObjective{n,m}, conval::ALConVal{<:TO.StageConstraint}) where {n,m}
    ix,iu = SVector{n}(1:n), SVector{m}((1:m) .+ n)
    for (i,k) in enumerate(conval.inds)
        E[k].q .+= conval.grad[i][ix]
        E[k].r .+= conval.grad[i][iu]
        E[k].Q .+= conval.hess[i][ix,ix]
        E[k].R .+= conval.hess[i][iu,iu]
        E[k].H .+= conval.hess[i][iu,ix]
        E.const_hess[k] &= conval.is_const[i]
    end
end

function copy_expansion!(E::QuadraticObjective{n,m}, conval::ALConVal{<:TO.CoupledConstraint}) where {n,m}
    ix,iu = SVector{n}(1:n), SVector{m}((1:m) .+ n)
    for (i,k) in enumerate(conval.inds)
        E[k].q .+= conval.grad[i][ix]
        E[k].r .+= conval.grad[i][iu]
        E[k].Q .+= conval.hess[i][ix,ix]
        E[k].R .+= conval.hess[i][iu,iu]
        E[k].H .+= conval.hess[i][iu,ix]
        E[k+1].q .+= conval.grad[i,2][ix]
        E[k+1].r .+= conval.grad[i,2][iu]
        E[k+1].Q .+= conval.hess[i,2][ix,ix]
        E[k+1].R .+= conval.hess[i,2][iu,iu]
        E[k+1].H .+= conval.hess[i,2][iu,ix]
        E.const_hess[k] &= conval.is_const[i]
    end
end