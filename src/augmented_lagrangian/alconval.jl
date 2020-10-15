Base.@kwdef mutable struct ConstraintParams{T}
	ϕ::T = NaN 	      # penalty scaling parameter
	μ0::T = NaN 	  # initial penalty parameter
	μ_max::T = NaN    # max penalty parameter
	λ_max::T = NaN    # max Lagrange multiplier
end

function ConstraintParams(ϕ::Real, μ0::Real, μ_max::Real, λ_max::Real)
    params = promote(ϕ, μ0, μ_max, λ_max)
	ConstraintParams{eltype(params)}(params...)
end

struct ALConVal{C,V,M} <: TO.AbstractConstraintValues{C}
    con::C
    inds::Vector{Int}
    vals::Vector{V}
    jac::Matrix{M}
    λ::Vector{V}
    μ::Vector{V}
    λbar::Vector{V}
    c_max::Vector{Float64}
    is_const::BitArray{2}
    iserr::Bool
	params::ConstraintParams{Float64}

    tmp::M
    ∇proj::Vector{Matrix{Float64}}   # (p,p) projection Jacobian
    ∇²proj::Vector{Matrix{Float64}}  # (p,p) projection "Hessian," or Jacobian of ∇Π'Π
    grad::Vector{Vector{Float64}}    # gradient of augmented Lagrangian wrt inputs
    hess::Vector{Matrix{Float64}}    # hessian of augmented Lagrangian wrt inputs
    const_hess::BitVector

	function ALConVal(n::Int, m::Int, con::TO.AbstractConstraint, inds::AbstractVector{Int}, 
			jac, vals, iserr::Bool=false)
		if !iserr && size(TO.gen_jacobian(con)) != size(jac[1])
			throw(DimensionMismatch("size of jac[i] $(size(jac[1])) does not match the expected size of $(size(gen_jacobian(con)))"))
		end
        vals2 = deepcopy(vals)
        λ = deepcopy(vals)
        μ = deepcopy(vals)
        λbar = deepcopy(vals)

        p = length(con)
        P = length(vals)
        ix = 1:n
        iu = n .+ (1:m)
        c_max = zeros(P)
        is_const = BitArray(undef, size(jac)) 
        params = ConstraintParams()

        # jac = [jac; jac[end:end,:]]  # append extra for temporary array
        tmp = zero(jac[1])
        
        ni = size(jac[1],2)  # size of inputs to the constraint
        ∇proj  = [zeros(p,p) for i = 1:P]
        ∇²proj = [zeros(p,p) for i = 1:P]
        grad = [zeros(ni) for i = 1:P]
        hess = [zeros(ni,ni) for i = 1:P]
        const_hess = BitArray(undef, P)

        new{typeof(con), eltype(vals), eltype(jac)}(con,
            collect(inds), vals,jac, λ, μ, λbar, c_max, is_const, iserr, params,
            tmp, ∇proj, ∇²proj, grad, hess, const_hess)
    end
end

function ALConVal(n::Int, m::Int, cval::ALConVal)
	# create a ConVal for the "raw" Jacobians, if needed
	# 	otherwise return the same ConVal
	if cval.iserr
		p = length(cval.con)
		ws = TO.widths(cval.con, n, m)
		jac = [SizedMatrix{p,w}(zeros(p,w)) for k in cval.inds, w in ws]
		ALConVal(n, m, cval.con, cval.inds, jac, cval.vals, false)
	else
		return cval
	end
end

function ALConVal(n::Int, m::Int, con::TO.AbstractConstraint, inds::UnitRange{Int}, iserr::Bool=false)
	C,c = TO.gen_convals(n,m,con,inds)
	ALConVal(n, m, con, inds, C, c)
end

TO.sense(cval::ALConVal) = TO.sense(cval.con)

function reset_duals!(con::ALConVal)
    for i in eachindex(con.λ)
        con.λ[i] .*= 0
    end
end

function reset_penalties!(con::ALConVal)
    for i in eachindex(con.μ)
        con.μ[i] .= con.params.μ0
    end
end

function set_params!(cval::ALConVal, opts)
    if isnan(cval.params.ϕ)
        cval.params.ϕ = opts.penalty_scaling
    end
    if isnan(cval.params.μ0)
        cval.params.μ0 = opts.penalty_initial
    end
    if isnan(cval.params.μ_max)
        cval.params.μ_max = opts.penalty_max
    end
    if isnan(cval.params.λ_max)
        cval.params.λ_max = opts.dual_max
    end
end