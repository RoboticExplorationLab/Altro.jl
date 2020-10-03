struct ALConVal{C,V,M} <: TO.AbstractConstraintValues{C}
    con::C
    inds::Vector{Int}
    vals::Vector{V}
    jac::Matrix{M}
    λ::Vector{V}
    μ::Vector{V}
    λbar::Vector{V}
    c_max::Vector{Float64}
    is_const::Vector{Vector{Bool}}
    iserr::Bool

    ∇proj::Vector{Matrix{Float64}}   # (p,p) projection Jacobian
    ∇²proj::Vector{Matrix{Float64}}  # (p,p) projection "Hessian," or Jacobian of ∇Π'Π
    grad::Vector{Vector{Float64}}    # gradient of augmented Lagrangian wrt inputs
    hess::Vector{Matrix{Float64}}    # hessian of augmented Lagrangian wrt inputs

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
        is_const = [zeros(Bool,P), zeros(Bool,P)]
        
        
        ni = size(jac[1],2)  # size of inputs to the constraint
        ∇proj  = [zeros(p,p) for i = 1:P]
        ∇²proj = [zeros(p,p) for i = 1:P]
        grad = [zeros(ni) for i = 1:P]
        hess = [zeros(ni,ni) for i = 1:P]

        new{typeof(con), eltype(vals), eltype(jac)}(con,
            collect(inds), vals,jac, λ, μ, λbar, c_max, is_const, iserr,
            ∇proj, ∇²proj, grad, hess)
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
