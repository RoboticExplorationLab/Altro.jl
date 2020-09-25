using StaticArrays
using BenchmarkTools

function projection(v)
    max.(0, v)
end

function ∇projection!(J, v)

end

function incone(x)
    all(x .>= 0)
end

a = @SVector rand(10)
@btime projection($a)
@btime incone($x)

x = @SVector rand(100)
@btime norm($x,2)
@btime norm($x,1)
@btime norm($x,Inf)

function penalty(c, λ)
	is_incone = all(c .<= 0)	   # check if primals are in the cone
	inactive = norm(λ,Inf) < 1e-9  # check duals are near the origin
	if is_incone && inactive
		return zero(c)
	else                           # penalize the projection onto the cone
		return max.(0, c)
	end
	a = @. (c >= 0) | (λ > 0)
	return a .* c
end