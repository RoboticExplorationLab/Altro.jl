
"$(SIGNATURES) Generate the indices into the concatenated constraint vector for each constraint.
Determines the bandedness of the Jacobian"
function gen_con_inds(conSet::ConstraintList, structure=:by_knotpoint)
	n,m = conSet.n, conSet.m
    N = length(conSet.p)
    numcon = length(conSet.constraints)
    conLen = length.(conSet.constraints)

    cons = [[@SVector ones(Int,length(con)) for j in eachindex(conSet.inds[i])]
		for (i,con) in enumerate(conSet.constraints)]

    # Dynamics and general constraints
    idx = 0
	if structure == :by_constraint
	    for (i,con) in enumerate(conSet.constraints)
			for (j,k) in enumerate(conSet.inds[i])
				cons[i][TO._index(con,k)] = idx .+ (1:conLen[i])
				idx += conLen[i]
	        end
	    end
	elseif structure == :by_knotpoint
		for k = 1:N
			for (i,con) in enumerate(conSet.constraints)
				inds = conSet.inds[i]
				if k in inds
					j = k -  inds[1] + 1
					cons[i][j] = idx .+ (1:conLen[i])
					idx += conLen[i]
				end
			end
		end
	end
    return cons
end

"$(SIGNATURES)
Get the constraint Jacobian structure as a sparse array, and fill in the linear indices
used for filling a vector of the non-zero elements of the Jacobian"
function constraint_jacobian_structure(solver::ConstrainedSolver,
		structure=:by_knopoint)
    n,m,N = size(solver)
    conSet = get_constraints(solver)
    idx = 0.0
    linds = jacobian_linear_inds(solver)

    NN = num_primals(solver)
    NP = num_duals(solver)
    D = spzeros(Int,NP,NN)

    # Number of elements in each block
    blk_len = map(con->length(con.∇c[1]), conSet.constraints)

    # Number of knot points for each constraint
    con_len = map(con->length(con.∇c), conSet.constraints)

    # Linear indices
	if structure == :by_constraint
	    for (i,con) in enumerate(conSet.constraints)
	        for (j,k) in enumerate(con.inds)
	            inds = idx .+ (1:blk_len[i])
	            linds[i][j] = inds
	            con.∇c[j] = inds
	            idx += blk_len[i]
	        end
	    end
	elseif structure == :by_knotpoint
		for k = 1:N
			for (i,con) in enumerate(conSet.constraints)
				if k in con.inds
					inds = idx .+ (1:blk_len[i])
					j = TO._index(con,k)
					linds[i][j] = inds
					con.∇c[j] = inds
					idx += blk_len[i]
				end
			end
		end
	end

    copy_jacobians!(D, solver)
    return D
end
