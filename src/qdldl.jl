
module Cqdldl

using SparseArrays
using LinearAlgebra

using QDLDL_jll
const libqdldl = QDLDL_jll.libqdldl

const QDLDL_int = Clonglong
const QDLDL_float = Cdouble
const QDLDL_bool = Cuchar

export QDLDLException, QDLDLSolver

struct QDLDLException <: Exception
    msg::String
end

Base.@kwdef mutable struct QDLDLFlags
    haselimtree::Bool = false
    isfactordone::Bool = false
    isfactorsuccess::Bool = false
end

mutable struct QDLDLBufferDims{Ti}
    n::Ti      # current dimension
    n0::Ti     # max dimension size
    nnzA0::Ti  # max nnzA
    nnzL0::Ti  # max nnzL
end

struct QDLDLSolver{Tv <: Union{Float32, Float64}, Ti <: Integer}
    flags::QDLDLFlags
    dims::QDLDLBufferDims{Ti}
    Ap::Vector{Ti}
    Ai::Vector{Ti}
    Ax::Vector{Tv}
    work::Vector{Ti}
    Lnz::Vector{Ti}
    etree::Vector{Ti}
    Lp::Vector{Ti}
    Li::Vector{Ti}
    Lx::Vector{Tv}
    D::Vector{Tv}
    Dinv::Vector{Tv}
    bwork::Vector{QDLDL_bool}
    iwork::Vector{Ti}
    fwork::Vector{Tv}
    # L::SparseMatrixCSC{Tv,Ti}
    function QDLDLSolver{Tv}(n::Integer, nnzA::Integer, nnzL::Integer) where Tv
        n, nnzA, nnzL = promote(n, nnzA, nnzL)
        dims = QDLDLBufferDims(n, n, nnzA, nnzL)
        Ti = typeof(n)

        flags = QDLDLFlags()
        Ap = zeros(Ti, n + 1)
        Ai = zeros(Ti, nnzA)
        Ax = zeros(Tv, nnzA)
        work = zeros(Ti,n)
        Lnz = zeros(Ti,n)
        etree = zeros(Ti,n)
        Lp = zeros(Ti, n+1) 
        Lp[end] = nnzL 
        Li = zeros(Ti, nnzL) 
        Lx = zeros(Tv, nnzL) 
        D = zeros(Tv, n)
        Dinv = zeros(Tv, n)
        bwork = zeros(QDLDL_bool, n)
        iwork = zeros(Ti, 3n)
        fwork = zeros(Tv, n)
        new{Tv,Ti}(flags, dims, Ap, Ai, Ax, work, Lnz, etree, Lp, Li, Lx, D, Dinv, bwork, iwork, fwork)
    end
    function QDLDLSolver(A::SparseMatrixCSC{Tv}; calctree::Bool=true) where Tv <: Union{Float32, Float64}
        Ti = QDLDL_int 
        flags = QDLDLFlags()

        # Convert to upper triangular
        if !istriu(A)
            @warn "A not upper triangular. Creating a new upper-triangular matrix."
            A = triu(A)
        end
        @assert A.n == A.m "A must be a square matrix."
        n = A.n
        Ap = A.colptr .- 1  # convert to 0-based indexing
        Ai = A.rowval .- 1  # convert to 0-based indexing
        Ax = A.nzval
        work = zeros(Ti,n)
        Lnz = zeros(Ti,n)
        etree = zeros(Ti,n)
        F = new{Tv,Ti}(flags, n, Ap, Ai, Ax, work, Lnz, etree)
        if calctree
            return eliminationtree!(F)
        end
        return F
    end
    function QDLDLSolver(qdldl::QDLDLSolver{Tv,Ti}) where {Tv,Ti}
        nnzL = sum(qdldl.Lnz)
        n = qdldl.n
        nnzL = sum(qdldl.Lnz)
        if nnzL == 0 || !qdldl.flags.haselimtree
            throw(QDLDLException(
                "Cannot initialize a complete QDLDL objective without calculating the elimination tree."
            ))
        end
        Lp = zeros(Ti, n+1) 
        Lp[end] = nnzL 
        Li = zeros(Ti, nnzL) 
        Lx = zeros(Tv, nnzL) 
        D = zeros(Tv, n)
        Dinv = zeros(Tv, n)
        bwork = zeros(QDLDL_bool, n)
        iwork = zeros(Ti, 3n)
        fwork = zeros(Tv, n)
        # L = SparseMatrixCSC{Tv,Ti}(n, n, copy(Li), copy(Lp), copy(Lx))
        new{Tv,Ti}(qdldl.flags,
            qdldl.n, qdldl.Ap, qdldl.Ai, qdldl.Ax, qdldl.work, qdldl.Lnz, qdldl.etree,
            Lp, Li, Lx, D, Dinv, bwork, iwork, fwork
        )
    end
end
Base.size(F::QDLDLSolver) = (F.dims.n, F.dims.n)
Base.size(F::QDLDLSolver, i::Integer) = i <= 2 ? F.dims.n : 1 

function Base.resize!(F::QDLDLSolver, n::Integer)
    if n > F.dims.n0
        @warn "New dimension $n is greater than the buffer size $(F.dims.n0)."
        F.dims.n0 = n
    end
    resize!(F.Ap, n+1)
    resize!(F.work, n)
    resize!(F.Lnz, n)
    resize!(F.etree, n)
    resize!(F.Lp, n+1)
    resize!(F.D, n)
    resize!(F.Dinv, n)
    resize!(F.bwork, n)
    resize!(F.iwork, 3n)
    resize!(F.fwork, n)
    F.dims.n = n
    return F
end

function checkfullyinitialized(F::QDLDLSolver)
    if !isdefined(F, :Lp)
        throw(QDLDLException(
            "Cannot call factor! on an object that hasn't been initialized.\n" * 
            "You may need to create a new QDLDL object using QDLDL(F::QDLDL)."
        ))
    end
end

function eliminationtree!(F::QDLDLSolver, A::SparseMatrixCSC; check::Bool=true)
    n = size(F,1)
    if !istriu(A)
        @warn "A not upper triangular. Creating a new upper-triangular matrix."
        A = triu(A)
    end
    nnzA = nnz(A)
    if nnzA > F.dims.nnzA0
        @warn "Nonzeros buffer size not large enough for A matrix. Increasing from $(F.dims.nnzA0) to $nnzA."
        F.dims.nnzA0 = nnzA
    end
    resize!(F.Ai, nnzA)
    resize!(F.Ax, nnzA)
    F.Ap .= A.colptr .- 1
    F.Ai .= A.rowval .- 1
    F.Ax .= A.nzval
    eliminationtree!(F; check=check, newsolver=false) 
    nnzL = sum(F.Lnz)
    if nnzL > F.dims.nnzL0
        @warn "Nonzeros buffer size not large enough for L matrix. Increasing from $(F.dims.nnzL0) to $nnzL."
        F.dims.nnzL0 = nnzL
    end
    resize!(F.Li, nnzL)
    resize!(F.Lx, nnzL)
    F.Lp[end] = nnzL
    return F
end

function eliminationtree!(F::QDLDLSolver; check::Bool=true, newsolver::Bool=true) 
    out = qdldl_etree(F.dims.n, F.Ap, F.Ai, F.work, F.Lnz, F.etree)
    if check && out == -1 
        throw(QDLDLException(
            "Cannot calculate the elimination tree. Matrix is not upper triangular or has an empty column."
        ))
    elseif check && out == -2
        throw(QDLDLException(
            "Cannot calculate the elimination tree. Number of nonzero elements in L overflowed an $QDLDL_int."
        ))
    end
    F.flags.haselimtree = out >= 0 
    if out >= 0
        return newsolver ? QDLDLSolver(F) : F
    else
        return F
    end
end

function factor!(F::QDLDLSolver; check::Bool=true)
    if !F.flags.haselimtree
        throw(QDLDLException("Cannot call factor! before eliminationtree!."))
    end
    checkfullyinitialized(F)
    
    out = qdldl_factor(
        F.dims.n, F.Ap, F.Ai, F.Ax, F.Lp, F.Li, F.Lx, F.D, F.Dinv, F.Lnz, F.etree, F.bwork, 
        F.iwork, F.fwork
    )
    if check && out == -1
        throw(QDLDLException(
            "Cannot calculate factorization. Matrix not quasidefinite."
        ))
    end
    F.flags.isfactorsuccess = out != -1
    F.flags.isfactordone = true
    return F
end

function solve!(F::QDLDLSolver{T}, b::Vector{T}) where T
    if !F.flags.isfactordone
        throw(QDLDLException("Cannot call solve! before factor!."))
    end
    if F.dims.n != size(b,1)
        throw(DimensionMismatch("Cannot solve system. Expected a vector of length $(F.dims.n), got $(size(b,1))"))
    end
    checkfullyinitialized(F)
    qdldl_solve(F.dims.n, F.Lp, F.Li, F.Lx, F.Dinv, b)
    return b
end

function solveL!(F::QDLDLSolver{T}, b::Vector{T}) where T
    if !F.flags.isfactordone
        throw(QDLDLException("Cannot call solve! before factor!."))
    end
    if F.dims.n != size(b,1)
        throw(DimensionMismatch("Cannot solve system. Expected a vector of length $(F.dims.n), got $(size(b,1))"))
    end
    checkfullyinitialized(F)
    qdldl_Lsolve(F.dims.n, F.Lp, F.Li, F.Lx, b)
    return b
end

function solveLt!(F::QDLDLSolver{T}, b::Vector{T}) where T
    if !F.flags.isfactordone
        throw(QDLDLException("Cannot call solve! before factor!."))
    end
    if F.dims.n != size(b,1)
        throw(DimensionMismatch("Cannot solve system. Expected a vector of length $(F.dims.n), got $(size(b,1))"))
    end
    checkfullyinitialized(F)
    qdldl_Ltsolve(F.dims.n, F.Lp, F.Li, F.Lx, b)
    return b
end


function getL(F::QDLDLSolver{Tv,Ti}) where {Tv,Ti}
    return SparseMatrixCSC{Tv,Ti}(F.dims.n, F.dims.n, F.Lp .+ 1, F.Li .+ 1, F.Lx)
end

##############################
# Factorization Interface 
##############################
struct QDLDLFactorization{Tv <: Union{Float32, Float64}, Ti <: Integer} <: LinearAlgebra.Factorization{Tv}
    solver::QDLDLSolver{Tv,Ti}
end
QDLDLFactorization(args...) = QDLDLFactorization(QDLDLSolver(args...))
getsolver(F::QDLDLFactorization) = getfield(F, :solver)

function qdldl(A::SparseMatrixCSC; check::Bool=true)
    F = QDLDLSolver(A)
    F = eliminationtree!(F)
    if F.flags.haselimtree
        factor!(F, check=check)
    end
    return QDLDLFactorization(F)
end

function Base.getproperty(F::QDLDLFactorization, d::Symbol)
    solver = getsolver(F)
    if d == :d
        solver.D 
    elseif d == :dinv
        solver.Dinv
    elseif d == :D
        Diagonal(solver.D)
    elseif d == :Dinv
        Diagonal(solver.Dinv)
    elseif d == :L
        LowerTriangular(getL(solver))
    end
end

LinearAlgebra.issuccess(F::QDLDLFactorization) = getsolver(F).flags.isfactorsuccess

function LinearAlgebra.logabsdet(F::QDLDLFactorization{T}) where T
    return sum(log ∘ abs, F.d), prod(sign, F.d)
end

function LinearAlgebra.ldiv!(F::QDLDLFactorization, b)
    solver = getsolver(F)
    solver.flags.haselimtree || eliminationtree!(solver)
    solver.flags.isfactordone || factor!(solver)
    solve!(solver, b)
end


##############################
# C Interface Wrapper
##############################
function qdldl_etree(n::Int, Ap::Vector{Int}, Ai::Vector{Int}, work::Vector{Int}, 
    Lnz::Vector{Int}, etree::Vector{Int}
)
    @assert length(Ap) == n+1
    nnz = Ap[end]
    @assert length(Ai) == nnz
    @assert length(work) == n
    @assert length(Lnz) == n
    @assert length(etree) == n
    ccall((:QDLDL_etree, libqdldl), QDLDL_int,
        (QDLDL_int, Ptr{QDLDL_int}, Ptr{QDLDL_int}, Ref{QDLDL_int}, Ref{QDLDL_int}, 
        Ref{QDLDL_int}),
        n, Ap, Ai, work, Lnz, etree
    )
end

function qdldl_factor(
    n::QDLDL_int,
    Ap::Vector{QDLDL_int},
    Ai::Vector{QDLDL_int},
    Ax::Vector{QDLDL_float},
    Lp::Vector{QDLDL_int},
    Li::Vector{QDLDL_int},
    Lx::Vector{QDLDL_float},
    D::Vector{QDLDL_float},
    Dinv::Vector{QDLDL_float},
    Lnz::Vector{QDLDL_int},
    etree::Vector{QDLDL_int},
    bwork::Vector{QDLDL_bool},
    iwork::Vector{QDLDL_int},
    fwork::Vector{QDLDL_float}
)
    @assert length(Ap) == n+1
    nnzA = Ap[end] 
    @assert length(Ai) == nnzA
    @assert length(Ax) == nnzA
    @assert length(Lp) == n+1
    nnzL = Lp[end]
    @assert length(Li) == nnzL
    @assert length(Lx) == nnzL
    @assert length(D) == n
    @assert length(Dinv) == n
    @assert length(Lnz) == n
    @assert length(etree) == n
    @assert length(bwork) == n
    @assert length(iwork) == 3n
    @assert length(fwork) == n

    ccall((:QDLDL_factor, libqdldl), QDLDL_int,
        (
            QDLDL_int,           # n
            Ptr{QDLDL_int},      # Ap
            Ptr{QDLDL_int},      # Ai
            Ptr{QDLDL_float},    # Ax
            Ref{QDLDL_int},      # Lx
            Ref{QDLDL_int},      # Li
            Ref{QDLDL_float},    # Lx
            Ref{QDLDL_float},    # D
            Ref{QDLDL_float},    # Dinv
            Ptr{QDLDL_int},      # Lnz
            Ptr{QDLDL_int},      # etree
            Ref{QDLDL_bool},     # bwork
            Ref{QDLDL_int},      # iwork
            Ref{QDLDL_float}     # fwork
        ),
        n, Ap, Ai, Ax, Lp, Li, Lx, D, Dinv, Lnz, etree, bwork, iwork, fwork
    )
end

function qdldl_solve(
    n::QDLDL_int,
    Lp::Vector{QDLDL_int},
    Li::Vector{QDLDL_int},
    Lx::Vector{QDLDL_float},
    Dinv::Vector{QDLDL_float},
    x::Vector{QDLDL_float}
)
    nnzL = Lp[end]
    @assert length(Li) == nnzL
    @assert length(Lx) == nnzL
    @assert length(Dinv) == n
    @assert length(x) == n
    ccall((:QDLDL_solve, libqdldl), Cvoid,
        (QDLDL_int, Ptr{QDLDL_int}, Ptr{QDLDL_int}, Ptr{QDLDL_float}, Ptr{QDLDL_float}, 
        Ref{QDLDL_float}),
        n, Lp, Li, Lx, Dinv, x
    )
end

"""
    qdldl_Lsolve

Solves ``(L+I)x = b``.
"""
function qdldl_Lsolve(
    n::QDLDL_int,
    Lp::Vector{QDLDL_int},
    Li::Vector{QDLDL_int},
    Lx::Vector{QDLDL_float},
    x::Vector{QDLDL_float}
)
    nnzL = Lp[end]
    @assert length(Li) == nnzL
    @assert length(Lx) == nnzL
    @assert length(x) == n
    ccall((:QDLDL_Lsolve, libqdldl), Cvoid,
        (QDLDL_int, Ptr{QDLDL_int}, Ptr{QDLDL_int}, Ptr{QDLDL_float}, Ref{QDLDL_float}),
        n, Lp, Li, Lx, x
    )
end

"""
    qdldl_Lsolve

Solves ``(L+I)^T x = b``.
"""
function qdldl_Ltsolve(
    n::QDLDL_int,
    Lp::Vector{QDLDL_int},
    Li::Vector{QDLDL_int},
    Lx::Vector{QDLDL_float},
    x::Vector{QDLDL_float}
)
    nnzL = Lp[end]
    @assert length(Li) == nnzL
    @assert length(Lx) == nnzL
    @assert length(x) == n
    ccall((:QDLDL_Ltsolve, libqdldl), Cvoid,
        (QDLDL_int, Ptr{QDLDL_int}, Ptr{QDLDL_int}, Ptr{QDLDL_float}, Ref{QDLDL_float}),
        n, Lp, Li, Lx, x
    )
end

end