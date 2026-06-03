using LinearAlgebra: svd, norm

# ── BondIndex ─────────────────────────────────────────────────────────────────

"""
    BondIndex(label, ndim)

An `AbstractIndex` produced by SVD. Represents a virtual bond whose dimension
equals the number of retained singular values.
"""
struct BondIndex <: AbstractIndex
    label::Symbol
    ndim::Int
    function BondIndex(label::Symbol, ndim::Int)
        ndim > 0 || throw(ArgumentError("BondIndex ndim must be positive, got $ndim"))
        new(label, ndim)
    end
end

ndim(b::BondIndex)  = b.ndim
label(b::BondIndex) = b.label

Base.:(==)(a::BondIndex, b::BondIndex) = a.label == b.label && a.ndim == b.ndim
Base.hash(b::BondIndex, h::UInt)       = hash(b.label, hash(b.ndim, h))

# ── AbstractTruncation ────────────────────────────────────────────────────────

abstract type AbstractTruncation end

"""Keep the `r` largest singular values; automatically caps at the numerical rank."""
struct KeepFirst <: AbstractTruncation
    r::Int
end

"""Keep all singular values strictly greater than `atol`."""
struct KeepAbove <: AbstractTruncation
    atol::Float64
end

"""Keep all singular values where `σ / σ_max > rtol`."""
struct KeepRelative <: AbstractTruncation
    rtol::Float64
end

# ── Internal truncation helpers ───────────────────────────────────────────────
# Each returns (r, ε): singular values to keep and the 2-norm truncation error.

function _truncate(S::AbstractVector{<:Real}, trunc::KeepFirst)
    # Use the same tolerance as LinearAlgebra.rank to skip numerical zeros.
    tol    = length(S) * eps(eltype(S)) * (isempty(S) ? one(eltype(S)) : S[1])
    nzrank = count(s -> s > tol, S)
    r      = min(trunc.r, nzrank)
    return r, norm(@view S[(r+1):end])
end

function _truncate(S::AbstractVector{<:Real}, trunc::KeepAbove)
    r = count(s -> s > trunc.atol, S)
    return r, norm(@view S[(r+1):end])
end

function _truncate(S::AbstractVector{<:Real}, trunc::KeepRelative)
    σ_max = isempty(S) ? one(eltype(S)) : S[1]
    r     = count(s -> s / σ_max > trunc.rtol, S)
    return r, norm(@view S[(r+1):end])
end

# ── tensor_svd ───────────────────────────────────────────────────────────────

"""
    tensor_svd(A, bp, trunc) -> (; U, S, Vd, ε)

Decompose `A` into `U * Diagonal(S) * Vd` according to the bipartition `bp`
and the truncation strategy `trunc`.

- `U`  carries the original left-partition indices followed by a `BondIndex`.
- `Vd` carries a `BondIndex` followed by the original right-partition indices.
- `S`  is a `Vector{<:Real}` of retained singular values, sorted descending.
- `ε`  is the 2-norm of the discarded singular values (exact truncation error).
"""
function tensor_svd(A::IndexedTensor, bp::Bipartition, trunc::AbstractTruncation)
    M = group_legs(A, bp)
    F = svd(M.data)

    r, ε = _truncate(F.S, trunc)

    bond = BondIndex(:χ, r)

    left_shape  = Tuple(ndim(idx) for idx in bp.left.indices)
    right_shape = Tuple(ndim(idx) for idx in bp.right.indices)

    U_data  = reshape(F.U[:, 1:r],    left_shape...,  r)
    Vd_data = reshape(F.Vt[1:r, :],  r, right_shape...)

    U_indices  = Tuple(AbstractIndex[bp.left.indices...;  bond])
    Vd_indices = Tuple(AbstractIndex[bond; bp.right.indices...])

    return (;
        U  = IndexedTensor(U_data,  U_indices),
        S  = F.S[1:r],
        Vd = IndexedTensor(Vd_data, Vd_indices),
        ε,
    )
end
