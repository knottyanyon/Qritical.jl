using LinearAlgebra: svd, norm

# ── BondIndex ─────────────────────────────────────────────────────────────────

"""
    BondIndex(label, ndim)

An `AbstractIndex` produced by SVD. Represents a virtual bond whose dimension
equals the Schmidt rank ``\\operatorname{Sch}(\\psi)`` of the decomposed state
after applying the chosen truncation strategy.
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

"""
    KeepFirst(r)

Truncation strategy that sets ``\\operatorname{Sch}(\\psi) = \\min(r, \\operatorname{rank}(A))``:
keep the `r` largest singular values, capped at the numerical rank of `A`.
"""
struct KeepFirst <: AbstractTruncation
    r::Int
end

"""
    KeepAbove(atol)

Truncation strategy that sets
``\\operatorname{Sch}(\\psi) = |\\{\\, i : \\sigma_i > \\texttt{atol} \\,\\}|``:
keep all singular values strictly greater than the absolute threshold `atol`.
"""
struct KeepAbove <: AbstractTruncation
    atol::Float64
end

"""
    KeepRelative(rtol)

Truncation strategy that sets
``\\operatorname{Sch}(\\psi) = |\\{\\, i : \\sigma_i / \\sigma_1 > \\texttt{rtol} \\,\\}|``:
keep all singular values whose ratio to the largest singular value ``\\sigma_1``
exceeds the relative threshold `rtol`.
"""
struct KeepRelative <: AbstractTruncation
    rtol::Float64
end

"""
    KeepMachineEps()

Truncation strategy that sets
``\\operatorname{Sch}(\\psi) = |\\{\\, i : \\sigma_i / \\sigma_1 > \\sqrt{\\varepsilon_T} \\,\\}|``,
where ``\\varepsilon_T = \\texttt{eps}(T)`` is the machine epsilon of the floating-point
type `T` used in the SVD.

The threshold ``\\sqrt{\\varepsilon_T}`` is the classical numerical-rank criterion
from Golub & Van Loan: singular values smaller than this relative to ``\\sigma_1``
are indistinguishable from zero given the backward error of the SVD algorithm.
Unlike `KeepRelative`, no threshold parameter is required — the cutoff
self-calibrates to the precision of the computation.

Concrete thresholds:
- `Float64`: ``\\sqrt{\\varepsilon} \\approx 1.5 \\times 10^{-8}``
- `Float32`: ``\\sqrt{\\varepsilon} \\approx 3.5 \\times 10^{-4}``
"""
struct KeepMachineEps <: AbstractTruncation end

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

function _truncate(S::AbstractVector{<:Real}, ::KeepMachineEps)
    T     = eltype(S)
    σ_max = isempty(S) ? one(T) : S[1]
    tol   = sqrt(eps(T)) * σ_max
    r     = count(s -> s > tol, S)
    return r, norm(@view S[(r+1):end])
end

# ── tensor_svd ───────────────────────────────────────────────────────────────

"""
    tensor_svd(A, bp, trunc; normalize=false) -> (; U, S, Vd, ε)

Decompose `A` into `U * Diagonal(S) * Vd` according to the bipartition `bp`
and the truncation strategy `trunc`.

- `U`  carries the original left-partition indices followed by a `BondIndex` of
  dimension ``\\operatorname{Sch}(\\psi)``.
- `Vd` carries a `BondIndex` of dimension ``\\operatorname{Sch}(\\psi)`` followed
  by the original right-partition indices.
- `S`  is a `Vector{<:Real}` of ``\\operatorname{Sch}(\\psi)`` retained singular
  values, sorted descending.
- `ε`  is the 2-norm of the discarded singular values (exact truncation error).

## Normalization

By default (`normalize=false`) `S` contains the raw singular values of `A`.
Pass `normalize=true` to obtain proper Schmidt coefficients:

```math
\\lambda_i = \\frac{\\sigma_i}{\\|A\\|_F}
```

where ``\\|A\\|_F = \\sqrt{\\sum_i \\sigma_i^2}`` is the Frobenius norm computed
from the full pre-truncation spectrum. The normalized `S` satisfies
``\\sum_i \\lambda_i^2 \\leq 1``, with equality iff no singular values were
discarded. `ε` is always returned in raw (unnormalized) form regardless of
this flag; the full norm satisfies ``\\|A\\|_F^2 = \\texttt{norm(S)}^2 + \\varepsilon^2``.

!!! note
    For MPS canonical-form sweeps the raw singular values are typically
    absorbed into a neighbouring site tensor; `normalize=false` is the
    right choice there. For Schmidt decompositions and entanglement entropy
    calculations, use `normalize=true`.
"""
function tensor_svd(
    A::IndexedTensor, bp::Bipartition, trunc::AbstractTruncation;
    normalize::Bool=false,
)
    M = group_legs(A, bp)
    F = svd(M.data)

    norm_full = norm(F.S)
    r, ε      = _truncate(F.S, trunc)

    bond = BondIndex(:χ, r)

    left_shape  = Tuple(ndim(idx) for idx in bp.left.indices)
    right_shape = Tuple(ndim(idx) for idx in bp.right.indices)

    U_data  = reshape(F.U[:, 1:r],   left_shape...,  r)
    Vd_data = reshape(F.Vt[1:r, :], r, right_shape...)

    U_indices  = Tuple(AbstractIndex[bp.left.indices...;  bond])
    Vd_indices = Tuple(AbstractIndex[bond; bp.right.indices...])

    S_out = normalize ? F.S[1:r] ./ norm_full : F.S[1:r]

    if !normalize
        @debug "tensor_svd: S contains raw (unnormalized) singular values. " *
               "Pass `normalize=true` to obtain Schmidt coefficients."
    end

    return (;
        U  = IndexedTensor(U_data,  U_indices),
        S  = S_out,
        Vd = IndexedTensor(Vd_data, Vd_indices),
        ε,
    )
end
