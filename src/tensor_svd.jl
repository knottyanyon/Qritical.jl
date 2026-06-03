using LinearAlgebra: svd, norm, Diagonal, diag

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

# ── Bond ──────────────────────────────────────────────────────────────────────

"""
    Bond(lower, upper[, trunc, ε])

Records a contraction between two tensors: `lower::TIx{Lower}` on the
from-tensor and `upper::TIx{Upper}` on the to-tensor, both carrying the same
label and dimension.

When the bond is produced by an SVD, the optional fields `trunc` and `ε`
store the truncation strategy and the 2-norm truncation error respectively.
"""
struct Bond
    lower :: TIx{Lower}
    upper :: TIx{Upper}
    trunc :: Union{AbstractTruncation, Nothing}
    ε     :: Float64

    function Bond(
        lower::TIx{Lower}, upper::TIx{Upper},
        trunc::Union{AbstractTruncation, Nothing}=nothing,
        ε::Real=0.0,
    )
        lower.label == upper.label || throw(ArgumentError(
            "Bond legs must share a label; got $(lower.label) and $(upper.label)"
        ))
        lower.ndim == upper.ndim || throw(ArgumentError(
            "Bond legs must have matching ndim; got $(lower.ndim) and $(upper.ndim)"
        ))
        new(lower, upper, trunc, Float64(ε))
    end
end

label(b::Bond) = b.lower.label
ndim(b::Bond)  = b.lower.ndim

# ── tensor_svd ───────────────────────────────────────────────────────────────

"""
    tensor_svd(A, bp, trunc; normalize=false) -> (; U, Σ, Vd, ε)

Decompose `A` as ``A^i_{\\ j} = U^i_{\\ \\lambda}\\, \\Sigma^\\lambda_{\\ \\lambda'}\\,
{(V^\\dagger)}^{\\lambda'}_{\\ j}`` according to the bipartition `bp` and the
truncation strategy `trunc`.

Bond labels are derived from the `MultiIx` autolabels of the bipartition by
prefixing with `:χ`: Bond₁ ``(U \\leftrightarrow \\Sigma)`` gets label
``\\texttt{Symbol}(:\\chi, \\ell_L)`` and Bond₂ ``(\\Sigma \\leftrightarrow V^\\dagger)``
gets ``\\texttt{Symbol}(:\\chi, \\ell_R)``, where ``\\ell_L, \\ell_R`` are the
autolabels of the left and right grouped indices.

- `U`   carries the original left-partition indices followed by a `TIx{Lower}`
  bond leg (outward, label ``\\chi\\ell_L``).
- `Σ`   is an `IndexedTensor{T,2,Diagonal}` with a `TIx{Upper}` leg (label
  ``\\chi\\ell_L``) and a `TIx{Lower}` leg (label ``\\chi\\ell_R``).
- `Vd`  carries a `TIx{Upper}` bond leg (label ``\\chi\\ell_R``) followed by the
  original right-partition indices.
- `ε`   is the 2-norm of the discarded singular values (exact truncation error).

## Normalization

Pass `normalize=true` to obtain Schmidt coefficients ``\\lambda_i = \\sigma_i / \\|A\\|_F``.
The diagonal of `Σ.data` then satisfies ``\\sum_i \\lambda_i^2 \\leq 1``.  `ε` is
always returned in raw form; ``\\|A\\|_F^2 = \\operatorname{tr}(\\Sigma^2) + \\varepsilon^2``.

!!! note
    For MPS canonical sweeps absorb `Σ` into a neighbouring site tensor and
    use `normalize=false`.  For Schmidt decompositions use `normalize=true`.
"""
function tensor_svd(
    A::IndexedTensor, bp::Bipartition, trunc::AbstractTruncation;
    normalize::Bool=false,
)
    M = group_legs(A, bp)
    F = svd(M.data)

    norm_full = norm(F.S)
    r, ε      = _truncate(F.S, trunc)

    # Derive bond labels from the grouped MultiIx autolabels, prefixed with :χ
    # so they never clash with original partition index labels.
    λ_label  = Symbol(:χ, label(M.indices[1]))   # Bond₁: U ↔ Σ
    λ′_label = Symbol(:χ, label(M.indices[2]))   # Bond₂: Σ ↔ Vd

    left_shape  = Tuple(ndim(idx) for idx in bp.left.indices)
    right_shape = Tuple(ndim(idx) for idx in bp.right.indices)

    svs    = normalize ? F.S[1:r] ./ norm_full : F.S[1:r]
    U_data = reshape(F.U[:, 1:r],   left_shape..., r)
    Σ_data = Diagonal(svs)
    Vd_data = reshape(F.Vt[1:r, :], r, right_shape...)

    U_indices  = Tuple(AbstractIndex[bp.left.indices...;  lower(λ_label,  r)])
    Σ_indices  = (upper(λ_label, r), lower(λ′_label, r))
    Vd_indices = Tuple(AbstractIndex[upper(λ′_label, r); bp.right.indices...])

    return (;
        U  = IndexedTensor(U_data,  U_indices),
        Σ  = IndexedTensor(Σ_data,  Σ_indices),
        Vd = IndexedTensor(Vd_data, Vd_indices),
        ε,
    )
end
