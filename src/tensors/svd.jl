using LinearAlgebra

# ==== SVD result types ========================================================

"""
    FullSVD

The result of an SVD with **no truncation** ([`NoTrunc`](@ref)).

Stores the complete factorisation ``A = U \\, \\Sigma \\, V^\\dagger`` where
no singular values were discarded. Reconstructing `A` from these three factors
is exact up to floating-point rounding, so no truncation error ``\\varepsilon``
is stored (it is identically zero).

The factors U and ``V^\\dagger`` satisfy:
- ``U^\\dagger U = \\mathbb{1}`` and ``V^\\dagger (V^\\dagger)^\\dagger = \\mathbb{1}``
  (both are *isometries*).
- When the original matrix is square, U and ``V^\\dagger`` are additionally
  *unitary*: ``U U^\\dagger = \\mathbb{1}`` and ``V^\\dagger V = \\mathbb{1}`` too.

# Fields
- `U        :: QTensor`          — left factor with legs `(original left legs..., λL)`
- `Σ        :: QTensor`          — diagonal tensor with legs `(λL, λR)`; `.data` is a `Diagonal`
- `Vd       :: QTensor`          — right factor with legs `(λR, original right legs...)`
- `spectrum :: SingValSpectrum`  — analysis view of the singular values; shares the same
  `values` vector as `Σ.data.diag` (no extra allocation)
- `center   :: BondCenter`       — bond on which the orthogonality centre lives;
  `.center.bond.left === Σ.indices[1]` and `.center.bond.right === Σ.indices[2]`

# See also
[`ReducedSVD`](@ref), [`do_svd`](@ref)
# TODO(v0.11): wire `.center` into the MPS canonical-form tracker once
# `canonicalize.jl` is routed through `do_svd` for the symmetric-tensor path.
"""
struct FullSVD
    U::QTensor
    Σ::QTensor
    Vd::QTensor
    spectrum::SingValSpectrum
    center::BondCenter
end

"""
    ReducedSVD

The result of an SVD with **truncation** ([`MaxBondDimTrunc`](@ref) or
[`ValCutoffTrunc`](@ref)).

Stores the approximate factorisation ``A \\approx U_r \\, \\Sigma_r \\, V_r^\\dagger``
where only the `r` largest singular values are kept. Reconstructing `A` from
these factors introduces an approximation error ``\\varepsilon``.

The truncated factors are *isometries*:
- ``U_r^\\dagger U_r = \\mathbb{1}_r`` — the `r × r` identity.
- ``V_r^\\dagger (V_r^\\dagger)^\\dagger = \\mathbb{1}_r``.

They are **not** unitary in general (``U_r U_r^\\dagger \\neq \\mathbb{1}``),
because columns were removed.

# Fields
- `U        :: QTensor`          — left isometry with legs `(original left legs..., λL)`
- `Σ        :: QTensor`          — diagonal tensor with legs `(λL, λR)`; `.data` is a `Diagonal`
- `Vd       :: QTensor`          — right isometry with legs `(λR, original right legs...)`
- `r        :: Int`              — number of singular values kept (the reduced bond dimension)
- `ε        :: Float64`          — 2-norm of the discarded singular values; satisfies
  ``\\|A - U_r \\Sigma_r V_r^\\dagger\\|_F = \\varepsilon``
- `spectrum :: SingValSpectrum`  — analysis view; `.spectrum.normalized` is `false` after
  truncation because discarded weight means ``\\sum_i \\sigma_i^2 < 1`` (see
  [`SingValSpectrum`](@ref) for the full explanation of the `normalized` flag)
- `center   :: BondCenter`       — bond on which the orthogonality centre lives

# See also
[`FullSVD`](@ref), [`do_svd`](@ref)
# TODO(v0.11): same as FullSVD — wire `.center` into the MPS canonical-form tracker
# once `canonicalize.jl` is routed through `do_svd`.
"""
struct ReducedSVD
    U::QTensor
    Σ::QTensor
    Vd::QTensor
    r::Int
    ε::Float64
    spectrum::SingValSpectrum
    center::BondCenter
end

# ==== Truncation strategies ===================================================

"""
    AbstractTrunc

Abstract supertype for all singular value truncation strategies.

When you perform an SVD on a tensor, you get a list of singular values
``\\sigma_1 \\geq \\sigma_2 \\geq \\dots \\geq \\sigma_k \\geq 0``.
A truncation strategy decides *how many of those to keep*.

Concrete subtypes:
- [`NoTrunc`](@ref)         — keep everything
- [`MaxBondDimTrunc`](@ref) — keep at most `max_χ` values
- [`ValCutoffTrunc`](@ref)  — keep all values above a threshold

Julia dispatches the correct `_truncate_singular_values` method
automatically based on which subtype you pass — no `if/else` needed.
"""
abstract type AbstractTrunc end

"""
    NoTrunc()

Truncation strategy that keeps **all** singular values — nothing is discarded.

This is a *singleton type*: a struct with no fields and exactly one possible
value, `NoTrunc()`. Its only purpose is to tell Julia's dispatch system "use
the no-truncation method." No data is stored because no parameter is needed.

When you use `NoTrunc`, [`do_svd`](@ref) returns a [`FullSVD`](@ref), which
guarantees that reconstructing the original tensor from the factors is exact
(up to floating-point rounding).

# Example
```julia
F = do_svd(A, bp, NoTrunc())   # F is a FullSVD
```
"""
struct NoTrunc <: AbstractTrunc end

"""
    MaxBondDimTrunc(max_χ)

Truncation strategy that keeps at most `max_χ` singular values — the `max_χ`
largest ones — and discards the rest.

In tensor network methods this is the most common truncation: `max_χ` is the
*bond dimension*, i.e. the maximum number of Schmidt values kept. Limiting it
controls memory and runtime at the cost of a controlled approximation error
``\\varepsilon``.

If the matrix has fewer than `max_χ` nonzero singular values, all of them
are kept (the bound is a cap, not a target).

# Fields
- `max_χ :: Int` — maximum number of singular values to keep (must be ≥ 1)

# Example
```julia
F = do_svd(A, bp, MaxBondDimTrunc(32))   # keep at most 32 singular values
F.ε                                        # approximation error
```
"""
struct MaxBondDimTrunc <: AbstractTrunc
    max_χ::Int
end

"""
    ValCutoffTrunc(minval)

Truncation strategy that keeps every singular value **strictly greater than**
`minval` and discards the rest.

This is useful when you care about the *physics* cutoff rather than the number
of states: singular values smaller than `minval` contribute negligibly to the
state and can be safely dropped.

# Fields
- `minval :: Float64` — absolute threshold; singular values ``\\sigma_i \\leq`` `minval` are discarded

# Example
```julia
F = do_svd(A, bp, ValCutoffTrunc(1e-10))   # discard σ ≤ 1e-10
```
"""
struct ValCutoffTrunc <: AbstractTrunc
    minval::Float64
end

# ==== Internal: decide how many values to keep ================================

"""
    _truncate_singular_values(Σ, trunc) -> (r, ε)

Internal helper: given a noise-cleaned, descending vector of singular values
and a truncation strategy, decide how many to keep.

Returns a 2-tuple `(r, ε)` where
- `r :: Int` — number of values to keep (the reduced bond dimension)
- `ε :: Float64` — 2-norm of the discarded tail:

```math
\\varepsilon = \\left\\|\\,(\\sigma_{r+1},\\, \\sigma_{r+2},\\, \\ldots)\\,\\right\\|_2
```

`ε` is identically `0.0` for [`NoTrunc`](@ref). For the other strategies it
equals the approximation error `‖A − U_r Σ_r V_r^†‖_F`.

**This function assumes the caller has already stripped numerical noise** (via
the Golub–Van Loan threshold in [`_compute_svd_factors`](@ref)).  Calling it
on a raw LAPACK output risks treating floating-point rounding as genuine
Schmidt values.

# Dispatch
- `NoTrunc`          — returns `(length(Σ), 0.0)`.  All values kept, zero error.
- `MaxBondDimTrunc`  — caps at `min(max_χ, length(Σ))`. Bond-dimension cutoff.
- `ValCutoffTrunc`   — counts values strictly above `minval`. Value-based cutoff.
"""
function _truncate_singular_values(Σ::AbstractVector{<:Real}, ::NoTrunc)
    return length(Σ), 0.0
end

function _truncate_singular_values(Σ::AbstractVector{<:Real}, trunc::MaxBondDimTrunc)
    # length(Σ) is the true numerical rank because noise was cleaned before this call
    r = min(trunc.max_χ, length(Σ))
    return r, norm(@view Σ[(r + 1):end])
end

function _truncate_singular_values(Σ::AbstractVector{<:Real}, trunc::ValCutoffTrunc)
    r = count(σ -> σ > trunc.minval, Σ)
    return r, norm(@view Σ[(r + 1):end])
end

# ==== Internal: shared SVD computation ========================================

"""
    _golub_van_loan_threshold(S::AbstractVector{<:Real}) -> Float64

Compute the classical Golub–Van Loan numerical-rank threshold for a vector of
singular values `S` (assumed sorted descending):

```math
\\tau = k \\cdot \\varepsilon_{\\mathrm{machine}} \\cdot \\sigma_1
```

where ``k = \\mathrm{length}(S)`` and ``\\sigma_1`` is the largest singular value.

Any ``\\sigma_i \\leq \\tau`` should be treated as a numerical zero rather than a
genuine Schmidt value. The rationale: LAPACK's SVD is backward stable — it computes
the exact SVD of a perturbed matrix ``A + E`` with ``\\|E\\| \\lesssim
\\varepsilon_{\\mathrm{machine}} \\cdot \\|A\\| = \\varepsilon_{\\mathrm{machine}} \\cdot
\\sigma_1``. Singular values below the noise floor ``\\tau`` are indistinguishable
from zero given this perturbation; the factor ``k`` accounts for accumulation across
``k`` floating-point operations.

Returns `0.0` for an empty `S`.

See also: [`_compute_svd_factors`](@ref)
"""
function _golub_van_loan_threshold(S::AbstractVector{<:Real})
    isempty(S) && return 0.0
    return length(S) * eps(eltype(S)) * S[1]
end

"""
    _compute_svd_factors(A, bp, trunc) -> (U_mat, svs, Vd_mat, r, ε)

Internal helper: runs the full SVD pipeline and returns raw Julia arrays.

Steps:
1. Reshape `A` into a matrix using `group_legs(A, bp)`.
2. Call LAPACK's thin SVD (`LinearAlgebra.svd`), giving a ``k \\times k``
   diagonal matrix with ``k = \\min(m, n)``.
3. Strip floating-point noise using [`_golub_van_loan_threshold`](@ref): singular
   values at or below the threshold are treated as numerical zeros, not genuine
   Schmidt values. This step happens *before* the truncation strategy sees the
   list, so strategies operate on a clean, physics-relevant spectrum.
4. Apply `_truncate_singular_values` to decide how many to keep.
5. Slice `U`, `Σ`, `Vd` to keep only the leading `r` columns/rows/values.

Returns raw `Matrix` / `Vector` values — not `QTensor`s. The leg metadata is
attached in [`_assemble_qtensors`](@ref), keeping the two concerns separate.
"""
function _compute_svd_factors(A::QTensor, bp::Bipartition, trunc::AbstractTrunc)
    M = group_legs(A, bp)
    decomp = svd(M.data)           # thin SVD: U (m×k), S (k,), Vt (k×n), k = min(m,n)

    tol = _golub_van_loan_threshold(decomp.S)
    S_cleaned = filter(σ -> σ > tol, decomp.S)

    r, ε = _truncate_singular_values(S_cleaned, trunc)

    U_mat = decomp.U[:, 1:r]
    svs = decomp.S[1:r]
    Vd_mat = decomp.Vt[1:r, :]

    return U_mat, svs, Vd_mat, r, ε
end

# ==== Internal: attach leg metadata ===========================================

"""
    _assemble_qtensors(U_mat, svs, Vd_mat, r, bp) -> (U_qt, Σ_qt, Vd_qt)

Internal helper: wraps raw factor matrices in [`QTensor`](@ref) with the
correct index legs.

Bond legs are named `:λL` (between U and Σ) and `:λR` (between Σ and Vd):
- U  gets `(original left legs..., λL::Lower)` — arrow out of U, toward ``\\Sigma``.
- Σ  gets `(λL::Upper, λR::Upper)` — ``\\Sigma`` is the orthogonality centre, so
  **both** bond arrows point into it (TNB.6: ``S^{\\lambda\\lambda'}`` carries
  two contravariant indices).
- Vd gets `(λR::Lower, original right legs...)` — arrow out of ``V^\\dagger``,
  toward ``\\Sigma``.

The `Upper`/`Lower` variance follows the package convention: bond arrows point
toward the orthogonality centre; `Upper` = incoming (domain), `Lower` = outgoing
(codomain). Each contracted pair is one `Upper` and one `Lower`.

`U_mat` and `Vd_mat` come out of the matrix SVD with the partitions fused into a
single row/column axis.  They are reshaped back into one axis per original leg —
`U` to `(left dims..., r)` and `Vd` to `(r, right dims...)` — so the factors are
genuine multi-leg tensors when a partition groups more than one leg.  This is the
exact inverse of the column-major fusion done in [`group_legs`](@ref); for the
single-leg-per-side (matrix) case the reshapes are no-ops.
"""
function _assemble_qtensors(U_mat, svs, Vd_mat, r, bp::Bipartition)
    left_dims = Tuple(dim(ix) for ix in bp.left)
    right_dims = Tuple(dim(ix) for ix in bp.right)
    U_arr = reshape(U_mat, left_dims..., r)
    Vd_arr = reshape(Vd_mat, r, right_dims...)
    U_qt = QTensor(U_arr, (bp.left..., lower(:λL, r)))
    Σ_qt = QTensor(Diagonal(svs), (upper(:λL, r), upper(:λR, r)))
    Vd_qt = QTensor(Vd_arr, (lower(:λR, r), bp.right...))
    return U_qt, Σ_qt, Vd_qt
end

# ==== Internal: build spectrum + center from assembled factors ================

"""
    _build_spectrum_and_center(svs, ε, Σ_qt) -> (SingValSpectrum, BondCenter)

Internal helper: wrap the kept singular values and the assembled ``\\Sigma``
tensor into the analysis objects that every SVD result exposes.

Two things are computed here:

1. **`normalized` flag** — checks whether ``\\sum_i \\sigma_i^2 \\approx 1``
   using a tolerance of ``\\sqrt{\\varepsilon_{\\mathrm{machine}}}``.  This is
   `true` after [`NoTrunc`](@ref) on a unit-norm state and `false` after any
   truncating strategy (because discarded singular values carried weight).  See
   [`SingValSpectrum`](@ref) for the full semantics of this flag.

2. **Bond identity** — the [`Bond`](@ref) is read directly from the legs of
   `Σ_qt`:  `Σ_qt.indices[1]` is `upper(:λL, r)` (the left face) and
   `Σ_qt.indices[2]` is `upper(:λR, r)` (the right face) — both `Upper`, since
   ``\\Sigma`` is the orthogonality centre and both bond arrows point into it.
   The resulting [`BondCenter`](@ref) therefore points to the *same* `TIx`
   objects as the ``\\Sigma`` factor — no label matching is needed downstream.

# Arguments
- `svs    :: AbstractVector{<:Real}` — kept singular values (already noise-cleaned and truncated)
- `ε      :: Float64`                — 2-norm of the discarded tail (0.0 for full SVD)
- `Σ_qt   :: QTensor`                — the assembled ``\\Sigma`` tensor; its leg indices are used to build the bond

# Returns
- `(SingValSpectrum, BondCenter)`
"""
function _build_spectrum_and_center(svs::AbstractVector{<:Real}, ε::Float64, Σ_qt::QTensor)
    normalized = isapprox(sum(abs2, svs), 1.0; atol=sqrt(eps(eltype(svs))))
    spectrum = SingValSpectrum(svs, ε, normalized)
    # Σ.indices[1] = upper(:λL, r),  Σ.indices[2] = upper(:λR, r) — both faces of the centre
    bond = Bond(Σ_qt.indices[1]::TIx{Upper}, Σ_qt.indices[2]::TIx{Upper})
    center = BondCenter(bond)
    return spectrum, center
end

# ==== Public API ==============================================================

"""
    do_svd(A, bp, trunc) -> FullSVD | ReducedSVD

Decompose the [`QTensor`](@ref) `A` into three factors ``U``, ``\\Sigma``,
``V^\\dagger`` according to the bipartition `bp` and the truncation strategy
`trunc`.

`bp` specifies how to split the legs of `A` into a "left" group (rows) and a
"right" group (columns) for the underlying matrix SVD. For example, if `A` has
legs `(σ, vL, vR)` and you want to split at the bond, pass
`Bipartition([σ, vL], [vR])`.

The return type depends on `trunc`:
- [`NoTrunc`](@ref)         → [`FullSVD`](@ref)     (exact, no error stored)
- [`MaxBondDimTrunc`](@ref) → [`ReducedSVD`](@ref)  (approximate, error in `.ε`)
- [`ValCutoffTrunc`](@ref)  → [`ReducedSVD`](@ref)  (approximate, error in `.ε`)

Both result types expose `.spectrum` ([`SingValSpectrum`](@ref)) and `.center`
([`BondCenter`](@ref)). The spectrum's `.normalized` flag is `true` after
`NoTrunc` on a unit-norm state and `false` after any truncating strategy —
see [`SingValSpectrum`](@ref) for the full semantics.

# Arguments
- `A   :: QTensor`     — the tensor to decompose
- `bp  :: Bipartition` — which legs form the rows vs columns
- `trunc :: AbstractTrunc` — how many singular values to keep

# Examples
```julia
i = upper(:i, 4); j = lower(:j, 6)
A = QTensor(randn(4, 6), (i, j))
bp = Bipartition(Partition([i]), Partition([j]))

F = do_svd(A, bp, NoTrunc())          # FullSVD, no approximation
G = do_svd(A, bp, MaxBondDimTrunc(2)) # ReducedSVD, keep 2 singular values
G.ε                                    # how much was discarded
G.spectrum.normalized                  # false — truncation lost weight
```

# See also
[`FullSVD`](@ref), [`ReducedSVD`](@ref), [`Bipartition`](@ref),
[`SingValSpectrum`](@ref)
"""
function do_svd(A::QTensor, bp::Bipartition, ::NoTrunc)
    U_mat, svs, Vd_mat, r, _ = _compute_svd_factors(A, bp, NoTrunc())
    U_qt, Σ_qt, Vd_qt = _assemble_qtensors(U_mat, svs, Vd_mat, r, bp)
    spectrum, center = _build_spectrum_and_center(svs, 0.0, Σ_qt)
    return FullSVD(U_qt, Σ_qt, Vd_qt, spectrum, center)
end

function do_svd(A::QTensor, bp::Bipartition, trunc::AbstractTrunc)
    U_mat, svs, Vd_mat, r, ε = _compute_svd_factors(A, bp, trunc)
    U_qt, Σ_qt, Vd_qt = _assemble_qtensors(U_mat, svs, Vd_mat, r, bp)
    spectrum, center = _build_spectrum_and_center(svs, ε, Σ_qt)
    return ReducedSVD(U_qt, Σ_qt, Vd_qt, r, ε, spectrum, center)
end
