# SECTION -  Spectrum hierarchy 

"""
    AbstractSpectrum

Abstract supertype for all singular- and eigenvalue spectra.

Concrete subtypes share a common set of analysis verbs:
[`schmidt_rank`](@ref), [`entanglement_entropy`](@ref),
[`entanglement_spectrum`](@ref), [`spectral_gap`](@ref).

| Subtype                   | Physical meaning                                                                |
|:------------------------- |:------------------------------------------------------------------------------- |
| [`SingValSpectrum`](@ref) | pure matrix spectrum — no location information                                  |
| [`EigValSpectrum`](@ref)  | eigenvalue spectrum — used for density matrices and ED                          |
| [`SchmidtSpectrum`](@ref) | Schmidt spectrum — wraps `SingValSpectrum` with a bipartition and bond location |
"""
abstract type AbstractSpectrum end   # root of the spectrum hierarchy; abstract type = Python ABC; no instances; subtyped by SingValSpectrum, EigValSpectrum, SchmidtSpectrum

"""
    SingValSpectrum{V}

The singular-value spectrum extracted from a matrix SVD.  This type carries
only the numerical data — it knows nothing about the physical bipartition that
produced it.  Use [`SchmidtSpectrum`](@ref) when the cut location matters.

# Fields

  - `values :: V`          — singular values ``\\sigma_1 \\geq \\sigma_2 \\geq \\cdots \\geq \\sigma_r > 0``,
    stored as a `Vector{Float64}` (already noise-cleaned and truncated).
  - `ε :: Float64`         — 2-norm of the **discarded** singular values:
    ``\\varepsilon = \\|\\sigma_{r+1}, \\sigma_{r+2}, \\ldots\\|_2``.
    Zero for a `FullSVD` (no truncation). Its square ``\\varepsilon^2`` is the **discarded
    weight** at this cut; for a unit-norm state that is the probability mass the truncation
    removed, and under real-time evolution the accumulated ``\\varepsilon^2`` across a run
    equals ``1 - \\|\\psi\\|^2``. See [`FiniteMPS`](@ref) for how these per-bond figures are
    combined (in quadrature, which is not a rigorous upper bound).
  - `normalized :: Bool`   — `true` when ``\\sum_i \\sigma_i^2 \\approx 1``.

## The `normalized` flag

`normalized` does **not** record whether the *source state* was unit-norm.
It records whether the **current values** satisfy ``\\sum_i \\sigma_i^2 \\approx 1``.

The two come apart under truncation:

| Scenario                                           | `normalized`                                                                  |
|:-------------------------------------------------- |:----------------------------------------------------------------------------- |
| `NoTrunc()` on a unit-norm state                   | `true` — no weight was lost                                                   |
| `MaxBondDimTrunc` or `ValCutoffTrunc` on any state | `false` — discarded singular values carry weight; ``\\sum_i \\sigma_i^2 < 1`` |
| Truncation followed by explicit renormalisation    | `true` — the kept values were rescaled                                        |

Callers that need a unit-norm bond spectrum (e.g. when computing overlaps via
the bond's Schmidt spectrum) should check this flag and renormalise if needed,
rather than assuming the source state was normalised.

# See also

[`SchmidtSpectrum`](@ref), [`do_svd`](@ref)   # `{V<:AbstractVector{<:Real}}` = parametric: V must be some vector of Real (allows Float64, Float32, etc.); `<: AbstractSpectrum` = subtype
"""
struct SingValSpectrum{V<:AbstractVector{<:Real}} <: AbstractSpectrum   # `{V<:AbstractVector{<:Real}}` = parametric: V must be some vector of Real (allows Float64, Float32, etc.); `<: AbstractSpectrum` = subtype
    values::V        # kept singular values σ₁ ≥ σ₂ ≥ ... > 0; noise-cleaned and truncated before storage; type V is inferred at construction
    ε::Float64       # 2-norm of the discarded singular values = approximation error ‖A - U_r Σ_r Vd_r‖_F; 0.0 for full (non-truncated) SVD
    normalized::Bool # true iff sum(σᵢ²) ≈ 1; physics: a unit-norm state has ∑σᵢ² = 1 but after truncation ∑σᵢ² < 1 because discarded values carry weight
end

"""
    EigValSpectrum{V}

The eigenvalue spectrum of a Hermitian operator (density matrix, Hamiltonian,
transfer matrix).  Unlike [`SingValSpectrum`](@ref), eigenvalues may be
negative, so no `normalized` flag or truncation error ``\\varepsilon`` is stored.

# Fields

  - `values :: V` — eigenvalues in descending order.
"""
struct EigValSpectrum{V<:AbstractVector} <: AbstractSpectrum   # `{V<:AbstractVector}` = any vector (not restricted to Real since eigenvalues can be complex for non-Hermitian ops); `<: AbstractSpectrum` = subtype
    values::V   # eigenvalues sorted descending; for a Hamiltonian: energies E₁ ≤ E₂ ≤ ... (ascending by convention); for a density matrix: probabilities pᵢ ≥ 0
end

"""
    SchmidtSpectrum{V}

A [`SingValSpectrum`](@ref) enriched with the **physical location** of the
bipartition: which legs were split and where the orthogonality centre lives.

```math
|\\psi\\rangle = \\sum_{i=1}^r \\sigma_i \\, |i\\rangle_A \\otimes |i\\rangle_B,
\\qquad \\sigma_1 \\geq \\sigma_2 \\geq \\cdots \\geq \\sigma_r > 0
```

The Schmidt values ``\\sigma_i`` are the singular values of the state tensor
reshaped as a matrix by the bipartition.  The entanglement entropy is derived
from them — it is **not** stored:

```math
S_b = -\\sum_{i=1}^r \\sigma_i^2 \\log_b \\sigma_i^2, \\qquad 0 \\cdot \\log_b 0 := 0
```

# Fields

  - `spectrum :: SingValSpectrum{V}` — the underlying numerical spectrum
  - `cut      :: Bipartition`        — which legs form the left/right subsystems
  - `center   :: BondCenter`         — the bond on which the Schmidt decomposition lives;
    `.center.bond` legs are the **same** `TIx` objects as the ``\\Sigma`` factor's legs —
    no label matching is needed.

# See also

[`SingValSpectrum`](@ref), [`entanglement_entropy`](@ref),
[`entanglement_spectrum`](@ref), [`schmidt_rank`](@ref)   # wraps SingValSpectrum with location metadata; `{V}` propagates the element-type parameter from the inner spectrum
"""
struct SchmidtSpectrum{V<:AbstractVector{<:Real}} <: AbstractSpectrum   # wraps SingValSpectrum with location metadata; `{V}` propagates the element-type parameter from the inner spectrum
    spectrum::SingValSpectrum{V}   # the underlying numerical singular-value data; `.spectrum.values` = the σᵢ vector
    cut::Bipartition               # which legs were put left (rows) vs right (columns) for the Schmidt decomposition
    center::BondCenter             # which bond the gauge centre Σ sits on; `.center.bond.left/right` are the exact TIx objects from the SVD Σ factor
end

# SECTION -  Spectrum verbs 

"""
    Base.length(s::SingValSpectrum) -> Int
    Base.length(s::SchmidtSpectrum) -> Int

Number of singular values in the spectrum (= Schmidt rank for a Schmidt spectrum).
"""
Base.length(s::SingValSpectrum) = length(s.values)                  # `Base.length` = extend Julia's built-in `length`. delegates to the values vector
Base.length(s::SchmidtSpectrum) = length(s.spectrum.values)         # unwrap the inner SingValSpectrum; `.spectrum.values` = chain of field accesses (Python: `s.spectrum.values`)

"""
    schmidt_rank(s::AbstractSpectrum) -> Int

Number of kept singular values.  Equal to `length(s)`.
For a [`SchmidtSpectrum`](@ref) this is the Schmidt rank of the cut.
"""
schmidt_rank(s::AbstractSpectrum) = length(s)   # physics: Schmidt rank = number of nonzero σᵢ = number of terms in |ψ⟩ = Σ σᵢ |i⟩_A|i⟩_B; dispatches to `Base.length` above

"""
    spectral_gap(s::SchmidtSpectrum) -> Float64

Difference between the two largest Schmidt values: ``\\sigma_1 - \\sigma_2``.
Returns ``\\sigma_1`` when the Schmidt rank is 1 (product state — the gap is
effectively infinite, bounded here by the largest value).

A large gap means the state is well-approximated by a rank-1 product state;
a small gap signals entanglement that is hard to truncate.
"""
function spectral_gap(s::SchmidtSpectrum)
    vals = s.spectrum.values   # access the underlying vector of σᵢ; field chain: SchmidtSpectrum → SingValSpectrum → values
    return length(vals) >= 2 ? vals[1] - vals[2] : vals[1]   # ternary: `cond ? a : b`. if rank ≥ 2 return σ₁-σ₂; if rank=1 return σ₁ (product state, gap is infinite but we return the only value)
end

"""
    bipartition(s::SchmidtSpectrum) -> Bipartition

The bipartition that produced this Schmidt spectrum.
Calling this on a bare [`SingValSpectrum`](@ref) raises a `MethodError` — pure
matrix spectra carry no location information.
"""
bipartition(s::SchmidtSpectrum) = s.cut   # accessor: returns the stored Bipartition; intentionally NOT defined for SingValSpectrum — calling on bare SingValSpectrum raises MethodError (type guard)

"""
    center(s::SchmidtSpectrum) -> BondCenter

The bond on which the orthogonality centre lives for this Schmidt spectrum.
"""
center(s::SchmidtSpectrum) = s.center   # accessor: returns the BondCenter; `.center.bond.left/right` are the same TIx objects as the SVD Σ factor's legs

"""
    entanglement_entropy(s::SchmidtSpectrum; base=2) -> Float64

Von Neumann entanglement entropy of the bipartition encoded in `s`:

```math
S_b(\\rho_A) = -\\sum_{i=1}^r \\sigma_i^2 \\log_b \\sigma_i^2
```

where ``\\sigma_i`` are the Schmidt values and ``b`` is the logarithm base.
The convention ``0 \\cdot \\log_b 0 := 0`` is enforced, so zero singular values
(from rank-deficient states or noise-cleaned tails) never produce `NaN`.

Default `base=2` returns the entropy in **bits**.  Pass `base=ℯ` for nats.
"""
function entanglement_entropy(s::SchmidtSpectrum; base=2)   # `; base=2` = keyword argument with default value 2 (base-2 = bits; use base=ℯ for nats); Python: `def f(s, *, base=2)`
    p = abs2.(s.spectrum.values)   # `abs2.(v)` = element-wise |x|² ; pᵢ = σᵢ² = eigenvalues of ρ_A; `.` before `(` = broadcasting over the vector
    # Normalise so that Σpᵢ = 1 before computing −Σ pᵢ log pᵢ.  Without this
    # the result is wrong when the spectrum comes from a truncated or non-canonical
    # state (Σσᵢ² < 1).  The design plan (Part 1 line 529) notes that the entropy
    # is only "free" when the canonical centre is at this bond; normalising here
    # makes the function safe to call regardless of the gauge.  Fixes #80.
    p ./= sum(p)   # `./=` = in-place broadcast division ; normalise probabilities so they sum to 1 even after truncation
    return -sum(pᵢ -> pᵢ > 0 ? pᵢ * log(base, pᵢ) : 0.0, p)   # `pᵢ -> ...` = anonymous function. `? :` ternary; `log(base, x)` = log_base(x); 0·log(0) = 0.0 guard avoids NaN; `sum(f, iter)` = Python `sum(f(x) for x in iter)`
end

"""
    entanglement_spectrum(s::SchmidtSpectrum) -> Vector{Float64}

The entanglement spectrum ``\\{\\varepsilon_i\\}`` defined as

```math
\\varepsilon_i = -2 \\ln \\sigma_i
```

where ``\\sigma_i`` are the Schmidt values (natural logarithm, base ``e``).
For a Bell pair with ``\\sigma_1 = \\sigma_2 = 1/\\sqrt{2}`` this gives
``\\varepsilon_i = -2\\ln(1/\\sqrt{2}) = \\ln 2`` for both levels.

The entanglement spectrum (a list of "energies") carries more information than
the scalar entropy: it reveals the level structure of the reduced density
matrix and is used to diagnose topological order.
"""
entanglement_spectrum(s::SchmidtSpectrum) = -2.0 .* log.(s.spectrum.values)   # `.` = element-wise: `log.(v)` = Python `np.log(v)` (natural log); `-2.0 .* ...` = broadcast multiply by -2; physics: εᵢ = -2 ln(σᵢ) are "energies" of the entanglement Hamiltonian H_E = -2 log(ρ_A)

"""
    schmidt_values(s::SchmidtSpectrum) -> AbstractVector{<:Real}

The Schmidt values ``\\sigma_1 \\geq \\sigma_2 \\geq \\cdots \\geq \\sigma_r > 0``
of the bipartition encoded in `s`.

Returns the `values` vector of the inner [`SingValSpectrum`](@ref) directly
(no copy). For a normalized state ``\\sum_i \\sigma_i^2 = 1``.

# See also

[`SchmidtSpectrum`](@ref), [`entanglement_entropy`](@ref), [`schmidt_rank`](@ref)
"""
schmidt_values(s::SchmidtSpectrum) = s.spectrum.values   # direct field access — no copy; returns a reference to the underlying σᵢ vector; physics: the Schmidt values satisfy |ψ⟩ = Σᵢ σᵢ |i⟩_A|i⟩_B with ∑σᵢ² = 1 for a normalised state
