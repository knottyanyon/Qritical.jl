# ==== Bond ====================================================================

"""
    Bond

Pure geometry of one link in a tensor network: the pair of legs (faces) that
sit on either side of a shared index.

```math
\\cdots \\underbrace{-[\\,U\\,]}_{}\\!\\underbrace{-\\!\\overbrace{|\\lambda_L\\rangle\\langle\\lambda_R|}^{\\text{Bond}}\\!-}_{}[\\,V^\\dagger\\,]-\\cdots
```

A `Bond` carries no numerical data — it only records which `TIx` objects name
the two faces of the link:

- `.lower :: TIx{Lower}` — the **outgoing** (codomain) face; sits on the right
  leg of the left tensor (e.g. the ``\\lambda_R`` leg of ``\\Sigma``).
- `.upper :: TIx{Upper}` — the **incoming** (domain) face; sits on the left leg
  of the right tensor (e.g. the ``\\lambda_L`` leg of ``\\Sigma``).

The naming matches the package convention: `Lower` = codomain = outgoing,
`Upper` = domain = incoming (§13, §23 of MasterPlan).

# Construction

After `do_svd`, the bond is derived directly from the ``\\Sigma`` factor:

```julia
F = do_svd(A, bp, NoTrunc())
F.center.bond.lower === F.Σ.indices[2]  # lower(:λR, r)
F.center.bond.upper === F.Σ.indices[1]  # upper(:λL, r)
```

No label matching is needed — the legs are the *same* `TIx` objects.

# See also
[`BondCenter`](@ref), [`OrthoCenter`](@ref)
"""
struct Bond
    lower::TIx{Lower}
    upper::TIx{Upper}
end

# ==== Orthogonality-centre hierarchy ==========================================

"""
    OrthoCenter

Abstract supertype for the **orthogonality centre** of an MPS: the unique
tensor (or bond) for which all tensors to the left are left-orthogonal and all
tensors to the right are right-orthogonal.

Concrete subtypes:
- [`BondCenter`](@ref) — centre sits on a bond (both faces of one link).
- [`SiteCenter`](@ref) — centre sits on a single physical site tensor.

The subtype encodes whether a canonical-form conversion needs to swap the
arrow direction on a bond (`BondCenter`) or absorb singular values into a
site tensor (`SiteCenter`). Dispatch on `OrthoCenter` is exhaustive over
these two cases.
"""
abstract type OrthoCenter end

"""
    BondCenter(bond)

Orthogonality centre located on a **bond**: both faces of one link carry the
gauge freedom.  Conversion to left-canonical shifts the centre rightward
(absorbing ``\\Sigma`` into ``V^\\dagger``); conversion to right-canonical shifts
it leftward (absorbing ``\\Sigma`` into ``U``).

# Fields
- `bond :: Bond` — the link on which the centre sits; `.bond.lower` is the
  right leg of ``U`` and `.bond.upper` is the left leg of ``V^\\dagger``.

# See also
[`SiteCenter`](@ref), [`Bond`](@ref)
"""
struct BondCenter <: OrthoCenter
    bond::Bond
end

"""
    SiteCenter(leg)

Orthogonality centre located on a **site**: the centre tensor carries a
physical leg and is neither left- nor right-orthogonal.  This form arises
after `move_center!` sweeps the gauge to a specific site.

# Fields
- `leg :: TIx{Lower}` — the physical leg of the site tensor that is the
  current centre.

# See also
[`BondCenter`](@ref), [`OrthoCenter`](@ref)
"""
struct SiteCenter <: OrthoCenter
    leg::TIx{Lower}
end

# ==== Spectrum hierarchy ======================================================

"""
    AbstractSpectrum

Abstract supertype for all singular- and eigenvalue spectra.

Concrete subtypes share a common set of analysis verbs:
[`schmidt_rank`](@ref), [`entanglement_entropy`](@ref),
[`entanglement_spectrum`](@ref), [`spectral_gap`](@ref).

| Subtype | Physical meaning |
|---------|-----------------|
| [`SingValSpectrum`](@ref) | pure matrix spectrum — no location information |
| [`EigValSpectrum`](@ref)  | eigenvalue spectrum — used for density matrices and ED |
| [`SchmidtSpectrum`](@ref) | Schmidt spectrum — wraps `SingValSpectrum` with a bipartition and bond location |
"""
abstract type AbstractSpectrum end

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
  Zero for a `FullSVD` (no truncation).
- `normalized :: Bool`   — `true` when ``\\sum_i \\sigma_i^2 \\approx 1``.

## The `normalized` flag

`normalized` does **not** record whether the *source state* was unit-norm.
It records whether the **current values** satisfy ``\\sum_i \\sigma_i^2 \\approx 1``.

The two come apart under truncation:

| Scenario | `normalized` |
|----------|-------------|
| `NoTrunc()` on a unit-norm state | `true` — no weight was lost |
| `MaxBondDimTrunc` or `ValCutoffTrunc` on any state | `false` — discarded singular values carry weight; ``\\sum_i \\sigma_i^2 < 1`` |
| Truncation followed by explicit renormalisation | `true` — the kept values were rescaled |

Callers that need a unit-norm bond spectrum (e.g. when computing overlaps via
the bond's Schmidt spectrum) should check this flag and renormalise if needed,
rather than assuming the source state was normalised.

# See also
[`SchmidtSpectrum`](@ref), [`do_svd`](@ref)
"""
struct SingValSpectrum{V<:AbstractVector{<:Real}} <: AbstractSpectrum
    values::V
    ε::Float64
    normalized::Bool
end

"""
    EigValSpectrum{V}

The eigenvalue spectrum of a Hermitian operator (density matrix, Hamiltonian,
transfer matrix).  Unlike [`SingValSpectrum`](@ref), eigenvalues may be
negative, so no `normalized` flag or truncation error ``\\varepsilon`` is stored.

First consumer: Week-10 exact diagonalisation (§17 of MasterPlan).

# Fields
- `values :: V` — eigenvalues in descending order.
"""
struct EigValSpectrum{V<:AbstractVector} <: AbstractSpectrum
    values::V
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
[`entanglement_spectrum`](@ref), [`schmidt_rank`](@ref)
"""
struct SchmidtSpectrum{V<:AbstractVector{<:Real}} <: AbstractSpectrum
    spectrum::SingValSpectrum{V}
    cut::Bipartition
    center::BondCenter
end

# ==== Spectrum verbs ==========================================================

"""
    Base.length(s::SingValSpectrum) -> Int
    Base.length(s::SchmidtSpectrum) -> Int

Number of singular values in the spectrum (= Schmidt rank for a Schmidt spectrum).
"""
Base.length(s::SingValSpectrum) = length(s.values)
Base.length(s::SchmidtSpectrum) = length(s.spectrum.values)

"""
    schmidt_rank(s::AbstractSpectrum) -> Int

Number of kept singular values.  Equal to `length(s)`.
For a [`SchmidtSpectrum`](@ref) this is the Schmidt rank of the cut.
"""
schmidt_rank(s::AbstractSpectrum) = length(s)

"""
    spectral_gap(s::SchmidtSpectrum) -> Float64

Difference between the two largest Schmidt values: ``\\sigma_1 - \\sigma_2``.
Returns ``\\sigma_1`` when the Schmidt rank is 1 (product state — the gap is
effectively infinite, bounded here by the largest value).

A large gap means the state is well-approximated by a rank-1 product state;
a small gap signals entanglement that is hard to truncate.
"""
function spectral_gap(s::SchmidtSpectrum)
    vals = s.spectrum.values
    return length(vals) >= 2 ? vals[1] - vals[2] : vals[1]
end

"""
    bipartition(s::SchmidtSpectrum) -> Bipartition

The bipartition that produced this Schmidt spectrum.
Calling this on a bare [`SingValSpectrum`](@ref) raises a `MethodError` — pure
matrix spectra carry no location information.
"""
bipartition(s::SchmidtSpectrum) = s.cut

"""
    center(s::SchmidtSpectrum) -> BondCenter

The bond on which the orthogonality centre lives for this Schmidt spectrum.
"""
center(s::SchmidtSpectrum) = s.center

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
function entanglement_entropy(s::SchmidtSpectrum; base=2)
    p = abs2.(s.spectrum.values)   # pᵢ = σᵢ²  (eigenvalues of ρ_A)
    return -sum(pᵢ -> pᵢ > 0 ? pᵢ * log(base, pᵢ) : 0.0, p)
end

"""
    entanglement_spectrum(s::SchmidtSpectrum) -> Vector{Float64}

The entanglement spectrum ``\\{\\varepsilon_i\\}`` defined as

```math
\\varepsilon_i = -2 \\ln \\sigma_i
```

where ``\\sigma_i`` are the Schmidt values (natural logarithm, base ``e``).
For a Bell pair with ``\\sigma_1 = \\sigma_2 = 1/\\sqrt{2}`` this gives
``\\varepsilon_i = \\ln 4`` for both levels.

The entanglement spectrum (a list of "energies") carries more information than
the scalar entropy: it reveals the level structure of the reduced density
matrix and is used to diagnose topological order.
"""
entanglement_spectrum(s::SchmidtSpectrum) = -2.0 .* log.(s.spectrum.values)

"""
    schmidt_values(s::SchmidtSpectrum) -> AbstractVector{<:Real}

The Schmidt values ``\\sigma_1 \\geq \\sigma_2 \\geq \\cdots \\geq \\sigma_r > 0``
of the bipartition encoded in `s`.

Returns the `values` vector of the inner [`SingValSpectrum`](@ref) directly
(no copy). For a normalized state ``\\sum_i \\sigma_i^2 = 1``.

# See also
[`SchmidtSpectrum`](@ref), [`entanglement_entropy`](@ref), [`schmidt_rank`](@ref)
"""
schmidt_values(s::SchmidtSpectrum) = s.spectrum.values
