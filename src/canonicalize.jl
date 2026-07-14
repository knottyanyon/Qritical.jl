# ==== Sweep helpers on existing MPS ==========================================
#
# Unlike _left_sweep / _right_sweep (which start from a dense d^L tensor),
# these helpers operate directly on the L order-3 site tensors of an existing
# MPS by propagating a "carry" factor between neighbours:
#
#   Left sweep at site i:  reshape A_i → (χL·d, χR), SVD, store U as the new
#   left-canonical A_i, absorb Σ·Vd into A_{i+1} via a matrix multiply.
#
#   Right sweep at site i: reshape B_i → (χL, d·χR), SVD, store Vd as the new
#   right-canonical B_i, absorb U·Σ into B_{i-1}.
#
#   Per-bond cost: O(χ²·d) — linear in L, not exponential in d.  This is the
#   fundamental efficiency of MPS methods.
#
# Each helper mutates the tensor/bond_svs vectors in place and returns the
# accumulated truncation error ε.
#
# Trade-off note (§3.2):
#   canonical_error (public API) tests only LEFT isometry (‖A†A − I‖).
#   Right-isometry checks (‖BB† − I‖) live in _right_isometry_error_mps,
#   used internally by is_canonical.  Extend canonical_error with a direction
#   kwarg if a public right-isometry API is needed later.

function _left_sweep_mps!(
    tensors::Vector{QTensor},
    bond_svs::Vector{SingValSpectrum},
    range::AbstractUnitRange{Int},
    trunc::AbstractTrunc,
)
    ε_total = 0.0
    for i in range
        A = tensors[i].data                        # (χL, d, χR)
        χL, d, χR = size(A)
        M = reshape(A, χL * d, χR)

        F = svd(M)
        tol = length(F.S) * eps(eltype(F.S)) * (isempty(F.S) ? 1.0 : F.S[1])
        S_clean = filter(s -> s > tol, F.S)
        r, ε_bond = _truncate_singular_values(S_clean, trunc)
        svs = F.S[1:r]
        ε_total += ε_bond

        tensors[i] = QTensor(
            reshape(F.U[:, 1:r], χL, d, r), (upper(:vL, χL), upper(:σ, d), lower(:vR, r))
        )
        normalized = isapprox(sum(abs2, svs), 1.0; atol=sqrt(eps(eltype(svs))))
        bond_svs[i + 1] = SingValSpectrum(svs, ε_bond, normalized)

        # Absorb Σ·Vd into next tensor; tagged left-canonical (Up,Up,Low) — transient
        # if the sweep continues, final if site i+1 is the left form's norm carrier.
        # Mixed-form configs retag their centre site to all-Upper afterwards.
        carry = Diagonal(svs) * F.Vt[1:r, :]    # (r, χR)
        A_next = tensors[i + 1].data                # (χR, d_next, χR_next)
        _, d_next, χR_next = size(A_next)
        merged = reshape(carry * reshape(A_next, χR, d_next * χR_next), r, d_next, χR_next)
        tensors[i + 1] = QTensor(
            merged, (upper(:vL, r), upper(:σ, d_next), lower(:vR, χR_next))
        )
    end
    return ε_total
end

function _right_sweep_mps!(
    tensors::Vector{QTensor},
    bond_svs::Vector{SingValSpectrum},
    range::AbstractUnitRange{Int},
    trunc::AbstractTrunc,
)
    ε_total = 0.0
    for i in reverse(range)
        B = tensors[i].data                        # (χL, d, χR)
        χL, d, χR = size(B)
        M = reshape(B, χL, d * χR)

        F = svd(M)
        tol = length(F.S) * eps(eltype(F.S)) * (isempty(F.S) ? 1.0 : F.S[1])
        S_clean = filter(s -> s > tol, F.S)
        r, ε_bond = _truncate_singular_values(S_clean, trunc)
        svs = F.S[1:r]
        ε_total += ε_bond

        tensors[i] = QTensor(
            reshape(F.Vt[1:r, :], r, d, χR), (lower(:vL, r), upper(:σ, d), upper(:vR, χR))
        )
        normalized = isapprox(sum(abs2, svs), 1.0; atol=sqrt(eps(eltype(svs))))
        bond_svs[i] = SingValSpectrum(svs, ε_bond, normalized)

        # Absorb U·Σ into previous tensor; tagged right-canonical (Low,Up,Up) — transient
        # if the sweep continues, final if site i-1 is the right form's norm carrier.
        # Mixed-form configs retag their centre site to all-Upper afterwards.
        carry = F.U[:, 1:r] * Diagonal(svs)   # (χL, r)
        A_prev = tensors[i - 1].data              # (χL_p, d_p, χL)
        χL_p, d_p, _ = size(A_prev)
        merged = reshape(reshape(A_prev, χL_p * d_p, χL) * carry, χL_p, d_p, r)
        tensors[i - 1] = QTensor(merged, (lower(:vL, χL_p), upper(:σ, d_p), upper(:vR, r)))
    end
    return ε_total
end

# Retag site k as the orthogonality centre: both bond arrows point INTO the
# centre, so every leg is Upper (domain). The data is untouched — with the
# trivial metric this is a pure re-labelling (a double bond-end flip).
function _tag_centre!(tensors::Vector{QTensor}, k::Int)
    χL, d, χR = size(tensors[k].data)
    tensors[k] = QTensor(tensors[k].data, (upper(:vL, χL), upper(:σ, d), upper(:vR, χR)))
    return nothing
end

# ==== Canonicalize configs ====================================================

"""
    CanonicalizeConfig

Supertype for canonicalization configurations passed to [`canonicalize`](@ref).
Concrete subtypes: [`LeftCanonical`](@ref), [`RightCanonical`](@ref),
[`BondCanonical`](@ref), [`SiteCanonical`](@ref).
"""
abstract type CanonicalizeConfig end

"""
    LeftCanonical(trunc = NoTrunc())

Config: sweep left-to-right, producing `CanonicalForm(L, L+1)`.
"""
struct LeftCanonical <: CanonicalizeConfig
    trunc::AbstractTrunc
end
LeftCanonical() = LeftCanonical(NoTrunc())

"""
    RightCanonical(trunc = NoTrunc())

Config: sweep right-to-left, producing `CanonicalForm(0, 1)`.
"""
struct RightCanonical <: CanonicalizeConfig
    trunc::AbstractTrunc
end
RightCanonical() = RightCanonical(NoTrunc())

"""
    BondCanonical(k, trunc = NoTrunc())

Config: mixed canonical with orthogonality centre at bond ``k \\leftrightarrow k+1``.
Sites ``1 \\ldots k-1`` become left-canonical, sites ``k+1 \\ldots L`` become
right-canonical, site ``k`` holds the full gauge weight.
Result is tagged `CanonicalForm(k, k+1)`.
"""
struct BondCanonical <: CanonicalizeConfig
    k::Int
    trunc::AbstractTrunc
end
BondCanonical(k::Int) = BondCanonical(k, NoTrunc())

"""
    SiteCanonical(k, trunc = NoTrunc())

Config: orthogonality centre at site ``k``.
Sites ``1 \\ldots k-1`` left-canonical, sites ``k+1 \\ldots L`` right-canonical,
site ``k`` is the un-gauged centre tensor.
Result is tagged `CanonicalForm(k-1, k+1)` (centre excluded from both isometry ranges).
"""
struct SiteCanonical <: CanonicalizeConfig
    k::Int
    trunc::AbstractTrunc
end
SiteCanonical(k::Int) = SiteCanonical(k, NoTrunc())

# ==== canonicalize ============================================================

"""
    canonicalize(mps::FiniteMPS, config::CanonicalizeConfig) -> FiniteMPS

Re-gauge `mps` according to `config` by sweeping with SVD steps on the
existing site tensors.  The input MPS is not mutated; a new `FiniteMPS` is
returned.

Supported configs:

  - [`LeftCanonical`](@ref): full left sweep → `CanonicalForm(L, L+1)`
  - [`RightCanonical`](@ref): full right sweep → `CanonicalForm(0, 1)`
  - [`BondCanonical`](@ref): mixed form, centre at bond k → `CanonicalForm(k, k+1)`
  - [`SiteCanonical`](@ref): mixed form, centre at site k → `CanonicalForm(k-1, k+1)`

# Algorithm: carry propagation on an existing MPS

Unlike [`to_mps`](@ref), which builds an MPS from a dense ``d^L`` state tensor,
`canonicalize` operates directly on the ``L`` rank-3 site tensors.  At each
bond the sweep SVD-factorises the current site tensor and contracts the
"carry" factor into the neighbouring site:

  - **Left sweep** at site ``i``: reshape ``A_i`` to ``(\\chi_L d \\times \\chi_R)``,
    SVD → ``U \\Sigma V^\\dagger``.  Store ``U`` as the new (left-canonical) ``A_i``
    and absorb ``\\Sigma V^\\dagger`` into ``A_{i+1}`` via a matrix multiply.

  - **Right sweep** at site ``i``: reshape ``B_i`` to ``(\\chi_L \\times d \\chi_R)``,
    SVD → ``U \\Sigma V^\\dagger``.  Store ``V^\\dagger`` as the new (right-canonical)
    ``B_i`` and absorb ``U \\Sigma`` into ``B_{i-1}``.

Because the carry is always a ``(r \\times \\chi)`` matrix multiply, the per-bond
cost is ``O(\\chi^2 d)`` rather than ``O(d^L)``.  This is the fundamental
efficiency of MPS methods: re-gauging the whole chain costs linear time in ``L``.
"""
function canonicalize(mps::FiniteMPS, config::LeftCanonical)
    L = length(mps.tensors)
    tensors = copy(mps.tensors)
    bond_svs = copy(mps.bond_svs)
    ε = _left_sweep_mps!(tensors, bond_svs, 1:(L - 1), config.trunc)
    return FiniteMPS(tensors, bond_svs, CanonicalForm(L, L + 1), ε)
end

function canonicalize(mps::FiniteMPS, config::RightCanonical)
    L = length(mps.tensors)
    tensors = copy(mps.tensors)
    bond_svs = copy(mps.bond_svs)
    ε = _right_sweep_mps!(tensors, bond_svs, 2:L, config.trunc)
    return FiniteMPS(tensors, bond_svs, CanonicalForm(0, 1), ε)
end

function canonicalize(mps::FiniteMPS, config::BondCanonical)
    L = length(mps.tensors)
    k = config.k
    tensors = copy(mps.tensors)
    bond_svs = copy(mps.bond_svs)
    ε = (k > 1) ? _left_sweep_mps!(tensors, bond_svs, 1:(k - 1), config.trunc) : 0.0
    ε += (k < L) ? _right_sweep_mps!(tensors, bond_svs, (k + 1):L, config.trunc) : 0.0
    _tag_centre!(tensors, k)
    return FiniteMPS(tensors, bond_svs, CanonicalForm(k, k + 1), ε)
end

function canonicalize(mps::FiniteMPS, config::SiteCanonical)
    L = length(mps.tensors)
    k = config.k
    tensors = copy(mps.tensors)
    bond_svs = copy(mps.bond_svs)
    ε = (k > 1) ? _left_sweep_mps!(tensors, bond_svs, 1:(k - 1), config.trunc) : 0.0
    ε += (k < L) ? _right_sweep_mps!(tensors, bond_svs, (k + 1):L, config.trunc) : 0.0
    _tag_centre!(tensors, k)
    return FiniteMPS(tensors, bond_svs, CanonicalForm(k - 1, k + 1), ε)
end

# ==== canonical_error / is_canonical ==========================================

"""
    canonical_error(A::AbstractArray{<:Number,3}) -> Float64

Measure how far a rank-3 site tensor deviates from **left**-isometry:

```math
\\text{err} = \\|A^\\dagger A - I\\|_F
```

where ``A`` is first reshaped from ``(\\chi_L, d, \\chi_R)`` to the matrix
``(\\chi_L d \\times \\chi_R)``.

# Physical significance

The isometry condition ``A^\\dagger A = I`` is what makes expectation values
computable in ``O(L)`` time instead of ``O(d^L)``.  When the bra and ket MPS
are contracted site by site from the left, the ``A^\\dagger A`` pair at site
``i`` collapses to the identity, so only the open right index survives.
`canonical_error` is the MPS "health metric" — it quantifies how much that
collapse deviates from exact cancellation.

!!! note "Left isometry only"

    This function checks ``A^\\dagger A = I`` (left-canonical condition) only.
    Right-isometry ``BB^\\dagger = I`` is checked internally by [`is_canonical`](@ref)
    but not exposed through this API.  A right-facing overload can be added if needed.
"""
function canonical_error(A::AbstractArray{<:Number,3})
    @debug "canonical_error: checking left isometry (A†A = I) only — right-isometry check not yet exposed"
    χL, d, χR = size(A)
    M = reshape(A, χL * d, χR)
    return norm(M' * M - I(χR))
end

# Private: right-isometry error used by is_canonical; not part of the public API.
function _right_isometry_error_mps(B::AbstractArray{<:Number,3})
    χL, d, χR = size(B)
    M = reshape(B, χL, d * χR)
    return norm(M * M' - I(χL))
end

"""
    is_canonical(ψ::FiniteMPS; tol = 1e-10) -> Bool

Return `true` if every site tensor in `ψ` satisfies its expected isometry
condition as given by `ψ.form`:

| Form tag                    | Sites checked                                     |
|:--------------------------- |:------------------------------------------------- |
| `CanonicalForm(llim, rlim)` | 1..llim-1 left-isometric; rlim..L right-isometric |
| `VidalForm()`               | always `true`                                     |
| `ArbitraryForm()`           | always `false`                                    |

The centre site(s) between `llim` and `rlim` (if any) are **not** checked —
they hold the gauge weight and need not be isometric.
"""
function is_canonical(ψ::FiniteMPS; tol::Float64=1e-10)
    L = length(ψ.tensors)
    form = ψ.form
    if form isa CanonicalForm
        llim, rlim = form.llim, form.rlim
        for i in 1:(llim - 1)
            canonical_error(ψ.tensors[i].data) > tol && return false
        end
        for i in rlim:L
            _right_isometry_error_mps(ψ.tensors[i].data) > tol && return false
        end
        return true
    elseif form isa VidalForm
        return true
    else
        return false
    end
end
