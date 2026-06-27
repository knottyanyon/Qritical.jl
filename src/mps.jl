"""
    AbstractMPSForm

Supertype for all MPS gauge-form tags. Every `FiniteMPS` carries one of these
to record which isometry conditions its site tensors currently satisfy.
Concrete subtypes: [`CanonicalForm`](@ref), [`VidalForm`](@ref), [`ArbitraryForm`](@ref).
"""
abstract type AbstractMPSForm end

"""
    CanonicalForm(llim, rlim)

Canonical form tag for a finite MPS.

- `llim::Int`: sites ``1, \\ldots, \\texttt{llim}`` are left-canonical  (``A_i^\\dagger A_i = I``)
- `rlim::Int`: sites ``\\texttt{rlim}, \\ldots, L`` are right-canonical (``B_i B_i^\\dagger = I``)

Common cases:

| Form | `llim` | `rlim` |
|------|--------|--------|
| Fully left-canonical  | ``L``   | ``L+1`` |
| Fully right-canonical | ``0``   | ``1``   |
| Mixed (site ``k`` is centre) | ``k`` | ``k`` |
"""
struct CanonicalForm <: AbstractMPSForm
    llim::Int
    rlim::Int
end

"""
    VidalForm()

Form tag indicating the MPS is in Vidal's ``\\Gamma\\Lambda`` representation.
Site tensors are stored as ``\\Gamma_i`` and bond tensors as ``\\Lambda_i``.
[`is_canonical`](@ref) returns `true` for this form without checking isometry.
"""
struct VidalForm <: AbstractMPSForm end

"""
    ArbitraryForm()

Form tag indicating no isometry conditions are guaranteed — the MPS has been
modified (e.g. by [`add_mps`](@ref)) without a subsequent canonicalization.
[`is_canonical`](@ref) returns `false` for this form.
"""
struct ArbitraryForm <: AbstractMPSForm end

"""
    FiniteMPS(tensors, bond_svs, form, ε)

Matrix-product state for a finite open chain with ``L`` sites.

# Fields
- `tensors::Vector{QTensor}`: ``L`` site tensors, each order-3 with legs
  ``(\\texttt{vL}::\\text{Upper},\\; \\sigma::\\text{Lower},\\; \\texttt{vR}::\\text{Lower})``
- `bond_svs::Vector{SingValSpectrum}`: ``L+1`` bond spectra; boundaries carry the
  trivial spectrum ``[1.0]``
- `form::AbstractMPSForm`: canonical-form tag
- `ε::Float64`: accumulated truncation error — sum of per-bond discarded singular-value
  2-norms (zero for `NoTrunc`)

# Index convention (MasterPlan §13/§23)
```
    vL (Upper, χ_L)
    │
    ● σ (Lower, d)
    │
    vR (Lower, χ_R)
```
Boundary sites have ``\\chi_L = 1`` (left) and ``\\chi_R = 1`` (right).
"""
struct FiniteMPS
    tensors::Vector{QTensor}
    bond_svs::Vector{SingValSpectrum}
    form::AbstractMPSForm
    ε::Float64
end

# ==== Internal helpers ========================================================

function _left_sweep(ψ::QTensor, d::Vector{Int}, trunc::AbstractTrunc)
    L = length(d)
    tensors  = Vector{QTensor}(undef, L)
    bond_svs = Vector{SingValSpectrum}(undef, L + 1)
    ε_total  = 0.0

    bond_svs[1]   = SingValSpectrum([1.0], 0.0, true)
    bond_svs[L+1] = SingValSpectrum([1.0], 0.0, true)

    carry  = reshape(ψ.data, 1, d...)  # (1, d₁, d₂, …, d_L)
    χ_left = 1

    for i in 1:(L - 1)
        right_dim = prod(d[(i + 1):end])
        M = reshape(carry, χ_left * d[i], right_dim)

        F   = svd(M)
        tol = length(F.S) * eps(eltype(F.S)) * (isempty(F.S) ? 1.0 : F.S[1])
        S_clean   = filter(s -> s > tol, F.S)
        r, ε_bond = _truncate_singular_values(S_clean, trunc)

        svs = F.S[1:r]
        ε_total += ε_bond

        tensors[i]    = QTensor(reshape(F.U[:, 1:r], χ_left, d[i], r),
                                (upper(:vL, χ_left), lower(:σ, d[i]), lower(:vR, r)))
        normalized    = isapprox(sum(abs2, svs), 1.0; atol=sqrt(eps(eltype(svs))))
        bond_svs[i+1] = SingValSpectrum(svs, ε_bond, normalized)

        carry  = reshape(Diagonal(svs) * F.Vt[1:r, :], r, d[(i + 1):end]...)
        χ_left = r
    end

    # Last site: carry has shape (χ_left, d_L)
    tensors[L] = QTensor(reshape(carry, χ_left, d[L], 1),
                         (upper(:vL, χ_left), lower(:σ, d[L]), lower(:vR, 1)))

    return FiniteMPS(tensors, bond_svs, CanonicalForm(L, L + 1), ε_total)
end

function _right_sweep(ψ::QTensor, d::Vector{Int}, trunc::AbstractTrunc)
    L = length(d)
    tensors  = Vector{QTensor}(undef, L)
    bond_svs = Vector{SingValSpectrum}(undef, L + 1)
    ε_total  = 0.0

    bond_svs[1]   = SingValSpectrum([1.0], 0.0, true)
    bond_svs[L+1] = SingValSpectrum([1.0], 0.0, true)

    carry   = reshape(ψ.data, d..., 1)  # (d₁, …, d_L, 1)
    χ_right = 1

    for i in L:-1:2
        left_dim = prod(d[1:(i - 1)])
        M = reshape(carry, left_dim, d[i] * χ_right)

        F   = svd(M)
        tol = length(F.S) * eps(eltype(F.S)) * (isempty(F.S) ? 1.0 : F.S[1])
        S_clean   = filter(s -> s > tol, F.S)
        r, ε_bond = _truncate_singular_values(S_clean, trunc)

        svs = F.S[1:r]
        ε_total += ε_bond

        tensors[i]  = QTensor(reshape(F.Vt[1:r, :], r, d[i], χ_right),
                              (upper(:vL, r), lower(:σ, d[i]), lower(:vR, χ_right)))
        normalized  = isapprox(sum(abs2, svs), 1.0; atol=sqrt(eps(eltype(svs))))
        bond_svs[i] = SingValSpectrum(svs, ε_bond, normalized)

        carry   = reshape(F.U[:, 1:r] * Diagonal(svs), d[1:(i - 1)]..., r)
        χ_right = r
    end

    # Site 1: carry has shape (d₁, χ_right)
    tensors[1] = QTensor(reshape(carry, 1, d[1], χ_right),
                         (upper(:vL, 1), lower(:σ, d[1]), lower(:vR, χ_right)))

    return FiniteMPS(tensors, bond_svs, CanonicalForm(0, 1), ε_total)
end

# ==== Public API ==============================================================

"""
    to_mps(ψ::QTensor; trunc::AbstractTrunc = NoTrunc(), form::Symbol = :left) -> FiniteMPS

Decompose a full quantum state tensor into a canonical matrix-product state via
iterated SVD.

# Arguments
- `ψ::QTensor`: full state tensor with ``L`` physical legs, all of type `Lower`
- `trunc::AbstractTrunc`: truncation strategy (default: `NoTrunc()`)
- `form::Symbol`: `:left` for left-canonical sweep or `:right` for right-canonical sweep

# Physical invariants
| Property | Left | Right |
|----------|------|-------|
| Isometry | ``A_i^\\dagger A_i = I`` (sites 1..L-1) | ``B_i B_i^\\dagger = I`` (sites 2..L) |
| Form tag | `CanonicalForm(L, L+1)` | `CanonicalForm(0, 1)` |
| Norm carrier | last site | first site |

- **Reconstruction**: contracting all tensors recovers ``\\psi`` within `mps.ε`
- **Error accounting**: `mps.ε = \\sum_i \\texttt{bond\\_svs}[i].\\varepsilon`
- **Boundary spectra**: `bond_svs[1] = bond_svs[L+1] = [1.0]`

# Algorithm: carry-propagation via iterated SVD

**Left sweep.** At each bond cut ``i``, the carry tensor (shape
``(\\chi_{i-1},\\, d_i,\\, d_{i+1},\\ldots,d_L)``) is reshaped into a matrix

```math
M = \\operatorname{reshape}(\\texttt{carry},\\; \\chi_{i-1} d_i,\\; d_{i+1}\\cdots d_L)
```

and factored as ``M = U\\Sigma V^\\dagger``.  ``U`` (shape ``\\chi_{i-1} d_i \\times r``,
isometric columns) becomes site tensor ``A_i``, while ``\\Sigma V^\\dagger`` becomes
the new carry.  Because ``U`` has orthonormal columns, ``A_i^\\dagger A_i = I_r``
automatically.  The norm of ``\\psi`` rides rightward in the carry and ends up
absorbed into the last site.

**Right sweep.** The right sweep is the exact mirror: the carry propagates
leftward as ``U\\Sigma``, and ``V^\\dagger`` (rows orthonormal) becomes site
tensor ``B_i``, satisfying ``B_i B_i^\\dagger = I_r``.  Site 1 inherits the norm.

**Noise cleaning before truncation.** `LinearAlgebra.svd` can return tiny
floating-point artifacts in the singular-value tail at the level of
``\\varepsilon_\\text{mach} \\times \\sigma_1``.  These are filtered out before
[`_truncate_singular_values`](@ref) is called, preventing them from being counted
as genuine Schmidt values and inflating ``\\chi`` when `trunc = NoTrunc()`.
The threshold used is the classical Golub–Van Loan numerical-rank criterion:
``\\sigma_k > n\\, \\varepsilon_\\text{mach}\\, \\sigma_1``.
"""
function to_mps(ψ::QTensor; trunc::AbstractTrunc=NoTrunc(), form::Symbol=:left)::FiniteMPS
    L = ndims(ψ.data)
    d = [dim(ψ.indices[i]) for i in 1:L]
    if form === :left
        return _left_sweep(ψ, d, trunc)
    elseif form === :right
        return _right_sweep(ψ, d, trunc)
    else
        throw(ArgumentError("to_mps: form must be :left or :right, got $form"))
    end
end

# ==== MPS addition ============================================================

"""
    add_mps(a, ψ::FiniteMPS, b, φ::FiniteMPS; trunc=NoTrunc()) -> FiniteMPS

Compute the superposition ``a|\\psi\\rangle + b|\\varphi\\rangle`` as a new MPS.

The result is built by the direct-sum (block-diagonal) construction: at each
interior site the virtual bond is split into a block for ``|\\psi\\rangle`` and a
block for ``|\\varphi\\rangle``, giving bond dimension ``\\chi_\\psi + \\chi_\\varphi``.
The boundary conditions stitch the two chains together so the boundary tensors
absorb the coefficients ``a`` and ``b``.

After the block-diagonal assembly, a left-to-right re-canonicalization sweep
with `trunc` is applied to compress the bond dimension and produce a
`CanonicalForm(L, L+1)` result.

# Arguments
- `a`, `b` — scalar coefficients (zero coefficient → zero contribution)
- `ψ`, `φ`  — input MPS (must have the same length and physical dimensions)
- `trunc`   — truncation strategy applied during the recompression sweep

# Returns
A left-canonical `FiniteMPS` representing ``a|\\psi\\rangle + b|\\varphi\\rangle``.

# See also
`Base.:+(ψ, φ)` — unit-coefficient shorthand.
"""
function add_mps(
    a::Number, ψ::FiniteMPS, b::Number, φ::FiniteMPS; trunc::AbstractTrunc=NoTrunc()
)::FiniteMPS
    L = length(ψ.tensors)
    L == length(φ.tensors) || throw(
        ArgumentError("add_mps: MPS lengths must match, got $L and $(length(φ.tensors))")
    )
    all(
        size(ψ.tensors[i].data, 2) == size(φ.tensors[i].data, 2) for i in 1:L
    ) || throw(
        ArgumentError("add_mps: physical dimensions must match at every site")
    )

    T = promote_type(typeof(a), typeof(b), eltype(ψ.tensors[1].data), eltype(φ.tensors[1].data))

    # Build direct-sum site tensors
    tensors = Vector{QTensor}(undef, L)
    for i in 1:L
        Aψ = convert(Array{T}, ψ.tensors[i].data)  # (χLψ, d, χRψ)
        Aφ = convert(Array{T}, φ.tensors[i].data)  # (χLφ, d, χRφ)
        χLψ, d, χRψ = size(Aψ)
        χLφ, _d, χRφ = size(Aφ)

        # Absorb coefficients at boundary sites
        if i == 1
            Aψ = a .* Aψ
            Aφ = b .* Aφ
        end

        χL_new = i == 1 ? 1 : χLψ + χLφ
        χR_new = i == L ? 1 : χRψ + χRφ

        if i == 1
            # Left boundary: χL = 1 in both; stack horizontally in vR direction
            blk = zeros(T, 1, d, χR_new)
            blk[1, :, 1:χRψ]         = Aψ[1, :, :]
            blk[1, :, (χRψ+1):end]   = Aφ[1, :, :]
        elseif i == L
            # Right boundary: χR = 1 in both; stack vertically in vL direction
            blk = zeros(T, χL_new, d, 1)
            blk[1:χLψ, :, 1]         = Aψ[:, :, 1]
            blk[(χLψ+1):end, :, 1]   = Aφ[:, :, 1]
        else
            # Interior: block-diagonal in (vL, vR) while sharing the physical leg
            blk = zeros(T, χL_new, d, χR_new)
            blk[1:χLψ, :, 1:χRψ]               = Aψ
            blk[(χLψ+1):end, :, (χRψ+1):end]   = Aφ
        end

        new_indices = (upper(:vL, χL_new), lower(:σ, d), lower(:vR, χR_new))
        tensors[i]  = QTensor(blk, new_indices)
    end

    # Build trivial boundary spectra — the block-diagonal MPS is in ArbitraryForm
    bond_svs = fill(SingValSpectrum([1.0], 0.0, true), L + 1)
    raw = FiniteMPS(tensors, bond_svs, ArbitraryForm(), 0.0)

    # Recompress via left sweep to get a canonical form and apply truncation
    return _recompress_left(raw, trunc)
end

"""
    _recompress_left(mps, trunc) -> FiniteMPS

Internal helper: apply a left-to-right QR/SVD sweep to bring `mps` into
left-canonical form and apply `trunc` at each bond. Used by `add_mps` to
compress the block-diagonal superposition.

This is identical in spirit to `_left_sweep` but operates on existing
site tensors (no full state tensor to start from) by treating the first
tensor as the initial carry.
"""
function _recompress_left(mps::FiniteMPS, trunc::AbstractTrunc)::FiniteMPS
    L = length(mps.tensors)
    tensors  = Vector{QTensor}(undef, L)
    bond_svs = Vector{SingValSpectrum}(undef, L + 1)
    ε_total  = 0.0

    bond_svs[1]   = SingValSpectrum([1.0], 0.0, true)
    bond_svs[L+1] = SingValSpectrum([1.0], 0.0, true)

    # carry starts as the first site tensor, reshape to matrix (d, χR)
    carry = mps.tensors[1].data  # (1, d, χR)
    χL = size(carry, 1)

    for i in 1:(L - 1)
        d   = size(carry, 2)
        χR  = size(carry, 3)
        M   = reshape(carry, χL * d, χR)

        F   = svd(M)
        tol = length(F.S) * eps(eltype(F.S)) * (isempty(F.S) ? 1.0 : F.S[1])
        S_clean   = filter(s -> s > tol, F.S)
        r, ε_bond = _truncate_singular_values(S_clean, trunc)

        svs = F.S[1:r]
        ε_total += ε_bond

        tensors[i]    = QTensor(reshape(F.U[:, 1:r], χL, d, r),
                                (upper(:vL, χL), lower(:σ, d), lower(:vR, r)))
        normalized    = isapprox(sum(abs2, svs), 1.0; atol=sqrt(eps(eltype(svs))))
        bond_svs[i+1] = SingValSpectrum(svs, ε_bond, normalized)

        # Next carry: Σ·Vt contracted with the next site tensor
        SV = Diagonal(svs) * F.Vt[1:r, :]   # (r, χR)
        next = mps.tensors[i + 1].data       # (χR_old, d_next, χR_next)
        d_next  = size(next, 2)
        χR_next = size(next, 3)
        carry   = reshape(SV * reshape(next, χR, d_next * χR_next), r, d_next, χR_next)
        χL      = r
    end

    # Last site: carry is already the final tensor
    tensors[L] = QTensor(carry, (upper(:vL, χL), lower(:σ, size(carry, 2)), lower(:vR, size(carry, 3))))

    return FiniteMPS(tensors, bond_svs, CanonicalForm(L, L + 1), ε_total)
end

"""
    Base.:+(ψ::FiniteMPS, φ::FiniteMPS) -> FiniteMPS

Compute ``|\\psi\\rangle + |\\varphi\\rangle`` with unit coefficients.

Sugar for `add_mps(1, ψ, 1, φ)`.
"""
Base.:+(ψ::FiniteMPS, φ::FiniteMPS) = add_mps(1, ψ, 1, φ)

"""
    _scale(mps::FiniteMPS, a) -> FiniteMPS

Scale the MPS by a scalar `a`, absorbing `a` into the first site tensor.

Returns a new `FiniteMPS` with `form = ArbitraryForm()` and the same bond
structure, since scaling a single tensor generally destroys any isometry.
"""
function _scale(mps::FiniteMPS, a::Number)
    T = promote_type(typeof(a), eltype(mps.tensors[1].data))
    new_tensors = copy(mps.tensors)
    old_data = mps.tensors[1].data
    new_data = T.(a .* old_data)
    new_tensors[1] = QTensor(new_data, mps.tensors[1].indices)
    return FiniteMPS(new_tensors, copy(mps.bond_svs), ArbitraryForm(), mps.ε)
end
