using LinearAlgebra

# ==== Vidal Γ–Λ form ==========================================================

"""
    to_vidal(mps::FiniteMPS) -> FiniteMPS

Convert a `FiniteMPS` to the **Vidal (Γ–Λ) form**.

In Vidal's representation every site tensor ``\\Gamma_i`` and every bond carries
an explicit singular-value vector ``\\Lambda_i``:

```math
|\\psi\\rangle = \\Lambda_0\\, \\Gamma_1\\, \\Lambda_1\\, \\Gamma_2\\, \\Lambda_2 \\cdots \\Gamma_L\\, \\Lambda_L
```

where ``\\Lambda_0 = \\Lambda_L = [1]`` (open-chain boundary).

**Relation to canonical tensors.** Given the left-canonical tensors ``A_i`` from a
left-to-right sweep and the bond singular values ``\\lambda^{(i)} = \\Lambda_i``:

```math
\\Gamma_i = \\Lambda_{i-1}^{-1}\\, A_i
```

The combination ``\\Gamma_i \\Lambda_i`` recovers the right-canonical tensor ``B_i``
**only when ``\\Lambda_i`` holds the true Schmidt values** of the full state at bond ``i``
(i.e. when the canonical centre was at that bond before `to_vidal` was called).  For a
truncated or non-canonical input the identity holds only approximately.

# Implementation

1. Run a full left-to-right sweep to get left-canonical ``A_i`` tensors and bond SVs.
2. At each site, invert the left bond SV vector (clamped to avoid division by zero on
   reduced bonds) and absorb it into the site tensor:
   ``\\Gamma_i[\\alpha, \\sigma, \\beta] = \\Lambda_{i-1}^{-1}[\\alpha] \\cdot A_i[\\alpha, \\sigma, \\beta]``
3. Store the ``\\Gamma_i`` as site tensors and keep `bond_svs` unchanged.
4. Tag the result with `VidalForm()`.

The bond SVs already live in `mps.bond_svs` (set by `to_mps`), so step 1 calls
`canonicalize(mps, LeftCanonical())` if the input is not already fully
left-canonical.

# Edge cases

Zeros in ``\\Lambda`` (reduced effective bond) are handled by setting the
reciprocal to zero rather than dividing — this keeps the Γ tensor finite on the
physical subspace and zero on the null directions.
"""
function to_vidal(mps::FiniteMPS)::FiniteMPS
    # Ensure we have a fully left-canonical MPS to start from
    canonical_mps = if mps.form == CanonicalForm(length(mps.tensors), length(mps.tensors) + 1)
        mps
    else
        canonicalize(mps, LeftCanonical())
    end

    L = length(canonical_mps.tensors)
    Γ_tensors = Vector{QTensor}(undef, L)

    for i in 1:L
        A_i = canonical_mps.tensors[i].data      # (χL, d, χR)
        λ_left = canonical_mps.bond_svs[i].values  # bond to the left of site i

        # Invert λ_left with zero-clamping for reduced bonds
        λ_inv = map(λ -> iszero(λ) ? zero(λ) : inv(λ), λ_left)

        # Γᵢ[α, σ, β] = λ_left_inv[α] * A_i[α, σ, β]
        Γ_data = λ_inv .* A_i   # broadcast over first axis
        Γ_tensors[i] = QTensor(Γ_data, canonical_mps.tensors[i].indices)
    end

    return FiniteMPS(Γ_tensors, canonical_mps.bond_svs, VidalForm(), canonical_mps.ε)
end

"""
    to_canonical(mps::FiniteMPS) -> FiniteMPS

Convert a `FiniteMPS` in **Vidal (Γ–Λ) form** back to **left-canonical form**.

The inverse of [`to_vidal`](@ref). Given Γ tensors and bond SV vectors ``\\Lambda_i``,
the left-canonical tensor at each site is:

```math
A_i = \\Lambda_{i-1}\\, \\Gamma_i
```

where ``\\Lambda_{i-1}`` is absorbed from the left.

The resulting MPS satisfies ``A_i^\\dagger A_i = I`` (left isometry) on every site
but the last, and is tagged `CanonicalForm(L, L+1)`.

# Arguments
- `mps::FiniteMPS` — must be in `VidalForm()` (no runtime check; caller's responsibility)

# Returns
A left-canonical `FiniteMPS` with the same state and `form == CanonicalForm(L, L+1)`.
"""
function to_canonical(mps::FiniteMPS)::FiniteMPS
    L = length(mps.tensors)
    A_tensors = Vector{QTensor}(undef, L)

    for i in 1:L
        Γ_i   = mps.tensors[i].data                # (χL, d, χR)
        λ_left = mps.bond_svs[i].values             # singular values on bond to the left

        # Aᵢ[α, σ, β] = λ_left[α] * Γᵢ[α, σ, β]
        A_data = λ_left .* Γ_i
        A_tensors[i] = QTensor(A_data, mps.tensors[i].indices)
    end

    return FiniteMPS(A_tensors, mps.bond_svs, CanonicalForm(L, L + 1), mps.ε)
end
