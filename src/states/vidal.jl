using LinearAlgebra
# `using LinearAlgebra` imports Julia's standard linear algebra module.
# Like Python's `import numpy.linalg` or `from numpy.linalg import *`.
# This is needed for `svd`, `Diagonal`, `norm`, `I`, etc.

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
    # `::FiniteMPS` is the return type annotation. Julia enforces this at runtime.

    # Ensure we have a fully left-canonical MPS to start from
    canonical_mps = if mps.form == CanonicalForm(length(mps.tensors), length(mps.tensors) + 1)
        mps
    else
        canonicalize(mps, LeftCanonical())
    end
    # This is Julia's IF-EXPRESSION: `x = if cond; val1; else; val2; end`.
    # Like Python's `x = val1 if cond else val2` (ternary), but works for multi-line blocks.
    # We check: is mps ALREADY fully left-canonical (CanonicalForm(L, L+1))?
    # If yes: use it directly (avoid redundant work). If no: run a left sweep first.
    # Physics: Γ = Λ⁻¹·A requires A to be left-canonical so that the bond SVs are valid Schmidt values.

    L = length(canonical_mps.tensors)
    Γ_tensors = Vector{QTensor}(undef, L)
    # Pre-allocate array of L QTensors (uninitialized). Like numpy's np.empty(L, dtype=object).

    for i in 1:L
        A_i = canonical_mps.tensors[i].data      # (χL, d, χR): the left-canonical site tensor
        λ_left = canonical_mps.bond_svs[i].values  # bond to the left of site i: Λ_{i-1}
        # `bond_svs[i]` = the bond between sites i-1 and i (bond i in 1-indexed notation).
        # For site 1: bond_svs[1].values = [1.0] (trivial left boundary).
        # For site k: bond_svs[k].values = Λ_{k-1} (Schmidt values at bond k-1|k).
        # Physics: Λ_{i-1} are the Schmidt values at the bond to the LEFT of site i.

        # Invert λ_left with zero-clamping for reduced bonds
        λ_inv = map(λ -> iszero(λ) ? zero(λ) : inv(λ), λ_left)
        # `map(f, collection)` = Python's `[f(x) for x in collection]` or `map(lambda x: f(x), coll)`.
        # `λ -> expr` is an anonymous function (lambda): `lambda λ: expr` in Python.
        # `iszero(λ)` = `λ == 0` but type-stable (works for any numeric type).
        # `zero(λ)` = zero of the same type as λ (e.g. 0.0 if λ is Float64). Like `type(λ)(0)`.
        # `inv(λ)` = 1/λ (multiplicative inverse). Like Python's `1/λ`.
        # Physics: if a Schmidt value is zero, the corresponding Schmidt state has no weight.
        # Setting the reciprocal to zero "projects out" the null directions rather than dividing by zero.

        # Γᵢ[α, σ, β] = λ_left_inv[α] * A_i[α, σ, β]
        Γ_data = λ_inv .* A_i   # broadcast over first axis
        # `λ_inv` has shape (χL,) and A_i has shape (χL, d, χR).
        # Julia's `.` broadcast aligns dimensions from the LEFT: λ_inv is treated as (χL, 1, 1).
        # So each slice A_i[α, :, :] gets multiplied by λ_inv[α].
        # In numpy: Γ_data = λ_inv[:, None, None] * A_i  (explicit broadcasting with None).
        # Physics: Γᵢ = Λ_{i-1}⁻¹ · Aᵢ — divides out the left Schmidt values to "un-gauge" the tensor.
        # This makes Γᵢ gauge-invariant: it encodes the local structure of the state
        # without reference to the specific canonical form (left, right, or mixed).

        Γ_tensors[i] = QTensor(Γ_data, canonical_mps.tensors[i].indices)
        # Wrap the data with the same index metadata as A_i (variance tags are unchanged).
        # The Γ tensor has the same shape and leg structure as A_i.
    end

    return FiniteMPS(Γ_tensors, canonical_mps.bond_svs, VidalForm(), canonical_mps.ε)
    # Store Γ tensors as site tensors, but keep the SAME bond_svs (those are the Λ values).
    # Tag with VidalForm() to indicate this is the Γ–Λ representation.
    # ε is carried over from the canonical MPS (total truncation error doesn't change here).
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
    A_tensors = Vector{QTensor}(undef, L)   # pre-allocate output array

    for i in 1:L
        Γ_i   = mps.tensors[i].data                # (χL, d, χR): the Γ tensor at site i
        λ_left = mps.bond_svs[i].values             # singular values on bond to the left: Λ_{i-1}
        # Same indexing as in to_vidal: bond_svs[i] = Λ_{i-1} (left bond of site i).

        # Aᵢ[α, σ, β] = λ_left[α] * Γᵢ[α, σ, β]
        A_data = λ_left .* Γ_i
        # `λ_left .* Γ_i` = numpy's `λ_left[:, None, None] * Γ_i`.
        # Each slice Γ_i[α, :, :] gets multiplied by λ_left[α].
        # Physics: Aᵢ = Λ_{i-1} · Γᵢ — re-gauges the Γ tensor back to left-canonical form
        # by restoring the Schmidt values. This is the INVERSE of what to_vidal did.
        # After this, A_i†A_i = I holds (left isometry) because:
        #   Γᵢ was derived from Aᵢ by Γᵢ = Λ_{i-1}⁻¹ Aᵢ, so
        #   Aᵢ = Λ_{i-1} Γᵢ = Λ_{i-1} (Λ_{i-1}⁻¹ Aᵢ) = Aᵢ — we recover the original.

        A_tensors[i] = QTensor(A_data, mps.tensors[i].indices)
        # Wrap with the same index metadata as the Γ tensor (shape and variance unchanged).
    end

    return FiniteMPS(A_tensors, mps.bond_svs, CanonicalForm(L, L + 1), mps.ε)
    # Bond SVs are UNCHANGED (they're the Λ values in Vidal form = bond SVs in canonical form).
    # Tag with CanonicalForm(L, L+1): sites 1..L-1 are left-canonical, last site is norm carrier.
    # ε carries over (no new truncation happens in this conversion).
end
