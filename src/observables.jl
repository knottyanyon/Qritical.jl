using LinearAlgebra: tr

# ── local_expectation ─────────────────────────────────────────────────────────

"""
    local_expectation(mps, op, site) -> Real

Single-site expectation value ``\\langle\\psi|O_i|\\psi\\rangle / \\langle\\psi|\\psi\\rangle``
computed via the mixed canonical form.

With the orthogonality center at `site`, left and right environments both
reduce to the identity, so the full contraction collapses to:

```math
\\langle O_i \\rangle = \\frac{\\operatorname{Tr}(O\\,\\rho_i)}{\\operatorname{Tr}(\\rho_i)},
\\qquad \\rho_i = A_{(i)}^\\dagger A_{(i)}
```

where ``A_{(i)}`` is the center tensor reshaped to ``(\\chi_L \\chi_R) \\times d``.

The denominator ``\\operatorname{Tr}(\\rho_i) = \\|A_{(i)}\\|_F^2`` normalizes
the result so that unnormalized input states are handled correctly.

`op` must be a ``d \\times d`` matrix compatible with the local Hilbert space
of the MPS.  `VidalForm` input is converted to canonical form automatically.
"""
function local_expectation(mps::FiniteMPS, op::AbstractMatrix, site::Int)
    mps_c = deepcopy(mps)
    if mps_c.form isa VidalForm
        mps_c = to_canonical(mps_c)
    elseif !(mps_c.form isa CanonicalForm)
        left_canonical_sweep!(mps_c)
    end
    move_center!(mps_c, site)
    A = mps_c.tensors[site].data     # (χL, d, χR)
    χL, d, χR = size(A)
    A_mat = reshape(permutedims(A, (1, 3, 2)), χL * χR, d)   # (χL·χR, d)
    ρ = A_mat' * A_mat               # (d, d) reduced density matrix
    return real(tr(op * ρ)) / real(tr(ρ))
end

# ── entanglement_spectrum ─────────────────────────────────────────────────────

"""
    entanglement_spectrum(mps, bond) -> Vector

Return the Schmidt values at `bond` (an index into `bond_svs`).

No new SVD is performed; the values are read directly from the stored
`bond_svs` array.  Bond indices run from 1 (left boundary) to ``L+1``
(right boundary); interior bond ``b`` sits between sites ``b-1`` and ``b``.
"""
function entanglement_spectrum(mps::FiniteMPS, bond::Int)
    return copy(mps.bond_svs[bond])
end

# ── entanglement_entropy — vector form ────────────────────────────────────────

"""
    entanglement_entropy(mps) -> Vector{Real}

Return a length-``(L-1)`` vector of per-bond entanglement entropies, one for
each interior bond.  Entry ``b`` equals ``S(b+1)`` from the scalar
`entanglement_entropy(mps, bond)` convention:

```math
S_b = -\\sum_i \\lambda_i^2 \\log \\lambda_i^2
```

where ``\\lambda_i`` are the normalized Schmidt values at bond ``b+1``.
"""
function entanglement_entropy(mps::FiniteMPS)
    return [entanglement_entropy(mps, b) for b in 2:mps.L]
end
