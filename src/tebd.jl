using LinearAlgebra: svd, Diagonal, norm, exp

# Promote a FiniteMPS to complex element type (used by time_evolve for real-time).
function _as_complex(mps::FiniteMPS{D, T, RT}) where {D, T<:Real, RT}
    CT = complex(T)
    tensors = [IndexedTensor(CT.(t.data), t.indices) for t in mps.tensors]
    return FiniteMPS{D, CT, RT}(mps.L, tensors, deepcopy(mps.bond_svs), mps.form)
end
_as_complex(mps::FiniteMPS{D, T, RT}) where {D, T<:Complex, RT} = deepcopy(mps)

# ── to_vidal ──────────────────────────────────────────────────────────────────

"""
    to_vidal(mps::FiniteMPS) -> FiniteMPS{D, T, RT}

Convert a left-canonical `FiniteMPS` to Vidal (Γ-Λ) form.

Each tensor stored in the result satisfies the Vidal normalization condition

```math
\\|\\Lambda_{i-1}\\,\\Gamma_i\\,\\Lambda_i\\|_F = 1
```

where ``\\Lambda_i`` is the diagonal matrix of singular values at bond ``i``
(stored in `bond_svs[i+1]`).

The Γ tensors are obtained from the left-canonical ``A`` tensors by

```math
\\Gamma_i = \\Lambda_{i-1}^{-1} A_i
```

where the inversion is carried out element-wise along the left virtual axis.
"""
function to_vidal(mps::FiniteMPS{D, T, RT}) where {D, T, RT}
    # Step 1: full left-canonical form → normalized, A_i left-isometric
    lc = deepcopy(mps)
    if lc.form != CanonicalForm(lc.L, lc.L + 1)
        left_canonical_sweep!(lc)
    end
    A_tensors = deepcopy(lc.tensors)   # save left-isometric tensors

    # Step 2: right sweep on the LC form gives PROPER Schmidt values at every bond.
    # bond_svs[i] after this sweep = true Schmidt values of |ψ⟩ at bond (i-1, i).
    right_canonical_sweep!(lc)
    λ = lc.bond_svs   # Λ_0 = [1], Λ_1..Λ_{L-1} = Schmidt SVs, Λ_L = [1]

    # Step 3: Γ_i = Λ_{i-1}^{-1} A_i
    L      = lc.L
    gammas = Vector{IndexedTensor{T, 3}}(undef, L)
    for i in 1:L
        A      = A_tensors[i].data
        inv_λL = map(x -> iszero(x) ? zero(RT) : one(RT) / x, λ[i])
        gammas[i] = IndexedTensor(inv_λL .* A, A_tensors[i].indices)
    end

    return FiniteMPS{D, T, RT}(L, gammas, deepcopy(λ), VidalForm())
end

# ── to_canonical ──────────────────────────────────────────────────────────────

"""
    to_canonical(mps::FiniteMPS) -> FiniteMPS{D, T, RT}

Convert a Vidal-form `FiniteMPS` to left-canonical form by absorbing each
``\\Lambda_{i-1}`` into ``\\Gamma_i``:

```math
A_i = \\Lambda_{i-1}\\,\\Gamma_i
```

The resulting MPS is left-canonicalized by a sweep so that `form` is set
correctly and `bond_svs` are refreshed.
"""
function to_canonical(mps::FiniteMPS{D, T, RT}) where {D, T, RT}
    mps.form isa VidalForm ||
        throw(ArgumentError("to_canonical requires VidalForm; got $(mps.form)"))

    L       = mps.L
    tensors = Vector{IndexedTensor{T, 3}}(undef, L)

    for i in 1:L
        Γ   = mps.tensors[i].data   # (χL, d, χR)
        λL  = mps.bond_svs[i]       # Λ_{i-1}
        # A_i = Λ_{i-1} Γ_i
        A_data = λL .* Γ
        tensors[i] = IndexedTensor(A_data, mps.tensors[i].indices)
    end

    canonical = FiniteMPS{D, T, RT}(L, tensors, deepcopy(mps.bond_svs), ArbitraryForm())
    left_canonical_sweep!(canonical)
    return canonical
end

# ── apply_gate! ───────────────────────────────────────────────────────────────

"""
    apply_gate!(mps, gate, bond; trunc=KeepMachineEps())

Apply a two-site unitary `gate` (a ``d^2 \\times d^2`` matrix) to `bond =
(i, i+1)` of a Vidal-form `FiniteMPS`.

The update follows the standard TEBD procedure:

1. Form the two-site tensor ``\\theta = \\Lambda_{i-1}\\,\\Gamma_i\\,\\Lambda_i\\,\\Gamma_{i+1}\\,\\Lambda_{i+1}``
2. Apply the gate: ``\\theta' = (\\text{gate} \\cdot \\theta_{\\text{mat}})``
3. SVD and truncate: ``\\theta' = U\\,\\Sigma\\,V^\\dagger``
4. Recover new ``\\Gamma_i = \\Lambda_{i-1}^{-1} U``, ``\\Lambda_i = \\Sigma / \\|\\Sigma\\|``,
   ``\\Gamma_{i+1} = V^\\dagger \\Lambda_{i+1}^{-1}``

The MPS is updated in-place.
"""
function apply_gate!(
    mps::FiniteMPS{D, T, RT},
    gate::AbstractMatrix,
    bond::Tuple{Int, Int};
    trunc::AbstractTruncation=KeepMachineEps(),
) where {D, T, RT}
    mps.form isa VidalForm ||
        throw(ArgumentError("apply_gate! requires VidalForm; got $(mps.form)"))

    i, j = bond
    j == i + 1 || throw(ArgumentError("apply_gate! requires adjacent sites; got $bond"))

    d      = hilbert_space(D())
    Γi     = mps.tensors[i].data     # (χL, d, χM)
    Γj     = mps.tensors[j].data     # (χM, d, χR)
    λL     = mps.bond_svs[i]         # Λ_{i-1}, length χL
    λM     = mps.bond_svs[j]         # Λ_i,     length χM
    λR     = mps.bond_svs[j + 1]     # Λ_i+1,   length χR

    χL = length(λL);  χM = length(λM);  χR = length(λR)

    # Step 1: θ[α, σi, σj, β] = λL[α] Γi[α,σi,γ] λM[γ] Γj[γ,σj,β] λR[β]
    # Build the (χL*d) × (d*χR) theta matrix for SVD
    TT = promote_type(T, eltype(gate), RT)
    TT <: T ||
        throw(ArgumentError(
            "Gate element type $TT is incompatible with MPS element type $T. " *
            "Create the MPS with T=$TT (e.g. FiniteMPS(...; T=$TT))."))
    θ  = zeros(TT, χL, d, d, χR)
    for α in 1:χL, σi in 1:d, γ in 1:χM, σj in 1:d, β in 1:χR
        θ[α, σi, σj, β] += λL[α] * Γi[α, σi, γ] * λM[γ] * Γj[γ, σj, β] * λR[β]
    end

    # Step 2: apply gate — reshape θ to (χL, d², χR) then (χL*d², χR) isn't right
    # gate acts on (σi, σj): reshape θ to (χL, d*d, χR) → (d*d, χL*χR) → apply → reshape back
    θ_mat = reshape(permutedims(θ, (2, 3, 1, 4)), d*d, χL*χR)   # (d², χL*χR)
    θ_mat = reshape(gate, d*d, d*d) * θ_mat                       # (d², χL*χR)
    θ′    = permutedims(reshape(θ_mat, d, d, χL, χR), (3, 1, 2, 4))  # (χL,d,d,χR)

    # Step 3: SVD — reshape to (χL*d, d*χR)
    M = reshape(permutedims(θ′, (1, 2, 3, 4)), χL*d, d*χR)
    F = svd(M; full=false)
    r, _ = _truncate(F.S, trunc)
    r     = max(r, 1)

    U  = F.U[:, 1:r]      # (χL*d, r)
    Σ  = F.S[1:r]          # (r,)
    Vt = F.Vt[1:r, :]     # (r, d*χR)

    # Normalize Σ so the MPS stays normalized
    nrm = norm(Σ)
    Σ ./= nrm

    # Step 4: recover Γ tensors
    inv_λL = map(x -> iszero(x) ? zero(RT) : one(RT) / x, λL)
    inv_λR = map(x -> iszero(x) ? zero(RT) : one(RT) / x, λR)

    # Γi_new[α, σi, γ] = Λ_{i-1}^{-1}[α] * U[α*d+σi, γ]
    Γi_new = reshape(inv_λL .* reshape(U, χL, d * r), χL, d, r)

    # Γj_new[γ, σj, β] = Vt[γ, σj*χR+β] * Λ_{i+1}^{-1}[β]
    Γj_new = reshape(reshape(Vt, r, d, χR) .* reshape(inv_λR, 1, 1, :), r, d, χR)

    _set_tensor!(mps, i, TT.(Γi_new))
    _set_tensor!(mps, j, TT.(Γj_new))
    mps.bond_svs[j] = RT.(Σ)

    return mps
end

# ── trotter_step! ─────────────────────────────────────────────────────────────

"""
    trotter_step!(mps, H_bonds, dt; imag=false, trunc=KeepMachineEps())

Apply one first-order Trotter step to a Vidal-form `FiniteMPS`.

`H_bonds[i]` is the ``d^2 \\times d^2`` local two-site Hamiltonian for the bond
between sites ``i`` and ``i+1``.  The gate applied at each bond is

```math
U_i(dt) = e^{-i\\,dt\\,h_i}  \\quad (\\text{real time})
\\quad\\text{or}\\quad
U_i(dt) = e^{-dt\\,h_i}  \\quad (\\text{imaginary time})
```

The step sweeps all odd bonds then all even bonds (first-order Lie-Trotter).
"""
function trotter_step!(
    mps::FiniteMPS{D, T, RT},
    H_bonds::AbstractVector,
    dt::Real;
    imag::Bool=false,
    trunc::AbstractTruncation=KeepMachineEps(),
) where {D, T, RT}
    L    = mps.L
    exponent = imag ? -dt : -im * dt

    # odd bonds: (1,2), (3,4), ...
    for i in 1:2:L-1
        gate = exp(exponent * H_bonds[i])
        apply_gate!(mps, gate, (i, i + 1); trunc)
    end
    # even bonds: (2,3), (4,5), ...
    for i in 2:2:L-1
        gate = exp(exponent * H_bonds[i])
        apply_gate!(mps, gate, (i, i + 1); trunc)
    end

    return mps
end

# ── time_evolve ───────────────────────────────────────────────────────────────

"""
    time_evolve(mps, H_bonds, t_end, dt; imag=false, trunc=KeepMachineEps())

Evolve `mps` from ``t=0`` to ``t=t_{\\mathrm{end}}`` in steps of `dt` using
first-order Trotter decomposition.  Returns a new normalized `FiniteMPS` in
`CanonicalForm`.

Set `imag=true` for imaginary time evolution (``e^{-\\tau H}``); the state is
re-normalized after every step so it converges to the ground state.
"""
function time_evolve(
    mps::FiniteMPS{D, T, RT},
    H_bonds::AbstractVector,
    t_end::Real,
    dt::Real;
    imag::Bool=false,
    trunc::AbstractTruncation=KeepMachineEps(),
) where {D, T, RT}
    n_steps = round(Int, t_end / dt)

    # Real-time evolution gates are complex → promote MPS element type.
    start = if imag
        deepcopy(mps)
    else
        _as_complex(mps)
    end
    start.form isa CanonicalForm || left_canonical_sweep!(start)
    vidal = to_vidal(start)

    for _ in 1:n_steps
        trotter_step!(vidal, H_bonds, dt; imag, trunc)
        if imag
            # re-normalize: absorb scale into bond_svs[L+1] boundary then re-canonicalize
            can = to_canonical(vidal)
            vidal = to_vidal(can)
        end
    end

    return to_canonical(vidal)
end
