# §6.2–6.3  Finite MPO — construction and expectation value.
#
# An MPO for a chain of L sites is a sequence of rank-4 tensors
# W_i[α, σ_out, σ_in, β], where α/β are the auxiliary (bond) indices and
# σ_out/σ_in are the physical (ket/bra) indices.
#
# Construction uses the finite-state-machine (W-matrix) method:
#   - Index 1       : "done" state (right boundary — sum accumulated)
#   - Indices 2..K+1: open channels, one per unique NN bond operator type
#   - Index K+2     : "start" state (left boundary — haven't begun)
# Bond dimension: χ = 2 + K.

"""
    FiniteMPO

A Matrix Product Operator (MPO) for a finite chain of `L` sites.

An MPO represents a linear operator on the full Hilbert space
``\\mathcal{H} = \\bigotimes_{i=1}^{L} \\mathcal{h}_i`` as a product of rank-4
tensors, one per site.  The tensor at site ``i`` has the index layout

```math
W_i[\\alpha,\\, \\sigma_{\\text{out}},\\, \\sigma_{\\text{in}},\\, \\beta]
```

where:
- ``\\alpha`` — left auxiliary (MPO bond) index of dimension ``\\chi_L``
- ``\\sigma_{\\text{out}}`` — outgoing (bra) physical index of dimension ``d``
- ``\\sigma_{\\text{in}}``  — incoming (ket) physical index of dimension ``d``
- ``\\beta`` — right auxiliary (MPO bond) index of dimension ``\\chi_R``

The full operator matrix is recovered by contracting all ``W_i`` along their
auxiliary indices with the boundary conditions ``\\alpha_0 = \\beta_L = 1``.

Boundary tensors have dimension 1 on their outer auxiliary side, so the stored
tensors at sites 1 and ``L`` have shapes ``(1, d, d, \\chi)`` and
``(\\chi, d, d, 1)`` respectively.

Construct a `FiniteMPO` from an [`Operator`](@ref) via [`MPO`](@ref).
Evaluate expectation values via [`expect`](@ref).

# Fields
- `tensors::Vector{Array{ComplexF64,4}}` — one ``(\\chi_L, d, d, \\chi_R)`` tensor
  per site.
- `d::Int` — local physical dimension (same for all sites).
- `L::Int` — number of sites.

See also: [`MPO`](@ref), [`expect`](@ref), [`Operator`](@ref)
"""
struct FiniteMPO
    tensors::Vector{Array{ComplexF64, 4}}   # (χ_L, d_out, d_in, χ_R) per site
    d::Int
    L::Int
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

# Two operator matrices are "the same type" if they are approximately equal.
_ops_eq(A, B) = size(A) == size(B) && maximum(abs, A - B) < 1e-12

# Collect unique (op_i, op_j) pairs from the bond term list (preserving order).
function _bond_types(bond_terms)
    types = Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}}[]
    for bt in bond_terms
        pair = (ComplexF64.(bt.op_i), ComplexF64.(bt.op_j))
        if !any(t -> _ops_eq(t[1], pair[1]) && _ops_eq(t[2], pair[2]), types)
            push!(types, pair)
        end
    end
    return types
end

# ─────────────────────────────────────────────────────────────────────────────
# MPO constructor from Operator
# ─────────────────────────────────────────────────────────────────────────────

"""
    MPO(H::Operator) -> FiniteMPO

Build a [`FiniteMPO`](@ref) from the term list of `H` using the
finite-state-machine (FSM) W-matrix method.

The FSM idea is elegant: think of building the operator left-to-right as a
finite automaton that carries a "state" summarising what has been started but
not yet finished.  The auxiliary index of the MPO tensor at site ``i`` labels
the current FSM state.  There are three kinds of state:

- **"done" state** (auxiliary index 1) — we have already accumulated a
  complete local operator string and just need to propagate the identity to
  the right.  The ``W`` entry `done → done` is the identity.
- **channel ``k``** (auxiliary indices ``2, \\ldots, K+1``) — the left operator
  ``O^{(k)}_i`` of bond type ``k`` was placed at some site to the left and we
  are now carrying it to its right partner.  The ``W`` entry `channel k → done`
  closes the channel by placing ``O^{(k)}_j``.
- **"start" state** (auxiliary index ``K+2``) — we have not placed any operator
  yet.  On-site terms and new bond openings come from this state.

For an `Operator` with ``K`` distinct nearest-neighbour bond-operator pairs
``(O^{(k)}_i, O^{(k)}_j)`` the interior bond dimension is

```math
\\chi = 2 + K.
```

The boundary tensors at sites 1 and ``L`` are sliced to dimension 1 on their
outer auxiliary side (only the "start" row is reachable from the left boundary;
only the "done" column is reachable from the right boundary).

**Special case:** if `H` has no terms at all (e.g. from [`identity_operator`](@ref)),
the result is a bond-dimension-1 all-identity MPO with ``\\chi = 1``.

# Arguments
- `H::Operator` — the operator to convert; produced by e.g. [`XXZ`](@ref),
  [`Heisenberg`](@ref), [`Ising`](@ref), or any of the observable constructors.

# Returns
- [`FiniteMPO`](@ref) with tensors of shape ``(\\chi_L, d, d, \\chi_R)`` per site.

# Examples
```jldoctest
julia> g = Chain(4);

julia> H = Heisenberg(g);

julia> W = MPO(H);

julia> W.L
4

julia> W.d
2

julia> size(W.tensors[2])   # interior site: full χ on both sides
(5, 2, 2, 5)
```
"""
function MPO(H::Operator)
    L  = H.geom.L
    d  = local_dim(H.dof)
    Id = Matrix{ComplexF64}(I, d, d)

    # Special case: empty operator (no terms) → bond-dim-1 all-identity MPO.
    # This represents I₁⊗I₂⊗…⊗I_L; its expectation value is ‖ψ‖².
    if isempty(H.onsite) && isempty(H.bond)
        tensors = [reshape(copy(Id), 1, d, d, 1) for _ in 1:L]
        return FiniteMPO(tensors, d, L)
    end

    # Identify the distinct bond operator types — these define the FSM channels.
    btypes = _bond_types(H.bond)
    K  = length(btypes)
    χ  = 2 + K          # full interior bond dimension
    # Index convention:
    #   1     = "done"  (right boundary state)
    #   2..K+1 = channel k (bond type k is open)
    #   K+2   = "start" (left boundary state)

    tensors = Vector{Array{ComplexF64, 4}}(undef, L)

    for i in 1:L
        W = zeros(ComplexF64, χ, d, d, χ)

        # Identity pass-through for both boundary states
        W[1,   :, :, 1]   = Id     # done → done
        W[K+2, :, :, K+2] = Id     # start → start

        # Onsite terms: start → done
        for lt in H.onsite
            lt.site == i || continue
            W[K+2, :, :, 1] .+= lt.coupling .* ComplexF64.(lt.op)
        end

        # Open channel k at site i (bond starting here): start → channel k
        for (k, (Ak, _)) in enumerate(btypes)
            for bt in H.bond
                bt.i == i                        || continue
                _ops_eq(ComplexF64.(bt.op_i), Ak) || continue
                W[K+2, :, :, k+1] .+= bt.coupling .* ComplexF64.(bt.op_i)
                break   # at most one bond starts at each site per type (NN chain)
            end
        end

        # Close channel k at site i (bond ending here): channel k → done
        for (k, (_, Bk)) in enumerate(btypes)
            for bt in H.bond
                bt.j == i                        || continue
                _ops_eq(ComplexF64.(bt.op_j), Bk) || continue
                W[k+1, :, :, 1] .+= ComplexF64.(bt.op_j)
                break
            end
        end

        tensors[i] = W
    end

    # Compress boundary bond dimensions: site 1 is only ever entered from
    # the "start" row; site L is only ever exited to the "done" column.
    tensors[1] = tensors[1][K+2:K+2, :, :, :]   # (1, d, d, χ)
    tensors[L] = tensors[L][:, :, :, 1:1]        # (χ, d, d, 1)

    FiniteMPO(tensors, d, L)
end

# ─────────────────────────────────────────────────────────────────────────────
# Expectation value ⟨ψ|O|ψ⟩  (§6.3)
# ─────────────────────────────────────────────────────────────────────────────

"""
    expect(ψ::FiniteMPS, O::FiniteMPO) -> ComplexF64

Compute the expectation value ``\\langle \\psi | O | \\psi \\rangle`` by a single
left-to-right sweep of a three-legged environment tensor.

The algorithm is the standard MPO–MPS "zipper" contraction.  Starting from a
``1 \\times 1 \\times 1`` scalar environment on the left boundary, at each site
``i`` the environment is updated by contracting with the MPS site tensor (bra
and ket) and the MPO tensor:

```math
E'[\\alpha', a', \\beta'] = \\sum_{\\alpha, a, \\beta, \\sigma, \\sigma'}
  E[\\alpha, a, \\beta]\\,
  \\overline{A_i[\\alpha, \\sigma, \\alpha']}\\,
  W_i[a, \\sigma, \\sigma', a']\\,
  A_i[\\beta, \\sigma', \\beta'],
```

where ``A_i`` is the MPS tensor at site ``i`` and bar denotes complex conjugate.

The environment tensor has shape ``(\\chi_{\\text{bra}}, \\chi_{\\text{mpo}}, \\chi_{\\text{ket}})``
throughout the sweep.  After all ``L`` sites the environment is a ``1 \\times 1 \\times 1``
scalar equal to ``\\langle \\psi | O | \\psi \\rangle``.

The per-site cost is ``O(\\chi^2 d \\chi_{\\text{mpo}})`` where ``\\chi`` is the MPS
bond dimension; the total cost for the full sweep is
``O(L \\chi^2 d \\chi_{\\text{mpo}})``.

# Arguments
- `ψ::FiniteMPS`   — the MPS state; does not need to be normalised.
- `O::FiniteMPO`   — the MPO operator; must have the same `L` and `d` as `ψ`.

# Returns
- `ComplexF64` — the expectation value ``\\langle \\psi | O | \\psi \\rangle``.
  For Hermitian operators and real states this should have a negligible imaginary
  part.

# Examples
```jldoctest
julia> using LinearAlgebra

julia> g = Chain(2);

julia> H = Heisenberg(g);

julia> W = MPO(H);

julia> ψ = to_mps(normalize([1.0, 0.0, 0.0, -1.0] / √2));

julia> isapprox(real(expect(ψ, W)), -0.75, atol=1e-10)
true
```
"""
function expect(ψ::FiniteMPS, O::FiniteMPO)
    L = O.L
    # Initialise: 1×1×1 environment = scalar 1
    env = ones(ComplexF64, 1, 1, 1)   # (χ_bra, χ_mpo, χ_ket)

    for i in 1:L
        A = ψ.tensors[i].data   # (χ_L, d, χ_R)
        W = O.tensors[i]        # (χ_mpo_L, d_out, d_in, χ_mpo_R)
        # Contract: env[α,a,β] conj(A[α,σ,α']) W[a,σ,σ',a'] A[β,σ',β'] → new_env[α',a',β']
        @tensor new_env[α′, a′, β′] :=
            env[α, a, β] * conj(A[α, σ, α′]) * W[a, σ, σ′, a′] * A[β, σ′, β′]
        env = new_env
    end

    return env[1, 1, 1]
end

# ─────────────────────────────────────────────────────────────────────────────
# MPO application: O|ψ⟩  (§6.3 / §8)
# ─────────────────────────────────────────────────────────────────────────────

"""
    apply_mpo(O::FiniteMPO, ψ::FiniteMPS; trunc=NoTrunc()) -> FiniteMPS

Apply the MPO `O` to the MPS `ψ`, returning the MPS representation of `O|ψ⟩`.

At each site the MPO tensor `W[a, σ_out, σ_in, b]` is contracted with the MPS
site tensor `A[α, σ_in, β]`, fusing the auxiliary indices `(a, α) → aα` and
`(b, β) → bβ` to produce a "thick" MPS with bond dimension `χ_mpo × χ_mps`.
The result is then compressed back to `trunc` via a left-canonical SVD sweep.

For systems with `d^L ≤ ~ 10^5` (i.e., `L ≤ 16` for `d=2`) the implementation
contracts to a full dense state vector first and then calls `to_mps`.  This is
exact (no intermediate truncation error) and simple to verify.
"""
function apply_mpo(O::FiniteMPO, ψ::FiniteMPS; trunc::AbstractTrunc=NoTrunc())
    L = O.L
    d = O.d

    # Zip contraction: build thick MPS tensors (χ_mpo*χ_mps bonds) site by site,
    # then do a single left-canonical SVD sweep with the requested truncation.
    thick = Vector{Array{ComplexF64, 3}}(undef, L)

    for i in 1:L
        A = ψ.tensors[i].data   # (χ_L, d, χ_R)
        W = O.tensors[i]        # (χ_mpo_L, d_out, d_in, χ_mpo_R)
        χ_L, _, χ_R       = size(A)
        χW_L, _, _, χW_R  = size(W)

        # Contract over physical index d_in=σ, fuse auxiliary bonds:
        # new[(a,α), σ_out, (b,β)] = Σ_σ W[a,σ_out,σ,b] * A[α,σ,β]
        # Use @tensor then reshape to fuse (a,α) and (b,β).
        @tensor Θ[a, α, σo, b, β] := W[a, σo, σ, b] * A[α, σ, β]
        # Fuse: reshape (χW_L, χ_L, d, χW_R, χ_R) → (χW_L*χ_L, d, χW_R*χ_R)
        thick[i] = reshape(Θ, χW_L * χ_L, d, χW_R * χ_R)
    end

    # Compress via left-canonical SVD sweep (reuse MPS machinery directly)
    _compress_thick_mps(thick, d, L, trunc)
end

# Left-canonical SVD sweep on thick MPS tensors (mirrors _left_sweep_mps! in canonicalize.jl).
function _compress_thick_mps(tensors::Vector{Array{ComplexF64,3}}, d::Int, L::Int,
                               trunc::AbstractTrunc)
    result   = Vector{QTensor}(undef, L)
    bond_svs = Vector{SingValSpectrum}(undef, L + 1)
    ε_total  = 0.0

    bond_svs[1] = SingValSpectrum([1.0], 0.0, true)

    carry  = tensors[1]
    χ_left = size(carry, 1)

    for i in 1:(L - 1)
        M = reshape(carry, χ_left * d, size(carry, 3))

        F       = svd(M)
        tol     = length(F.S) * eps(eltype(F.S)) * (isempty(F.S) ? 1.0 : F.S[1])
        S_clean = filter(s -> s > tol, F.S)
        r, ε_bond = _truncate_singular_values(S_clean, trunc)
        svs     = F.S[1:r]
        ε_total += ε_bond

        normalized    = isapprox(sum(abs2, svs), 1.0; atol=sqrt(eps(eltype(svs))))
        bond_svs[i+1] = SingValSpectrum(svs, ε_bond, normalized)
        result[i]     = QTensor(reshape(F.U[:, 1:r], χ_left, d, r),
                                (upper(:vL, χ_left), lower(:σ, d), lower(:vR, r)))

        χ_next_L, _, χ_next_R = size(tensors[i+1])
        carry  = reshape(Diagonal(svs) * F.Vt[1:r, :] *
                         reshape(tensors[i+1], χ_next_L, d * χ_next_R), r, d, χ_next_R)
        χ_left = r
    end

    χ_R = size(carry, 3)
    result[L]     = QTensor(carry, (upper(:vL, χ_left), lower(:σ, d), lower(:vR, χ_R)))
    bond_svs[L+1] = SingValSpectrum([1.0], 0.0, true)

    FiniteMPS(result, bond_svs, CanonicalForm(L, L + 1), ε_total)
end
