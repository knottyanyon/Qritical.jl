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

Build a Matrix Product Operator from the term list of `H` using the
finite-state-machine W-matrix method.

For a Hamiltonian with K distinct nearest-neighbour bond operator types the
resulting bond dimension is χ = 2 + K (plus the two boundary states).
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

Compute ⟨ψ|O|ψ⟩ by sweeping the left environment tensor from site 1 to L.

The left environment at each step has shape `(χ_bra, χ_mpo, χ_ket)`.
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
