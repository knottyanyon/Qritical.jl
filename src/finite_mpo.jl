using LinearAlgebra: kron, I

# ── FiniteMPO ─────────────────────────────────────────────────────────────────

"""
    FiniteMPO{D <: AbstractDoF, T <: Number}

Finite matrix product operator on `L` sites.

Each site tensor `tensors[i]` has four legs `(vL, σ_ket, σ_bra, vR)`:
- `vL`:    left virtual bond  — `TIx{Upper}`, inward from left
- `σ_ket`: ket physical leg   — `TIx{Lower}`, input index (contracts with MPS σ)
- `σ_bra`: bra physical leg   — `TIx{Upper}`, output index (becomes new MPS σ)
- `vR`:    right virtual bond — `TIx{Lower}`, outward to right

`IdL` and `IdR` are the 1-based virtual-bond indices initialising the left and
right boundary environments respectively.
"""
mutable struct FiniteMPO{D <: AbstractDoF, T <: Number} <: AbstractMPS
    L       :: Int
    tensors :: Vector{IndexedTensor{T, 4}}
    IdL     :: Int
    IdR     :: Int
end

Base.length(mpo::FiniteMPO) = mpo.L

# ── _mpo_tensor: attach index metadata to a raw W array ──────────────────────

function _mpo_tensor(data::Array{T,4}, i::Int, L::Int) where {T}
    χL, d_ket, d_bra, χR = size(data)
    vL     = upper(bond_label(:w, i - 1), χL)
    σ_ket  = lower(:σ, d_ket)
    σ_bra  = upper(:σ̄, d_bra)
    vR     = lower(bond_label(:w, i),     χR)
    return IndexedTensor(data, (vL, σ_ket, σ_bra, vR))
end

# ── heisenberg_mpo ────────────────────────────────────────────────────────────

"""
    heisenberg_mpo(L; J=1.0, T=Float64)

Build the `L`-site spin-1/2 Heisenberg MPO for

```math
H = J \\sum_{i=1}^{L-1} \\left( S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + S^z_i S^z_{i+1} \\right)
  = \\frac{J}{2} \\sum_i \\left( S^+_i S^-_{i+1} + S^-_i S^+_{i+1} \\right)
  + J \\sum_i S^z_i S^z_{i+1}
```

The virtual bond dimension is 5 (finite-state-machine with 3 coupling channels).
`IdL = 1`, `IdR = 5`.
"""
function heisenberg_mpo(L::Int; J::Real=1.0, T::Type{<:Number}=Float64)
    d  = hilbert_space(Spin{1//2}())   # = 2
    χ  = 5
    Jt = T(J)

    # Spin-1/2 operators (d×d), basis |↑⟩=1, |↓⟩=2
    Id  = Matrix{T}(I, d, d)
    Sz  = T[1/2  0; 0  -1/2]
    Sp  = T[0  1; 0  0]        # S^+
    Sm  = T[0  0; 1  0]        # S^-

    # Interior W tensor: W[α, σ_ket, σ_bra, β] (1-indexed)
    # FSM channels: 1=IdL, 2=S^+, 3=S^-, 4=S^z, 5=IdR
    function make_W()
        W = zeros(T, χ, d, d, χ)
        for σ in 1:d, σ′ in 1:d
            W[1, σ, σ′, 1] = Id[σ′, σ]          # identity propagates left
            W[1, σ, σ′, 2] = Sp[σ′, σ]          # start S^+
            W[1, σ, σ′, 3] = Sm[σ′, σ]          # start S^-
            W[1, σ, σ′, 4] = Sz[σ′, σ]          # start S^z
            W[2, σ, σ′, 5] = (Jt/2) * Sm[σ′, σ] # complete S^+ ⋅ S^-
            W[3, σ, σ′, 5] = (Jt/2) * Sp[σ′, σ] # complete S^- ⋅ S^+
            W[4, σ, σ′, 5] = Jt     * Sz[σ′, σ] # complete S^z ⋅ S^z
            W[5, σ, σ′, 5] = Id[σ′, σ]          # identity propagates right
        end
        return W
    end

    W = make_W()

    tensors = Vector{IndexedTensor{T, 4}}(undef, L)
    for i in 1:L
        if L == 1
            # trivial single-site: take W[IdL=1, :, :, IdR=5] reshaped
            data = reshape(W[1:1, :, :, 5:5], 1, d, d, 1)
        elseif i == 1
            # left boundary: fix left virtual to IdL=1
            data = reshape(W[1:1, :, :, :], 1, d, d, χ)
        elseif i == L
            # right boundary: fix right virtual to IdR=5
            data = reshape(W[:, :, :, 5:5], χ, d, d, 1)
        else
            data = copy(W)
        end
        tensors[i] = _mpo_tensor(data, i, L)
    end

    return FiniteMPO{Spin{1//2}, T}(L, tensors, 1, 5)
end

# ── identity_mpo ──────────────────────────────────────────────────────────────

"""
    identity_mpo(dof, L; T=Float64)

Build the `L`-site identity MPO ``\\hat{I}`` with virtual bond dimension 1.
`expectation_value(mps, identity_mpo(dof, L)) == overlap(mps, mps)`.
"""
function identity_mpo(::D, L::Int; T::Type{<:Number}=Float64) where {D <: AbstractDoF}
    d  = hilbert_space(D())
    Id = Matrix{T}(I, d, d)
    tensors = Vector{IndexedTensor{T, 4}}(undef, L)
    for i in 1:L
        data = zeros(T, 1, d, d, 1)
        for σ in 1:d, σ′ in 1:d
            data[1, σ, σ′, 1] = Id[σ′, σ]
        end
        tensors[i] = _mpo_tensor(data, i, L)
    end
    return FiniteMPO{D, T}(L, tensors, 1, 1)
end

# ── expectation_value ─────────────────────────────────────────────────────────

"""
    expectation_value(mps, mpo) -> Real

Compute ``\\langle\\psi|\\hat{O}|\\psi\\rangle`` by boundary-to-boundary
environment contraction.  Works for any `FiniteMPS` / `FiniteMPO` pair with
the same length and physical dimension.

The environment tensor `env` has shape ``(\\chi_\\text{bra}, \\chi_W, \\chi_\\text{ket})``.
"""
function expectation_value(mps::FiniteMPS, mpo::FiniteMPO)
    mps.L == mpo.L || throw(ArgumentError("MPS and MPO lengths differ"))
    L = mps.L

    # Initialize left boundary environment: shape (1, 1, 1)
    T   = promote_type(eltype(mps.tensors[1]), eltype(mpo.tensors[1]))
    env = ones(T, 1, 1, 1)   # (χ_bra, χ_W, χ_ket)

    for i in 1:L
        A = conj.(mps.tensors[i].data)   # (χL_bra, d, χR_bra)  — bra
        W = mpo.tensors[i].data           # (χL_W, d_ket, d_bra, χR_W)
        B = mps.tensors[i].data           # (χL_ket, d, χR_ket)  — ket

        χL_b, d,     χR_b = size(A)
        χL_W, d_k, d_bra, χR_W = size(W)
        χL_k, _,   χR_k = size(B)

        # Contract env (χL_b × χL_W × χL_k) with A, W, B over left bonds and physical legs.
        # New env has shape (χR_b × χR_W × χR_k).
        new_env = zeros(T, χR_b, χR_W, χR_k)
        for α_b in 1:χL_b, α_W in 1:χL_W, α_k in 1:χL_k
            e = env[α_b, α_W, α_k]
            iszero(e) && continue
            for σ_k in 1:d_k, σ_b in 1:d_bra
                w = W[α_W, σ_k, σ_b, :]   # length χR_W
                a = A[α_b, σ_b, :]         # length χR_b  (bra uses σ_bra)
                b = B[α_k, σ_k, :]         # length χR_k  (ket uses σ_ket)
                for β_b in 1:χR_b, β_W in 1:χR_W, β_k in 1:χR_k
                    new_env[β_b, β_W, β_k] += e * a[β_b] * w[β_W] * b[β_k]
                end
            end
        end
        env = new_env
    end

    return real(env[1, 1, 1])
end

# ── apply ─────────────────────────────────────────────────────────────────────

"""
    apply(mpo, mps) -> FiniteMPS

Apply `mpo` to `mps`, returning a new `FiniteMPS` with bond dimension
``\\chi_\\text{MPS} \\times \\chi_\\text{MPO}`` at each interior bond.
"""
function apply(mpo::FiniteMPO{D, TW}, mps::FiniteMPS{D, TA, RTA}) where {D, TW, TA, RTA}
    mpo.L == mps.L || throw(ArgumentError("MPO and MPS lengths differ"))
    L  = mps.L
    T  = promote_type(TW, TA)
    RT = real(T)
    d  = hilbert_space(D())

    tensors  = Vector{IndexedTensor{T, 3}}(undef, L)
    bond_svs = Vector{Vector{RT}}(undef, L + 1)
    bond_svs[1]     = RT[1]
    bond_svs[L + 1] = RT[1]

    for i in 1:L
        W = mpo.tensors[i].data   # (χL_W, d_k, d_b, χR_W)
        A = mps.tensors[i].data   # (χL_A, d,   χR_A)

        χL_W, d_k, d_b, χR_W = size(W)
        χL_A, _,        χR_A = size(A)

        # New tensor: shape (χL_W*χL_A, d_b, χR_W*χR_A)
        new_data = zeros(T, χL_W * χL_A, d_b, χR_W * χR_A)

        for α_W in 1:χL_W, α_A in 1:χL_A
            for σ_b in 1:d_b
                for σ_k in 1:d_k
                    w_row = W[α_W, σ_k, σ_b, :]   # length χR_W
                    a_row = A[α_A, σ_k, :]          # length χR_A
                    for β_W in 1:χR_W, β_A in 1:χR_A
                        new_data[(α_W-1)*χL_A + α_A, σ_b,
                                 (β_W-1)*χR_A + β_A] +=
                            w_row[β_W] * a_row[β_A]
                    end
                end
            end
        end

        χL_new = χL_W * χL_A
        χR_new = χR_W * χR_A
        vL = upper(bond_label(:χ, i - 1), χL_new)
        σ  = lower(:σ, d_b)
        vR = lower(bond_label(:χ, i),     χR_new)
        tensors[i] = IndexedTensor(new_data, (vL, σ, vR))

        if i in 2:L
            bond_svs[i] = fill(RT(NaN), χL_new)
        end
    end

    return FiniteMPS{D, T, RT}(L, tensors, bond_svs, ArbitraryForm())
end
