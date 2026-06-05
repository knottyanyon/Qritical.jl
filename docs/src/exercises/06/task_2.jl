# # Task 6.2 — MPO II: all-to-all ``S^z S^z``

# !!! question "Task 6.2 — MPO II"
#     Write a function that returns the MPO of
#
#     ```math
#     M = \sum_{i=1}^{L} \sum_{j=1}^{L} S_i^z S_j^z = \left(\sum_i S_i^z\right)^2
#     ```

using LinearAlgebra, Qritical
#--

# ## Construction via bond-dimension 3 FSM
#
# ``M = \hat S_\text{tot}^z \cdot \hat S_\text{tot}^z`` can be represented as
# an MPO with virtual bond dimension 3:
#
# | State | Meaning |
# |-------|---------|
# | 1 = IdL | no ``S^z`` applied yet |
# | 2 = Sz  | one ``S^z`` started; waiting for the second at a later site |
# | 3 = IdR | both ``S^z``s have been applied |
#
# The non-zero W blocks at interior site ``i`` are:
#
# ```math
# W[1, \sigma, \sigma', 1] = I, \qquad
# W[1, \sigma, \sigma', 2] = 2 S^z, \qquad
# W[1, \sigma, \sigma', 3] = (S^z)^2,
# ```
# ```math
# W[2, \sigma, \sigma', 2] = I, \qquad
# W[2, \sigma, \sigma', 3] = S^z, \qquad
# W[3, \sigma, \sigma', 3] = I.
# ```
#
# The factor **2** on ``W[1 \to 2]`` ensures both orderings ``(i,j)`` and
# ``(j,i)`` are counted, while ``W[1 \to 3] = (S^z)^2 = I/4`` handles the
# diagonal terms ``i = j``.
#
# **Verification for ``L = 2``:**
# The three FSM paths from IdL to IdR give:
# - ``1 \to 1 \to 3``: ``I \otimes (S^z)^2``  (diagonal at site 2)
# - ``1 \to 2 \to 3``: ``2 S^z \otimes S^z``  (off-diagonal, both orderings)
# - ``1 \to 3 \to 3``: ``(S^z)^2 \otimes I``  (diagonal at site 1)
#
# Sum = ``(S_1^z)^2 + 2 S_1^z S_2^z + (S_2^z)^2 = \sum_{ij} S_i^z S_j^z`` ✓

function sz_sz_mpo(L::Int; T::Type{<:Number}=Float64)
    ## Returns Vector{Array{T,4}} of W tensors with shape (χL, d_ket, d_bra, χR).
    ## Bond dimension is 3.  IdL = 1, IdR = 3.
    d   = 2
    χ   = 3
    Id  = Matrix{T}(I, d, d)
    Sz  = T[1//2 0; 0 -1//2]
    Sz2 = Sz * Sz   ## = I/4 for spin-1/2

    tensors = Vector{Array{T,4}}(undef, L)
    for i in 1:L
        W = zeros(T, χ, d, d, χ)
        for σ in 1:d, σ′ in 1:d
            W[1, σ, σ′, 1] = Id[σ′, σ]
            W[1, σ, σ′, 2] = 2 * Sz[σ′, σ]   ## factor 2: both orderings (i,j) and (j,i)
            W[1, σ, σ′, 3] = Sz2[σ′, σ]       ## diagonal (S^z)^2 = I/4
            W[2, σ, σ′, 2] = Id[σ′, σ]
            W[2, σ, σ′, 3] = Sz[σ′, σ]
            W[3, σ, σ′, 3] = Id[σ′, σ]
        end

        tensors[i] = if L == 1
            reshape(W[1:1, :, :, 3:3], 1, d, d, 1)
        elseif i == 1
            reshape(W[1:1, :, :, :], 1, d, d, χ)
        elseif i == L
            reshape(W[:, :, :, 3:3], χ, d, d, 1)
        else
            copy(W)
        end
    end
    return tensors
end
#--

# ## Verify: all-up state ``|↑↑…↑⟩``
#
# For the all-up state ``|\psi\rangle = |↑↑\cdots↑\rangle``:
#
# ```math
# \langle M \rangle
#   = \left(\sum_i \langle S_i^z \rangle\right)^2
#   = \left(\frac{L}{2}\right)^2 = \frac{L^2}{4}
# ```

L = 6
M_tensors = sz_sz_mpo(L)
println("Bond dimensions: ", [size(W, 1) for W in M_tensors], " → ", size(M_tensors[end], 4))
println("Interior W shape: ", size(M_tensors[3]))
#--

# Build the all-up MPS ``|↑↑…↑⟩`` and contract ⟨ψ|M|ψ⟩ by hand.
# We use a simple environment sweep matching the W tensor convention.

function contract_expectation(mps_tensors, mpo_tensors)
    T   = promote_type(eltype(mps_tensors[1]), eltype(mpo_tensors[1]))
    env = ones(T, 1, 1, 1)   ## (χL_bra, χL_W, χL_ket)
    for i in eachindex(mps_tensors)
        A = conj.(mps_tensors[i])   ## bra: (χL_b, d, χR_b)
        W = mpo_tensors[i]          ## (χL_W, d_k, d_b, χR_W)
        B = mps_tensors[i]          ## ket: (χL_k, d, χR_k)
        χL_b, _, χR_b = size(A)
        χL_W, d_k, _, χR_W = size(W)
        χL_k, _, χR_k = size(B)
        new_env = zeros(T, χR_b, χR_W, χR_k)
        for α_b in 1:χL_b, α_W in 1:χL_W, α_k in 1:χL_k
            e = env[α_b, α_W, α_k]
            iszero(e) && continue
            for σ_k in 1:d_k
                for σ_b in 1:size(W,3)
                    w = W[α_W, σ_k, σ_b, :]
                    a = A[α_b, σ_b, :]
                    b = B[α_k, σ_k, :]
                    for β_b in 1:χR_b, β_W in 1:χR_W, β_k in 1:χR_k
                        new_env[β_b, β_W, β_k] += e * a[β_b] * w[β_W] * b[β_k]
                    end
                end
            end
        end
        env = new_env
    end
    return real(env[1, 1, 1])
end
#--

## All-up state: one-hot tensors with σ=1 (↑), bond dim 1 everywhere
all_up = [reshape([1.0, 0.0], 1, 2, 1) for _ in 1:L]
## Normalize: ⟨ψ|ψ⟩ = 1 already for product state with one-hot entries

expected = (L / 2)^2
result   = contract_expectation(all_up, M_tensors)

println("\n⟨↑↑…↑|M|↑↑…↑⟩ = ", round(result; sigdigits=6))
println("Expected (L/2)² = ", expected)
println("Error: ", abs(result - expected))
#--

# ## Néel state ``|↑↓↑↓…⟩`` verification
#
# For the alternating Néel state, total ``S^z_\text{tot} = 0`` (for even ``L``),
# so ``\langle M \rangle = 0``.

neel = [reshape(isodd(i) ? [1.0, 0.0] : [0.0, 1.0], 1, 2, 1) for i in 1:L]
neel_result = contract_expectation(neel, M_tensors)
println("\n⟨Néel|M|Néel⟩ = ", round(neel_result; sigdigits=6), "  (expected 0)")
#--
