# # Task 6.3 — Expectation value of MPO

# !!! question "Task 6.3 — Expectation Value of MPO"
#     Write a function that evaluates the expectation value of a given MPO
#     w.r.t. a given MPS.  Test your routine for the MPO of Eq. (1) with
#     ``h_i = J_i^z = 0`` and ``J_i = 1.0``.  You can also test other
#     parameters or use the MPO of Eq. (2) and compare to colleagues.

using LinearAlgebra, Qritical
#--

# ## Environment contraction
#
# For an MPS ``|\psi\rangle`` with tensors ``A_i`` and an MPO ``H`` with
# W tensors ``W_i``, the expectation value
#
# ```math
# \langle\psi|H|\psi\rangle
#   = \operatorname{tr}\!\left(E^{[0]} \cdot T^{[1]} \cdot T^{[2]} \cdots T^{[L]}\right)
# ```
#
# is built by sweeping a **left environment** ``E^{[i]}`` from left to right:
#
# ```math
# E^{[i]}_{\beta_b \beta_W \beta_k}
#   = \sum_{\alpha_b \alpha_W \alpha_k \sigma \sigma'}
#     E^{[i-1]}_{\alpha_b \alpha_W \alpha_k}
#     \,\bar A_i^*_{[\alpha_b, \sigma', \beta_b]}
#     \, W_i^{[\alpha_W, \sigma, \sigma', \beta_W]}
#     \, A_i^{[\alpha_k, \sigma, \beta_k]}
# ```
#
# The environment has three virtual legs: bra (from ``A^*``), MPO (from ``W``),
# and ket (from ``A``).  At the left boundary it is ``E^{[0]} = [1]``; at the
# right boundary ``\langle\psi|H|\psi\rangle = E^{[L]}_{1,1,1}``.

function expectation_value_mpo(mps::FiniteMPS, W_tensors::Vector{<:Array{<:Number,4}})
    T   = promote_type(eltype(mps.tensors[1]), eltype(W_tensors[1]))
    env = ones(T, 1, 1, 1)   ## (χL_bra, χL_W, χL_ket)

    for i in eachindex(W_tensors)
        A = conj.(mps.tensors[i].data)   ## bra
        W = W_tensors[i]                 ## (χL_W, d_k, d_b, χR_W)
        B = mps.tensors[i].data          ## ket
        χL_b, _, χR_b = size(A)
        χL_W, d_k, d_b, χR_W = size(W)
        χL_k, _, χR_k = size(B)

        new_env = zeros(T, χR_b, χR_W, χR_k)
        for α_b in 1:χL_b, α_W in 1:χL_W, α_k in 1:χL_k
            e = env[α_b, α_W, α_k]
            iszero(e) && continue
            for σ_k in 1:d_k, σ_b in 1:d_b
                w = W[α_W, σ_k, σ_b, :]
                a = A[α_b, σ_b, :]
                b = B[α_k, σ_k, :]
                for β_b in 1:χR_b, β_W in 1:χR_W, β_k in 1:χR_k
                    new_env[β_b, β_W, β_k] += e * a[β_b] * w[β_W] * b[β_k]
                end
            end
        end
        env = new_env
    end

    return real(env[1, 1, 1])
end
#--

# ## Setup: XXZ MPO and random MPS

## Reuse xxz_mpo from Task 6.1
function xxz_mpo(L, Js, Jzs, hs; T=Float64)
    d = 2; χ = 5
    Id = Matrix{T}(I, d, d); Sz = T[1//2 0; 0 -1//2]
    Sp = T[0 1; 0 0]; Sm = T[0 0; 1 0]
    tensors = Vector{Array{T,4}}(undef, L)
    for i in 1:L
        W = zeros(T, χ, d, d, χ)
        J_L  = i > 1 ? T(Js[i-1])  : zero(T)
        Jz_L = i > 1 ? T(Jzs[i-1]) : zero(T)
        h_i  = T(hs[i])
        for σ in 1:d, σ′ in 1:d
            W[1, σ, σ′, 1] = Id[σ′, σ]
            W[1, σ, σ′, 2] = Sp[σ′, σ]; W[1, σ, σ′, 3] = Sm[σ′, σ]
            W[1, σ, σ′, 4] = Sz[σ′, σ]; W[1, σ, σ′, 5] = -h_i * Sz[σ′, σ]
            W[2, σ, σ′, 5] = (J_L/2) * Sm[σ′, σ]; W[3, σ, σ′, 5] = (J_L/2) * Sp[σ′, σ]
            W[4, σ, σ′, 5] = Jz_L * Sz[σ′, σ]; W[5, σ, σ′, 5] = Id[σ′, σ]
        end
        tensors[i] = L==1 ? reshape(W[1:1,:,:,5:5],1,d,d,1) :
                     i==1 ? reshape(W[1:1,:,:,:],1,d,d,χ) :
                     i==L ? reshape(W[:,:,:,5:5],χ,d,d,1) : copy(W)
    end
    return tensors
end

L = 8; χ_mps = 16; J = 1.0
W_xxz = xxz_mpo(L, fill(J,L-1), fill(J,L-1), zeros(L))
mps   = FiniteMPS(Spin{1//2}(), L, χ_mps)
left_canonical_sweep!(mps)
#--

# ## Compare student vs Qritical.jl

E_student  = expectation_value_mpo(mps, W_xxz)
mpo_qrit   = heisenberg_mpo(L; J=J)
E_qritical = expectation_value(mps, mpo_qrit)

println("⟨H⟩  (student):    ", round(E_student;  sigdigits=8))
println("⟨H⟩  (Qritical):   ", round(E_qritical; sigdigits=8))
println("Difference:         ", abs(E_student - E_qritical))
#--

# ## Physical test: identity MPO
#
# ``\langle\psi|\hat{I}|\psi\rangle = \langle\psi|\psi\rangle``.
# Build the identity MPO as a 1-dimensional MPO (``\chi = 1``):

## Identity MPO: W[1, σ_ket, σ_bra, 1] = δ_{σ_ket, σ_bra}
Id_tensor = zeros(1, 2, 2, 1)
Id_tensor[1, 1, 1, 1] = 1.0
Id_tensor[1, 2, 2, 1] = 1.0
Id_W = [copy(Id_tensor) for _ in 1:L]

E_Id  = expectation_value_mpo(mps, Id_W)
norm2 = real(overlap(mps, mps))
println("\n⟨ψ|I|ψ⟩  = ", round(E_Id; sigdigits=8), "  ⟨ψ|ψ⟩ = ", round(norm2; sigdigits=8))
#--
