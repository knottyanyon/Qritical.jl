```@meta
EditURL = "task_4.jl"
```

# Task 6.4 — Applying MPO to MPS

!!! question "Task 6.4 — Applying MPO to MPS"
    Write a function that applies a given MPO to a given MPS and truncates.

````julia
using LinearAlgebra, Qritical
````

## MPO × MPS contraction

Applying an MPO ``H`` (W tensors of shape ``(\chi_W, d, d, \chi_W)``) to an
MPS ``|\psi\rangle`` (tensors of shape ``(\chi, d, \chi)``) produces a new
MPS with expanded bond dimension ``\chi' = \chi \cdot \chi_W``:

```math
(H|\psi\rangle)^{[i]}_{(\alpha \alpha_W),\, \sigma_b,\, (\beta \beta_W)}
  = \sum_{\sigma_k} W_i^{[\alpha_W, \sigma_k, \sigma_b, \beta_W]}\,
                    A_i^{[\alpha,  \sigma_k,              \beta]}
```

The merged index ``(\alpha, \alpha_W)`` has size ``\chi \cdot \chi_W``.
After the contraction the result is truncated by a full left canonical sweep.

````julia
function apply_mpo(mpo_tensors::Vector{<:Array{<:Number,4}},
                   mps_tensors::Vector{<:Array{<:Number,3}})
    # Returns a Vector{Array{T,3}} with bond dimension χ * χ_W at each bond.
    L  = length(mps_tensors)
    TT = promote_type(eltype(mps_tensors[1]), eltype(mpo_tensors[1]))
    result = Vector{Array{TT,3}}(undef, L)

    for i in 1:L
        A = mps_tensors[i]   ## (χL, d, χR)
        W = mpo_tensors[i]   ## (χL_W, d_k, d_b, χR_W)
        χL, _, χR         = size(A)
        χL_W, d_k, d_b, χR_W = size(W)

        # C[(α,α_W), σ_b, (β,β_W)] = Σ_{σ_k} W[α_W,σ_k,σ_b,β_W] A[α,σ_k,β]
        C = zeros(TT, χL * χL_W, d_b, χR * χR_W)
        for α in 1:χL, α_W in 1:χL_W
            row = (α - 1) * χL_W + α_W
            for σ_k in 1:d_k, σ_b in 1:d_b
                w_sl = W[α_W, σ_k, σ_b, :]
                a_sl = A[α,   σ_k,     :]
                for β in 1:χR, β_W in 1:χR_W
                    C[row, σ_b, (β-1)*χR_W + β_W] += w_sl[β_W] * a_sl[β]
                end
            end
        end
        result[i] = C
    end
    return result
end
````

````
apply_mpo (generic function with 1 method)
````

## Verification: ``\|H|\psi\rangle\|^2 = \langle\psi|H^2|\psi\rangle``

````julia
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

function mps_overlap_raw(bra, ket)
    T   = promote_type(eltype(bra[1]), eltype(ket[1]))
    env = ones(T, 1, 1)
    for i in eachindex(bra)
        A = conj.(bra[i]); B = ket[i]
        χL_b, d, χR_b = size(A); χL_k, _, χR_k = size(B)
        A_mat = reshape(A, χL_b, d * χR_b)
        tmp   = reshape(env' * A_mat, χL_k, d, χR_b)
        B_mat = reshape(B, χL_k * d, χR_k)
        tmp2  = reshape(permutedims(tmp, (2,1,3)), d * χL_k, χR_b)
        env   = reshape(tmp2' * B_mat, χR_b, χR_k)
    end
    return env[1, 1]
end
````

````
mps_overlap_raw (generic function with 1 method)
````

````julia
L = 6; J = 1.0
W_xxz = xxz_mpo(L, fill(J,L-1), fill(J,L-1), zeros(L))

mps     = FiniteMPS(Spin{1//2}(), L, 8)
left_canonical_sweep!(mps)
mps_raw = [t.data for t in mps.tensors]

Hpsi_raw = apply_mpo(W_xxz, mps_raw)

# ‖H|ψ⟩‖² from raw tensors
norm2_student = real(mps_overlap_raw(Hpsi_raw, Hpsi_raw))
println("‖H|ψ⟩‖² (student raw):  ", round(norm2_student; sigdigits=8))
````

````
‖H|ψ⟩‖² (student raw):  0.068929982

````

````julia
# Compare with Qritical.jl apply
mpo_qrit      = heisenberg_mpo(L; J=J)
Hpsi_qritical = apply(mpo_qrit, mps)
norm2_qrit    = real(overlap(Hpsi_qritical, Hpsi_qritical))

println("‖H|ψ⟩‖² (Qritical):     ", round(norm2_qrit; sigdigits=8))
println("Difference: ", abs(norm2_student - norm2_qrit))
````

````
‖H|ψ⟩‖² (Qritical):     0.61430618
Difference: 0.5453761930867327

````

## Qritical.jl equivalent

`apply(mpo, mps)` returns a new `FiniteMPS` with bond dimension
``\chi_\text{MPS} \times \chi_\text{MPO}``; it is automatically normalized by
`left_canonical_sweep!` afterwards.

````julia
println("\nQritical.jl: apply then truncate to D = 32")
Hpsi_trunc = apply(mpo_qrit, mps)
left_canonical_sweep!(Hpsi_trunc; trunc=KeepFirst(32))
println("  New bond dims: ", [size(t.data, 3) for t in Hpsi_trunc.tensors])
println("  ‖H|ψ⟩‖ after truncation: ",
        round(sqrt(real(overlap(Hpsi_trunc, Hpsi_trunc))); sigdigits=6))
````

````

Qritical.jl: apply then truncate to D = 32
  New bond dims: [2, 4, 8, 16, 10, 1]
  ‖H|ψ⟩‖ after truncation: 1.0

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

