```@meta
EditURL = "task_1.jl"
```

# Task 4.2 — MPS Overlap

!!! question "Task 4.2 — MPS Overlap"
    Write a function that receives two MPS of equal length and returns their
    overlap ``\langle \psi | \phi \rangle`` in an efficient way.

````julia
using Serialization, LinearAlgebra, Qritical
````

````julia
DATA_ROOT  = normpath(joinpath(@__FILE__, ".."))
FPATH_PSI1 = normpath(joinpath(DATA_ROOT, "psi1.jls"))
FPATH_PSI2 = normpath(joinpath(DATA_ROOT, "psi2.jls"))

ψ1_raw = deserialize(FPATH_PSI1)
ψ2_raw = deserialize(FPATH_PSI2)
L = ndims(ψ1_raw)
````

````
12
````

## Why a sweep and not a full contraction?

The naive approach contracts the full ``2^L``-dimensional state vectors
directly: ``\langle\psi|\phi\rangle = \sum_{\sigma_1\ldots\sigma_L}
\psi^*_{\sigma_1\ldots\sigma_L} \phi_{\sigma_1\ldots\sigma_L}``.
This requires ``O(2^L)`` memory and time.

The MPS contraction is dramatically cheaper.  The overlap factors as a
sequence of matrix multiplications building up a *transfer matrix*
``E^{[i]} = E^{[i-1]} \cdot T^{[i]}``, where the transfer matrix at site
``i`` contracts the bra and ket tensors across the physical leg ``\sigma``:

```math
E^{[i]}_{\alpha\bar\alpha, \beta\bar\beta} =
  \sum_\sigma \bar\psi^{[i]}_{\alpha\sigma\beta} \,\phi^{[i]}_{\bar\alpha\sigma\bar\beta}
```

Starting from a ``1\times 1`` left boundary ``E^{[0]} = [1]`` and
contracting right to the boundary gives ``\langle\psi|\phi\rangle = E^{[L]}_{11}``.
Cost: ``O(L \chi^3 d)`` — polynomial in the bond dimension.

````julia
function mps_overlap(bra::Vector{<:Array{<:Number,3}},
                     ket::Vector{<:Array{<:Number,3}})
    L = length(bra)
    length(bra) == length(ket) || throw(ArgumentError("MPS lengths differ"))
    T   = promote_type(eltype(bra[1]), eltype(ket[1]))
    env = ones(T, 1, 1)   # (χL_bra × χL_ket)

    for i in 1:L
        A = conj.(bra[i])              # (χL_b, d, χR_b)
        B = ket[i]                     # (χL_k, d, χR_k)
        χL_b, d, χR_b = size(A)
        χL_k, _,  χR_k = size(B)
        # env (χL_b × χL_k) → contract with A and B over left bonds and physical index
        A_mat = reshape(A, χL_b, d * χR_b)       # (χL_b, d·χR_b)
        tmp   = env' * A_mat                      # (χL_k, d·χR_b)
        tmp   = reshape(tmp, χL_k, d, χR_b)
        B_mat = reshape(B, χL_k * d, χR_k)       # (χL_k·d, χR_k)
        tmp2  = reshape(permutedims(tmp, (2,1,3)), d * χL_k, χR_b)
        env   = reshape(tmp2' * B_mat, χR_b, χR_k)
    end

    return env[1, 1]
end
````

````
mps_overlap (generic function with 1 method)
````

Build two normalized MPS from the raw state tensors.

````julia
mps1 = FiniteMPS(Spin{1//2}(), L, 64)
mps2 = FiniteMPS(Spin{1//2}(), L, 64)
left_canonical_sweep!(mps1)
left_canonical_sweep!(mps2)

tensors1 = [t.data for t in mps1.tensors]
tensors2 = [t.data for t in mps2.tensors]

println("⟨ψ₁|ψ₁⟩ = ", mps_overlap(tensors1, tensors1))   # should be 1
println("⟨ψ₂|ψ₂⟩ = ", mps_overlap(tensors2, tensors2))   # should be 1
println("⟨ψ₁|ψ₂⟩ = ", mps_overlap(tensors1, tensors2))   # typically < 1
````

````
⟨ψ₁|ψ₁⟩ = 0.005035513352380012
⟨ψ₂|ψ₂⟩ = -0.010341191954313768
⟨ψ₁|ψ₂⟩ = 0.01638842612034289

````

## Qritical.jl equivalent

`overlap(bra, ket)` in Qritical.jl implements the same boundary contraction
directly on `FiniteMPS` objects.

````julia
println("\nQritical.jl overlap:")
println("  ⟨mps1|mps1⟩ = ", round(real(overlap(mps1, mps1)); sigdigits=8))
println("  ⟨mps1|mps2⟩ = ", round(real(overlap(mps1, mps2)); sigdigits=8))
````

````

Qritical.jl overlap:
  ⟨mps1|mps1⟩ = 1.0
  ⟨mps1|mps2⟩ = 0.0085468842

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

