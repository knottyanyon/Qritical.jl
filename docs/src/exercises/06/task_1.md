```@meta
EditURL = "task_1.jl"
```

# Task 6.1 — MPO I: site-dependent XXZ chain

!!! question "Task 6.1 — MPO I"
    Write a function that receives arrays ``J_i``, ``J_i^z``, and ``h_i`` and
    returns the MPO of

    ```math
    H = \sum_{i=1}^{L-1} \frac{J_i}{2}\left(S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+\right)
        + J_i^z S_i^z S_{i+1}^z
        - \sum_{i=1}^{L} h_i S_i^z
    ```

````julia
using LinearAlgebra, Qritical
````

## MPO as a finite-state machine (FSM)

An MPO is represented as a product of *W matrices*, one per site.  Each W
tensor has four legs: two virtual bonds (left/right, dimension ``\chi``) and
two physical legs (ket ``\sigma`` and bra ``\sigma'``):

```math
W_i[\alpha, \sigma, \sigma', \beta]
```

The contraction ``W_1 \cdot W_2 \cdots W_L`` reconstructs the full operator.
We think of ``\alpha`` and ``\beta`` as the **state** of a finite automaton
sweeping left to right:

| State | Meaning |
|-------|---------|
| 1 = IdL | "nothing applied yet" (left identity) |
| 2 = S⁺ | ``S^+`` applied; waiting for ``S^-`` at the next site |
| 3 = S⁻ | ``S^-`` applied; waiting for ``S^+`` |
| 4 = Sᶻ | ``S^z`` applied; waiting for ``S^z`` (Ising) |
| 5 = IdR | "all terms closed" (right identity) |

At interior site ``i`` the non-zero blocks are:

```math
W_i[\text{IdL}, \sigma, \sigma', \beta] = \begin{cases}
  I          & \beta = \text{IdL} \\
  S^+        & \beta = 2          \\
  S^-        & \beta = 3          \\
  S^z        & \beta = 4          \\
  -h_i S^z   & \beta = \text{IdR}
\end{cases}
```

```math
W_i[2, \sigma, \sigma', \text{IdR}] = \tfrac{J_{i-1}}{2} S^-, \quad
W_i[3, \sigma, \sigma', \text{IdR}] = \tfrac{J_{i-1}}{2} S^+, \quad
W_i[4, \sigma, \sigma', \text{IdR}] = J^z_{i-1} S^z, \quad
W_i[\text{IdR}, \sigma, \sigma', \text{IdR}] = I
```

The **coupling indices** follow the convention that
- ``J_i, J_i^z`` couple sites ``i`` and ``i+1`` (the bond to the RIGHT of site ``i``),
- completion at site ``i`` uses ``J_{i-1}`` (the bond from the LEFT).

````julia
function xxz_mpo(L::Int, Js::AbstractVector, Jzs::AbstractVector, hs::AbstractVector;
                 T::Type{<:Number}=Float64)
    # Returns a Vector{Array{T,4}} of W tensors, each of shape (χL, d_ket, d_bra, χR).
    # Convention: W[α, σ_ket, σ_bra, β] = ⟨σ_bra | op | σ_ket⟩ at FSM edge (α→β).
    d  = 2
    χ  = 5
    Id = Matrix{T}(I, d, d)
    Sz = T[1//2 0; 0 -1//2]
    Sp = T[0 1; 0 0]   ## S⁺
    Sm = T[0 0; 1 0]   ## S⁻

    tensors = Vector{Array{T,4}}(undef, L)
    for i in 1:L
        W = zeros(T, χ, d, d, χ)
        J_L  = i > 1 ? T(Js[i-1])  : zero(T)   ## coupling completed here (from left)
        Jz_L = i > 1 ? T(Jzs[i-1]) : zero(T)
        h_i  = T(hs[i])

        for σ in 1:d, σ′ in 1:d
            W[1, σ, σ′, 1]  = Id[σ′, σ]
            W[1, σ, σ′, 2]  = Sp[σ′, σ]
            W[1, σ, σ′, 3]  = Sm[σ′, σ]
            W[1, σ, σ′, 4]  = Sz[σ′, σ]
            W[1, σ, σ′, 5]  = -h_i * Sz[σ′, σ]
            W[2, σ, σ′, 5]  = (J_L/2) * Sm[σ′, σ]
            W[3, σ, σ′, 5]  = (J_L/2) * Sp[σ′, σ]
            W[4, σ, σ′, 5]  = Jz_L * Sz[σ′, σ]
            W[5, σ, σ′, 5]  = Id[σ′, σ]
        end

        tensors[i] = if L == 1
            reshape(W[1:1, :, :, 5:5], 1, d, d, 1)
        elseif i == 1
            reshape(W[1:1, :, :, :], 1, d, d, χ)
        elseif i == L
            reshape(W[:, :, :, 5:5], χ, d, d, 1)
        else
            copy(W)
        end
    end
    return tensors
end
````

````
xxz_mpo (generic function with 1 method)
````

## Verify: uniform coupling matches `heisenberg_mpo`

With ``J_i = J_i^z = J`` and ``h_i = 0``, the XXZ chain reduces to the
isotropic Heisenberg model.  Check that `xxz_mpo` gives the same 2-site
matrix as `heisenberg_mpo`.

````julia
L = 6; J = 1.0
Js  = fill(J, L - 1)
Jzs = fill(J, L - 1)
hs  = zeros(L)

W_student = xxz_mpo(L, Js, Jzs, hs)
println("Bond dimensions: ", [size(W, 1) for W in W_student], " → ", size(W_student[end], 4))
println("Interior W shape: ", size(W_student[3]))
````

````
Bond dimensions: [1, 5, 5, 5, 5, 5] → 1
Interior W shape: (5, 2, 2, 5)

````

Compare interior W tensor against Qritical.jl's heisenberg_mpo:

````julia
mpo_qritical = heisenberg_mpo(L; J=J)
W_ref = mpo_qritical.tensors[3].data   ## interior site

err = norm(W_student[3] - W_ref)
println("\n‖W_student - W_heisenberg‖ = ", round(err; sigdigits=4), "  (should be ≈ 0)")
````

````

‖W_student - W_heisenberg‖ = 0.0  (should be ≈ 0)

````

## Physical test: two-site spectrum

For ``L = 2``, ``J = J^z = 1``, ``h = 0``, the Heisenberg dimer has
eigenvalues ``\{-3J/4, J/4, J/4, J/4\}`` (singlet and triplet).
Contract the MPO to a ``4 \times 4`` matrix and verify the spectrum.

````julia
L2 = 2
W2 = xxz_mpo(L2, [1.0], [1.0], [0.0, 0.0])
# Contract W[1] and W[2] into the 4×4 two-site Hamiltonian
# H4[row, col] = Σ_α W[1][1, σ_ket, σ_bra, α] * W[2][α, σ_ket', σ_bra', 1]
# row = (σ_ket-1)*2 + σ_ket', col = (σ_bra-1)*2 + σ_bra'
H4 = sum(kron(W2[1][1, :, :, α], W2[2][α, :, :, 1]) for α in 1:5)
evs = sort(real(eigvals(H4)))
println("Eigenvalues: ", round.(evs; sigdigits=5))
println("Expected:    ", round.([-3/4, 1/4, 1/4, 1/4]; sigdigits=5))
````

````
Eigenvalues: [-0.75, 0.25, 0.25, 0.25]
Expected:    [-0.75, 0.25, 0.25, 0.25]

````

## Qritical.jl equivalent

`heisenberg_mpo(L; J)` builds the uniform-coupling version.  For
site-dependent couplings, `xxz_mpo` defined above (or the unimplemented
`Qritical.xxz_mpo` from ROADMAP v0.4) is needed.

````julia
mpo_heis = heisenberg_mpo(L; J=J)
println("\nQritical.jl FiniteMPO: L = ", mpo_heis.L,
        "  IdL = ", mpo_heis.IdL, "  IdR = ", mpo_heis.IdR)
````

````

Qritical.jl FiniteMPO: L = 6  IdL = 1  IdR = 5

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

