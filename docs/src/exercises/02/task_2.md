```@meta
EditURL = "task_2.jl"
```

# Task 2.2 — Right Canonical State

!!! question "Task 2.2 — Right Canonical State"
    Write a function that performs a right canonical decomposition of the state
    `psi.jls` of [Exercise 1](@ref "Task 1.3 — SVD a state").  The function
    should allow for a maximum matrix dimension ``D`` to truncate the state
    after each SVD.

````julia
using Serialization, LinearAlgebra, Qritical
````

````julia
DATA_ROOT = normpath(joinpath(@__FILE__, ".."))
FPATH_PSI = normpath(joinpath(DATA_ROOT, "psi.jls"))

ψ = deserialize(FPATH_PSI)
L = ndims(ψ)
d = size(ψ, 1)
````

````
2
````

## The right-canonical MPS decomposition

An MPS in **right-canonical form** is a sequence of site tensors
``B_1, B_2, \ldots, B_L`` satisfying the *right-isometry* condition at every site:

```math
\sum_{\sigma, \alpha_R} (B_i)_{\alpha_L \sigma \alpha_R} \,
                        (B_i)^*_{\alpha_L' \sigma \alpha_R} = \delta_{\alpha_L \alpha_L'}
\qquad\Longleftrightarrow\qquad
B_i B_i^\dagger = \mathbb{1}
```

The construction mirrors the left-canonical sweep but runs **right to left**.
At step ``i`` (counting from the right) we reshape the current factor into a
``d^{i-1} \times (d \, \chi_R)`` matrix, take a thin SVD, keep at most ``D``
singular values, store the ``V^\dagger`` factor as the right-canonical tensor
``B_i``, and absorb ``U S`` leftward:

```math
M_i \;=\; U_i \, S_i \, V_i^\dagger,\qquad
B_i = \operatorname{reshape}(V_i^\dagger[1{:}r, :],\, r_i, d, \chi_R),\qquad
M_{i-1} = \operatorname{reshape}(U_i[:, 1{:}r]\, S_i,\, d^{i-1})
```

``V^\dagger V = \mathbb{1}`` holds automatically, so ``B_i B_i^\dagger = \mathbb{1}``
follows from the reshape.

````julia
function right_canonical_mps(ψ::Array, D::Int)
    L    = ndims(ψ)
    dims = size(ψ)
    T    = eltype(ψ)
    RT   = real(T)

    tensors  = Vector{Array{T, 3}}(undef, L)
    bond_svs = Vector{Vector{RT}}(undef, L + 1)
    bond_svs[1]     = RT[1]
    bond_svs[L + 1] = RT[1]

    χR      = 1
    current = reshape(ψ, :, 1)   # (d^L, 1)

    for i in L:-1:2
        M = reshape(current, :, dims[i] * χR)
        F = svd(M; full=false)
        r = max(min(count(>(0), F.S), D), 1)
        tensors[i]  = reshape(F.Vt[1:r, :], r, dims[i], χR)
        bond_svs[i] = F.S[1:r]
        current     = F.U[:, 1:r] * Diagonal(F.S[1:r])
        χR          = r
    end
    tensors[1] = reshape(current, 1, dims[1], χR)

    return tensors, bond_svs
end
````

````
right_canonical_mps (generic function with 1 method)
````

## Verifying right-isometry

For each tensor ``B_i`` of shape ``(\chi_{i-1}, d, \chi_i)``, reshape to
``\chi_{i-1} \times (d \, \chi_i)`` and check ``M M^\dagger = \mathbb{1}``.

````julia
function check_right_isometry(tensors::Vector{<:Array{<:Number,3}}; atol=1e-12)
    all_ok = true
    for (i, B) in enumerate(tensors)
        χL, d, χR = size(B)
        M   = reshape(B, χL, d * χR)
        err = norm(M * M' - I(χL))
        label = err < atol ? "✓" : "✗"
        println("  Site $i ($label):  ‖BB† − I‖ = $(round(err; sigdigits=4))")
        err > atol && (all_ok = false)
    end
    all_ok || @warn "Some sites are not right-isometric!"
    return all_ok
end
````

````
check_right_isometry (generic function with 1 method)
````

````julia
D = 64
tensors, bond_svs = right_canonical_mps(ψ, D)

println("Right-canonical MPS (D = $D):")
check_right_isometry(tensors)
println("Bond dimensions: ", join([size(t, 1) for t in tensors], " · "), " → 1")
````

````
Right-canonical MPS (D = 64):
  Site 1 (✓):  ‖BB† − I‖ = 1.998e-15
  Site 2 (✓):  ‖BB† − I‖ = 5.068e-16
  Site 3 (✓):  ‖BB† − I‖ = 1.287e-15
  Site 4 (✓):  ‖BB† − I‖ = 2.046e-15
  Site 5 (✓):  ‖BB† − I‖ = 3.969e-15
  Site 6 (✓):  ‖BB† − I‖ = 5.947e-15
  Site 7 (✓):  ‖BB† − I‖ = 3.11e-15
  Site 8 (✓):  ‖BB† − I‖ = 1.716e-15
  Site 9 (✓):  ‖BB† − I‖ = 3.156e-16
  Site 10 (✓):  ‖BB† − I‖ = 0.0
Bond dimensions: 1 · 2 · 4 · 8 · 16 · 32 · 16 · 8 · 4 · 2 → 1

````

## Comparing left and right canonical forms

Both forms represent the same quantum state.  As a sanity check, the singular
value spectrum at each bond should be identical regardless of which sweep
direction was used to canonicalize — the singular values encode physical
entanglement, not the gauge.

We reuse the `left_canonical_mps` implementation from Task 2.1.

````julia
function left_canonical_mps(ψ::Array, D::Int)
    L = ndims(ψ); dims = size(ψ); T = eltype(ψ); RT = real(T)
    tensors  = Vector{Array{T, 3}}(undef, L)
    bond_svs = Vector{Vector{RT}}(undef, L + 1)
    bond_svs[1] = RT[1]; bond_svs[L + 1] = RT[1]
    χL = 1; current = reshape(ψ, 1, :)
    for i in 1:(L-1)
        M = reshape(current, χL * dims[i], :)
        F = svd(M; full=false)
        r = max(min(count(>(0), F.S), D), 1)
        tensors[i] = reshape(F.U[:, 1:r], χL, dims[i], r)
        bond_svs[i + 1] = F.S[1:r]
        current = Diagonal(F.S[1:r]) * F.Vt[1:r, :]
        χL = r
    end
    tensors[L] = reshape(current, χL, dims[L], 1)
    return tensors, bond_svs
end

left_tensors, left_svs = left_canonical_mps(ψ, D)

println("\nSingular value spectra at bond 5 (middle):")
println("  Left canonical:  ", round.(left_svs[6]; sigdigits=4))
println("  Right canonical: ", round.(bond_svs[6]; sigdigits=4))
````

````

Singular value spectra at bond 5 (middle):
  Left canonical:  [0.7035, 0.7035, 0.05067, 0.05067, 0.05067, 0.05067, 0.00365, 0.00365, 0.0006184, 0.0006184, 0.0006184, 0.0006184, 4.455e-5, 4.455e-5, 4.455e-5, 4.455e-5, 4.455e-5, 4.455e-5, 4.455e-5, 4.455e-5, 3.209e-6, 3.209e-6, 3.209e-6, 3.209e-6, 5.437e-7, 5.437e-7, 3.916e-8, 3.916e-8, 3.916e-8, 3.916e-8, 2.821e-9, 2.821e-9]
  Right canonical: [0.7035, 0.7035, 0.05067, 0.05067, 0.05067, 0.05067, 0.00365, 0.00365, 0.0006184, 0.0006184, 0.0006184, 0.0006184, 4.455e-5, 4.455e-5, 4.455e-5, 4.455e-5, 4.455e-5, 4.455e-5, 4.455e-5, 4.455e-5, 3.209e-6, 3.209e-6, 3.209e-6, 3.209e-6, 5.437e-7, 5.437e-7, 3.916e-8, 3.916e-8, 3.916e-8, 3.916e-8, 2.821e-9, 2.821e-9]

````

## Qritical.jl equivalent

`right_canonical_sweep!` performs the same right-to-left sweep on any
`FiniteMPS` and sets `mps.form = CanonicalForm(0, 1)`.

````julia
mps = FiniteMPS(Spin{1//2}(), L, D)
right_canonical_sweep!(mps)

println("\nQritical.jl result:")
println("  form   : ", mps.form)
println("  ⟨ψ|ψ⟩  : ", round(real(overlap(mps, mps)); sigdigits=8))
println("  bond dims: ", join([size(t.data, 1) for t in mps.tensors], " · "))
````

````

Qritical.jl result:
  form   : Qritical.CanonicalForm(0, 1)
  ⟨ψ|ψ⟩  : 1.0
  bond dims: 1 · 2 · 4 · 8 · 16 · 32 · 16 · 8 · 4 · 2

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

