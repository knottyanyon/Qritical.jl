```@meta
EditURL = "task_3.jl"
```

# Task 3.3 — Checking the Normalization

!!! question "Task 3.3 — Checking the Normalization"
    Write a function that receives an MPS and checks at each site for left
    (``A``) and right (``B``) normalization.  Think about a good measure
    telling you how far away you are from unity in these normalizations.

````julia
using Serialization, LinearAlgebra, Qritical
````

````julia
DATA_ROOT = normpath(joinpath(@__FILE__, ".."))

ψ = deserialize(normpath(joinpath(DATA_ROOT, "psi.jls")))
L = ndims(ψ)
d = size(ψ, 1)
````

````
2
````

## What does "distance from isometry" mean?

A tensor ``T`` of shape ``(\chi_L, d, \chi_R)`` is:

- **Left-canonical** if ``M^\dagger M = \mathbb{1}_{\chi_R}`` where
  ``M = \operatorname{reshape}(T, \chi_L d, \chi_R)``
- **Right-canonical** if ``M M^\dagger = \mathbb{1}_{\chi_L}`` where
  ``M = \operatorname{reshape}(T, \chi_L, d \chi_R)``

A natural scalar measure is the Frobenius norm of the deviation:
``\delta_L(T) = \|M^\dagger M - \mathbb{1}\|_F`` and
``\delta_R(T) = \|M M^\dagger - \mathbb{1}\|_F``.
Both equal zero for a perfectly isometric tensor and grow with the
deviation.

````julia
function check_normalization(tensors::Vector{<:Array{<:Number,3}})
    N = length(tensors)
    left_errors  = Vector{Float64}(undef, N)
    right_errors = Vector{Float64}(undef, N)
    println("  site │  δ_L (left iso)   │  δ_R (right iso)")
    println("  ─────┼───────────────────┼──────────────────")
    for (i, T) in enumerate(tensors)
        χL, d, χR = size(T)
        ML = reshape(T, χL * d, χR)
        MR = reshape(T, χL, d * χR)
        left_errors[i]  = norm(ML' * ML - I(χR))
        right_errors[i] = norm(MR * MR' - I(χL))
        println("  $(lpad(i,4)) │  $(rpad(round(left_errors[i]; sigdigits=3), 17)) │  $(round(right_errors[i]; sigdigits=3))")
    end
    return (left_errors=left_errors, right_errors=right_errors)
end
````

````
check_normalization (generic function with 1 method)
````

## Test on known canonical forms

A freshly left-canonicalized MPS should have left_errors ≈ 0 everywhere
and right_errors growing away from zero.  Verify this:

````julia
mps_L = FiniteMPS(Spin{1//2}(), L, 32)
left_canonical_sweep!(mps_L)
tensors_L = [t.data for t in mps_L.tensors]

println("Left-canonical MPS:")
errs_L = check_normalization(tensors_L)
````

````
(left_errors = [3.1401849173675503e-16, 1.4579930984379224e-15, 1.9291371526262905e-15, 4.763015347724085e-15, 5.9543162503598736e-15, 4.456679024754443e-15, 2.640050282368916e-15, 1.8177799065704617e-15, 2.323428808430142e-16, 2.220446049250313e-16], right_errors = [1.0000000000000004, 1.4142135623730963, 1.9999999999999996, 2.8284271247461876, 3.999999999999999, 4.914528278122275, 2.8887743359129425, 1.8756544269470754, 1.2352839813995256, 0.8758389336071151])
````

A right-canonical MPS flips the pattern — right_errors ≈ 0, left_errors
growing.

````julia
mps_R = deepcopy(mps_L)
right_canonical_sweep!(mps_R)
tensors_R = [t.data for t in mps_R.tensors]

println("\nRight-canonical MPS:")
errs_R = check_normalization(tensors_R)
````

````
(left_errors = [0.8755659760427649, 1.3423005771693388, 1.9028973505273454, 2.842155802835751, 4.537396162120754, 3.7476511121441365, 2.8284271247461894, 2.0000000000000027, 1.414213562373094, 1.0000000000000004], right_errors = [0.0, 4.494593724252825e-16, 5.300628044220478e-16, 1.794130856203107e-15, 3.857360781821686e-15, 6.002096677994779e-15, 4.55764172519846e-15, 2.937966073527204e-15, 1.1479614784404827e-15, 3.1401849173675503e-16])
````

## Mixed canonical form

After `move_center!(mps, l)`, sites ``1 \ldots l-1`` are left-canonical,
site ``l`` is unconstrained, and sites ``l+1 \ldots L`` are right-canonical.
Your function should reflect this pattern clearly.

````julia
l = L ÷ 2
mps_M = deepcopy(mps_L)
move_center!(mps_M, l)
tensors_M = [t.data for t in mps_M.tensors]

println("\nMixed canonical (center = site $l):")
check_normalization(tensors_M)
````

````
(left_errors = [3.1401849173675503e-16, 1.4579930984379224e-15, 1.9291371526262905e-15, 4.763015347724085e-15, 5.318076542020608, 3.7476511121441365, 2.8284271247461894, 2.0000000000000027, 1.414213562373094, 1.0000000000000004], right_errors = [1.0000000000000004, 1.4142135623730963, 1.9999999999999996, 2.8284271247461876, 3.7861749769433253, 6.002096677994779e-15, 4.55764172519846e-15, 2.937966073527204e-15, 1.1479614784404827e-15, 3.1401849173675503e-16])
````

## Qritical.jl: reading the canonical form directly

`FiniteMPS` tracks its own form in `mps.form`.  After any sweep or
`move_center!` call you can query `mps.form` without running any
extra SVDs:

````julia
println("\nCanonical form stored on mps: ", mps_M.form)
println("  → sites 1…$(l-1) left-canonical, sites $(l+1)…$L right-canonical")
````

````

Canonical form stored on mps: Qritical.CanonicalForm(4, 6)
  → sites 1…4 left-canonical, sites 6…10 right-canonical

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

