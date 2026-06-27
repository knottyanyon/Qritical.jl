# §9  Disorder realizations for MBL studies.
#
# A disorder realization is simply a random field array drawn from a distribution.
# The field array is passed directly to `XXZ(g; h=h_vec)`, which already supports
# per-site fields; all the MBL / imaginary-time GS machinery uses the same TEBD
# pipeline already implemented in §7–8.

using Random

"""
    Uniform(lo::Real, hi::Real)

A uniform distribution on the interval ``[lo, hi]``.  Used as the `dist` argument
to [`disorder_realization`](@ref).
"""
struct Uniform
    lo::Float64
    hi::Float64
    Uniform(lo::Real, hi::Real) = new(Float64(lo), Float64(hi))
end

"""
    disorder_realization(n, dist, rng) -> Vector{Float64}

Draw `n` independent samples from `dist` using random-number generator `rng`.

The same `rng` seed produces an identical disorder realization every time, making
studies exactly reproducible.  For a uniform distribution `Uniform(lo, hi)`:
```math
h_i \\sim \\mathcal{U}(lo,\\, hi), \\quad i = 1, \\ldots, n.
```

Pass the returned vector as the `h` keyword to [`XXZ`](@ref):
```julia
h = disorder_realization(L, Uniform(-W, W), MersenneTwister(seed))
H = XXZ(Chain(L); J=1.0, Jz=1.0, h=h)
```
"""
function disorder_realization(n::Int, dist::Uniform, rng::AbstractRNG)
    rand(rng, n) .* (dist.hi - dist.lo) .+ dist.lo
end
