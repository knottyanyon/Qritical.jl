# §9  Disorder realizations for MBL studies.
#
# A disorder realization is simply a random field array drawn from a distribution.
# The field array is passed directly to `XXZ(g; h=h_vec)`, which already supports
# per-site fields; all the MBL / imaginary-time GS machinery uses the same TEBD
# pipeline already implemented in §7–8.

using Random

"""
    Uniform(lo::Real, hi::Real)

A uniform distribution on the closed interval ``[lo, hi]``, used to draw on-site
random fields for disordered spin-chain models.

Pass a `Uniform` to [`disorder_realization`](@ref) to generate a random-field
vector ``h``, which you can then hand to [`XXZ`](@ref) via its `h` keyword.

The standard choice for MBL studies is `Uniform(-W, W)` where ``W`` is the
*disorder strength*: at ``W = 0`` the chain is translationally invariant; beyond
a critical ``W_c \\approx 3.5`` the 1D XXZ chain enters the many-body-localised
(MBL) phase.

# Arguments
- `lo::Real`: lower bound of the distribution interval.
- `hi::Real`: upper bound.  Must satisfy `hi ≥ lo` conceptually; no runtime
  check is performed, but drawing from a reversed interval returns nonsense.

# Examples
```jldoctest
julia> d = Uniform(-3.5, 3.5);

julia> d.lo, d.hi
(-3.5, 3.5)
```
"""
struct Uniform
    lo::Float64
    hi::Float64
    Uniform(lo::Real, hi::Real) = new(Float64(lo), Float64(hi))
end

"""
    disorder_realization(n::Int, dist::Uniform, rng::AbstractRNG) -> Vector{Float64}

Draw `n` independent on-site random fields from `dist` and return them as a
length-``n`` vector.  This is the standard way to build a single disorder
realization for an MBL or Anderson-localization study.

### Physical picture

Adding random on-site fields to the XXZ chain breaks its translational symmetry
and can localise all eigenstates.  The disordered Hamiltonian is

```math
H = \\frac{J}{2} \\sum_{i} \\bigl(S^+_i S^-_{i+1} + S^-_i S^+_{i+1}\\bigr)
  + J_z \\sum_{i} S^z_i S^z_{i+1}
  - \\sum_{i} h_i S^z_i
```

where each ``h_i`` is drawn independently from a uniform distribution
``\\mathcal{U}(-W, W)``.  At weak disorder ``W \\ll 1`` the ground state is
entangled and TEBD needs large bond dimension.  Above the MBL transition
(``W \\gtrsim W_c \\approx 3.5`` for the 1D XXZ chain) eigenstates are
near-product states with area-law entanglement, so imaginary-time TEBD
converges quickly at small bond dimension.

### Reproducibility

Passing the same `rng` seed every time gives you bit-identical field vectors
across runs — essential for disorder averaging, where you want to accumulate
statistics over many realizations of ``\\{h_i\\}`` while keeping each
realization fixed between method comparisons.

# Arguments
- `n::Int`: number of sites (length of the returned field vector).
- `dist::Uniform`: the distribution; typically `Uniform(-W, W)` for disorder
  strength ``W``.
- `rng::AbstractRNG`: a seeded RNG, e.g. `MersenneTwister(42)`.  The caller
  controls the seed so that realizations are reproducible.

# Returns
- `Vector{Float64}` of length `n`: the on-site fields
  ``h_1, \\ldots, h_n \\sim \\mathcal{U}(\\mathrm{lo}, \\mathrm{hi})``.

# Examples
```jldoctest
julia> using Random

julia> h = disorder_realization(4, Uniform(-3.5, 3.5), MersenneTwister(0));

julia> length(h)
4

julia> all(-3.5 .≤ h .≤ 3.5)
true
```

A complete ground-state workflow for one disorder realization:

```julia
using Random
L    = 8
W    = 3.5                                          # near the MBL transition
seed = 42
rng  = MersenneTwister(seed)
h    = disorder_realization(L, Uniform(-W, W), rng) # random fields
H    = XXZ(Chain(L); J=1.0, Jz=1.0, h=h)           # disordered XXZ Hamiltonian
res  = solve(H, GroundState(), ExactDiagonalization(:ground))
res.energy                                          # ground-state energy for this realization
```
"""
function disorder_realization(n::Int, dist::Uniform, rng::AbstractRNG)
    rand(rng, n) .* (dist.hi - dist.lo) .+ dist.lo
end

"""
    parameter_sweep(f, params) -> Vector

Apply `f` to each element of `params` and collect the results.

This is the standard driver for parameter scans: coupling sweeps (J, Jz, W),
disorder-averaged observables, or finite-size scaling studies.  The interface
is intentionally minimal — `f` is a user-supplied closure that builds the
Hamiltonian, solves it, and returns any observable; `parameter_sweep` simply
maps `f` over `params` and returns the collected results.

# Arguments
- `f`: a callable `f(p) -> result`.  Typically a `do`-block closure that
  constructs a Hamiltonian from the parameter value and calls [`solve`](@ref).
- `params`: any iterable of parameter values (e.g. `[0.5, 1.0, 2.0]` for a J
  sweep or `1:10` for disorder seeds).

# Returns
- `Vector` of the same length as `params`, with element type inferred from `f`.

# Examples
```julia
J_vals  = 0.5:0.5:2.0
g       = Chain(6)
energies = parameter_sweep(J_vals) do J
    H = Heisenberg(g; J=J)
    solve(H, GroundState(), ExactDiagonalization(:ground)).energy
end
```
"""
parameter_sweep(f, params) = map(f, params)
