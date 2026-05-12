using HalfIntegers
using TensorKit: TensorKit

# ── AbstractSite ──────────────────────────────────────────────────────────────

"""
    AbstractSite

Abstract base type for all lattice site kinds. Concrete subtypes encode the
local Hilbert space structure (dimension, symmetry sectors) at a single site
in a lattice model.

Subtypes: `SpinSite`, `SpinlessFermionicSite`, `SpinlessHardCoreBosonicSite`,
`SpinlessBosonicSite`.
"""
abstract type AbstractSite end

# ── SpinSite ──────────────────────────────────────────────────────────────────

"""
    SpinSite(spin_quantum_number::HalfInt, lattice_ordinal::Int; symmetry=:none)

A spin-S lattice site. The local Hilbert space has dimension `2S+1` with basis
states `|Sz = -S⟩, ..., |Sz = +S⟩`.

# Arguments
- `spin_quantum_number`: total spin S as a `HalfInt` (e.g. `half(1)` for spin-1/2)
- `lattice_ordinal`: ordered position of this site in the lattice (`i = 1, 2, ..., N`)
- `symmetry=:none`: symmetry to exploit for block-diagonal structure
  - `:U1`  — conserves Sz; sectors are `Sz = -S, ..., +S`, each multiplicity 1
  - `:SU2` — full rotational invariance; one sector labeled by total spin j = S
  - `:none` — no symmetry, plain `ℂ^(2S+1)`

# Example
```julia
SpinSite(half(1), 3; symmetry=:U1)   # spin-1/2 at site 3, U(1) symmetry
SpinSite(half(2), 1; symmetry=:SU2)  # spin-1 at site 1, SU(2) symmetry
```

```jldoctest
julia> local_hilbert_dim(SpinSite(half(1), 1))  # spin-1/2: 2*(1/2)+1 = 2
2

julia> local_hilbert_dim(SpinSite(half(2), 1))  # spin-1: 2*1+1 = 3
3
```
"""
struct SpinSite <: AbstractSite
    spin_quantum_number::HalfInt
    lattice_ordinal::Int
    space::TensorKit.ElementarySpace
end

function SpinSite(
    spin_quantum_number::HalfInt, lattice_ordinal::Int; symmetry::Symbol=:none
)
    ## validation
    spin_quantum_number >= 0 || throw(
        ArgumentError("spin_quantum_number must be non-negative, got $spin_quantum_number"),
    )

    ## symmetry
    space = if symmetry == :none
        # ℂⁿ — just an n-dimensional complex vector space with no sectors, no block structure
        TensorKit.ComplexSpace(Int(2 * spin_quantum_number + 1))
    elseif symmetry == :U1
        # sectors: Sz = +S, S-1, ..., -S each with multiplicity 1
        pairs = [m => 1 for m in spin_quantum_number:-1:(-spin_quantum_number)]
        TensorKit.U1Space(pairs...)
    elseif symmetry == :SU2
        # one sector labeled by total spin j = S, multiplicity 1
        TensorKit.SU2Space(spin_quantum_number => 1)
    else
        throw(ArgumentError("SpinSite does not support symmetry=:$symmetry. Use :U1 or :SU2"))
    end
    return SpinSite(spin_quantum_number, lattice_ordinal, space)
end

# ── SpinlessFermionicSite ─────────────────────────────────────────────────────

"""
    SpinlessFermionicSite(lattice_ordinal::Int; symmetry=:none)

A spinless fermionic site with two basis states: empty `|0⟩` and occupied `|1⟩`.
Local Hilbert space dimension is 2. Obeys fermionic (antisymmetric) exchange
statistics — distinct from `SpinlessHardCoreBosonicSite` despite the same
Hilbert space dimension.

# Arguments
- `lattice_ordinal`: ordered position of this site in the lattice
- `symmetry=:none`: symmetry to exploit for block-diagonal structure
  - `:U1` — conserves particle number; sectors `n=0` and `n=1`
  - `:Z2` — conserves fermionic parity (even/odd occupation); useful for
             superconductors where particle number fluctuates but parity is fixed
  - `:none` — no symmetry, plain `ℂ²`

# Example
```julia
SpinlessFermionicSite(2; symmetry=:U1)  # site 2, particle number conservation
SpinlessFermionicSite(1; symmetry=:Z2)  # site 1, parity conservation only
```

```jldoctest
julia> local_hilbert_dim(SpinlessFermionicSite(1))  # empty |0⟩ or occupied |1⟩
2
```
"""
struct SpinlessFermionicSite <: AbstractSite
    lattice_ordinal::Int
    space::TensorKit.ElementarySpace
end

function SpinlessFermionicSite(lattice_ordinal::Int; symmetry::Symbol=:none)
    space = if symmetry == :none
        TensorKit.ComplexSpace(2)
    elseif symmetry == :U1
        TensorKit.U1Space(0 => 1, 1 => 1)   # n=0 empty, n=1 occupied
    elseif symmetry == :Z2
        TensorKit.Z2Space(0 => 1, 1 => 1)   # even/odd parity
    else
        throw(
            ArgumentError(
                "SpinlessFermionicSite does not support symmetry=:$symmetry. Use :U1, :Z2, or :none",
            ),
        )
    end
    return SpinlessFermionicSite(lattice_ordinal, space)
end

# ── SpinlessHardCoreBosonicSite ───────────────────────────────────────────────

"""
    SpinlessHardCoreBosonicSite(lattice_ordinal::Int; symmetry=:none)

A spinless hard-core bosonic site. The hard-core constraint limits occupation
to at most 1 boson per site, giving two basis states: `|0⟩` and `|1⟩`.
Local Hilbert space dimension is 2.

Shares the same Hilbert space as `SpinlessFermionicSite` but obeys bosonic
(symmetric) exchange statistics. The creation operator satisfies `(b†)² = 0`
and is equivalent to the Pauli raising operator S⁺ — the basis of the
Holstein-Primakoff mapping between spin-1/2 chains and hard-core bosons.

# Arguments
- `lattice_ordinal`: ordered position of this site in the lattice
- `symmetry=:none`: `:U1` (conserves particle number) or `:none`

# Example
```julia
SpinlessHardCoreBosonicSite(3; symmetry=:U1)  # site 3, particle number conservation
```

```jldoctest
julia> local_hilbert_dim(SpinlessHardCoreBosonicSite(1))  # hard-core: at most 1 boson, dim = 2
2
```
"""
struct SpinlessHardCoreBosonicSite <: AbstractSite
    lattice_ordinal::Int
    space::TensorKit.ElementarySpace
end

function SpinlessHardCoreBosonicSite(lattice_ordinal::Int; symmetry::Symbol=:none)
    space = if symmetry == :none
        TensorKit.ComplexSpace(2)
    elseif symmetry == :U1
        TensorKit.U1Space(0 => 1, 1 => 1)   # n=0 empty, n=1 occupied
    else
        throw(
            ArgumentError(
                "SpinlessHardCoreBosonicSite does not support symmetry=:$symmetry. Use :U1, or :none",
            ),
        )
    end
    return SpinlessHardCoreBosonicSite(lattice_ordinal, space)
end

# ── SpinlessBosonicSite ───────────────────────────────────────────────────────

"""
    SpinlessBosonicSite(lattice_ordinal::Int; n_max_occ::Int, symmetry=:none)

A spinless soft-core bosonic site allowing up to `n_max_occ` bosons per site.
Basis states are `|0⟩, |1⟩, ..., |n_max_occ⟩`, giving local Hilbert space
dimension `n_max_occ + 1`.

The creation operator satisfies `b†|n⟩ = √(n+1)|n+1⟩` — the `√(n+1)` factor
distinguishes soft-core from hard-core bosons and allows multiple occupancy.

# Arguments
- `lattice_ordinal`: ordered position of this site in the lattice
- `n_max_occ`: maximum occupation number (must be ≥ 0)
- `symmetry=:none`: `:U1` (conserves particle number, sectors `n=0,...,n_max_occ`)
                    or `:none`

# Example
```julia
SpinlessBosonicSite(1; n_max_occ=4, symmetry=:U1)  # up to 4 bosons, U(1) symmetry
SpinlessBosonicSite(2; n_max_occ=10)                # up to 10 bosons, no symmetry
```

```jldoctest
julia> local_hilbert_dim(SpinlessBosonicSite(1; n_max_occ=3))  # |0⟩,|1⟩,|2⟩,|3⟩: n_max_occ+1 = 4
4

julia> local_hilbert_dim(SpinlessBosonicSite(1; n_max_occ=0))  # vacuum only: dim = 1
1
```
"""
struct SpinlessBosonicSite <: AbstractSite
    lattice_ordinal::Int
    n_max_occ::Int
    space::TensorKit.ElementarySpace
end

function SpinlessBosonicSite(lattice_ordinal::Int; n_max_occ::Int, symmetry::Symbol=:none)
    ## validation
    n_max_occ >= 0 || throw(ArgumentError("n_max_occ must be non-negative, got $n_max_occ"))
    # build space for :none, :U1
    space = if symmetry == :none
        TensorKit.ComplexSpace(n_max_occ + 1)
    elseif symmetry == :U1
        pairs = [m => 1 for m in 0:n_max_occ]
        TensorKit.U1Space(pairs...)
    else
        throw(
            ArgumentError(
                "SpinlessBosonicSite does not support symmetry=:$symmetry. Use :U1, or :none",
            ),
        )
    end

    return SpinlessBosonicSite(lattice_ordinal, n_max_occ, space)
end

local_hilbert_dim(s::SpinSite) = Int(2 * s.spin_quantum_number + 1)
local_hilbert_dim(::SpinlessFermionicSite) = 2
local_hilbert_dim(::SpinlessHardCoreBosonicSite) = 2
local_hilbert_dim(s::SpinlessBosonicSite) = s.n_max_occ + 1
