using HalfIntegers: HalfInteger

# ── AbstractDoF hierarchy ─────────────────────────────────────────────────────

"""
    AbstractDoF

Root of the degree-of-freedom hierarchy.  Every physical site type carries a
`D <: AbstractDoF` type parameter encoding the local physics at compile time.
"""
abstract type AbstractDoF end

"""
    Spin{S}

Spin-S degree of freedom.  The local Hilbert space has dimension ``2S + 1``.
`S` should be a half-integer or integer (e.g. `1//2`, `1`, `3//2`).
"""
struct Spin{S} <: AbstractDoF end

"""
    Fermionic

Spinless fermionic site.  Local Hilbert space: ``|0\\rangle`` (empty) and
``|1\\rangle`` (occupied), subject to a ``\\mathbb{Z}_2`` parity superselection
rule.  In `:native` mode `hilbert_space` returns `2`; in `:tensorkit` mode it
returns `Rep{FermionParity}(0 => 1, 1 => 1)`.
"""
struct Fermionic <: AbstractDoF end

"""
    HardCoreBoson

Hard-core bosonic site.  Same two-dimensional local space as `Fermionic` but
without the fermionic parity superselection rule, so superpositions of empty
and occupied are physically allowed.
"""
struct HardCoreBoson <: AbstractDoF end

# ── hilbert_space — native mode ───────────────────────────────────────────────

"""
    hilbert_space(dof::AbstractDoF) -> Int

Return the local Hilbert space dimension for `dof`.

In `:native` mode this returns a plain `Int`.  In future `:tensorkit` mode it
will return a `TensorKit.ElementarySpace` carrying full symmetry-sector
information.

# Examples
```jldoctest
julia> hilbert_space(Spin{1//2}())
2

julia> hilbert_space(Spin{1}())
3

julia> hilbert_space(Fermionic())
2

julia> hilbert_space(HardCoreBoson())
2
```
"""
hilbert_space(::Spin{S}) where {S} = Int(2S + 1)
hilbert_space(::Fermionic)          = 2
hilbert_space(::HardCoreBoson)      = 2

# ── AbstractSite hierarchy ────────────────────────────────────────────────────

"""
    AbstractSite{D <: AbstractDoF}

Root of the site hierarchy.  The type parameter `D` encodes the local physics
at compile time; all dispatch on DoF type is resolved statically.
"""
abstract type AbstractSite{D <: AbstractDoF} end

"""
    StateSite{D}

MPS/PEPS site: one open physical leg whose dimension is `hilbert_space(D())`.
"""
struct StateSite{D <: AbstractDoF} <: AbstractSite{D} end

"""
    OperatorSite{D}

MPO/PEPO site: two open physical legs (bra + ket), each of dimension
`hilbert_space(D())`.
"""
struct OperatorSite{D <: AbstractDoF} <: AbstractSite{D} end
