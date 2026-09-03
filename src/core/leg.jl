#=META
source:
  author: Bavithra
  reviewer:
docstrings:
  author: Claude Sonnet 5
  coauthor: 
  reviewer:
refs: coecke_kissinger_2016a
credits: N/A
=#

using TensorKit

# SECTION -  PenroseOrientation: the dual/normal type-level tag for a PenroseLabel

"""
    PenroseOrientation

Abstract root of the "is this wire dual or not" trait family, named after Penrose's
diagrammatic calculus for abstract tensor systems (Penrose, 1971) that Coecke & Kissinger's
Definition 3.42 abstract tensor notation traces back to. A named wire `A` and its dual `A*`
(Coecke & Kissinger §4.6.2) are the *same* [`PenroseLabel`](@ref) identity (same `family`, same
`index`) differing only in this type-level orientation tag - exactly mirroring how directed
wires in a string diagram carry one label but point in one of two directions.

Concrete subtypes: [`Normal`](@ref) (`A`), [`Dual`](@ref) (`A*`).
"""
abstract type PenroseOrientation end

"""
    Normal <: PenroseOrientation

The "un-starred" orientation of a named wire (`A`, not `A*`).
"""
struct Normal <: PenroseOrientation end

"""
    Dual <: PenroseOrientation

The "starred" (dual-type) orientation of a named wire (`A*`, Coecke & Kissinger Definition 4.93).
"""
struct Dual <: PenroseOrientation end

# SECTION -  PenroseLabel: a named, oriented wire identity

"""
    PenroseLabel{O<:PenroseOrientation}

A named wire identity in the sense of Penrose's abstract tensor notation (Coecke & Kissinger
Definition 3.42): a family name (`A`, `B`, ...), an optional enumerating index (`A_1`, `A_2`,
...), and a type-level [`PenroseOrientation`](@ref) marking whether this is the wire `A` or its
dual `A*` (Definition 4.93).

`O` is carried as a type parameter, not a field, because dispatch needs to branch on
orientation at compile time - e.g. sequential composition only connecting an output wire to a
dual-matching input wire (Coecke & Kissinger's directed-wire convention, §4.6.2) - exactly the
same "Holy traits" reasoning [`LegRole`](@ref)'s [`PhysicalLeg`](@ref)/[`VirtualLeg`](@ref)
split already uses. `family`/`index` stay plain fields: nothing ever needs to specialize code on
*which* index a wire has, only compare two labels for equality, so making `index` a type
parameter too would bloat the type domain for no dispatch benefit.

# Fields

$(Glossaries.Field{Core}()([:penrose_family, :penrose_index]))

# Examples

```jldoctest
julia> a = PenroseLabel(:A);

julia> a2 = PenroseLabel(:A, 2);

julia> a == PenroseLabel(:A)
true

julia> a == a2
false

julia> orientation_dual(orientation_dual(a)) == a
true
```
"""
struct PenroseLabel{O<:PenroseOrientation}
    family::Symbol
    index::Int
end
PenroseLabel(family::Symbol, index::Int=0) = PenroseLabel{Normal}(family, index)

"""
    orientation_dual(w::PenroseLabel) -> PenroseLabel

Flip `w`'s [`PenroseOrientation`](@ref) (`A` ↔ `A*`), keeping `family`/`index` unchanged.
`orientation_dual(orientation_dual(w)) == w` by construction - the involution law Coecke &
Kissinger's Definition 4.97 requires of a dagger functor's action on types (`(A*)* := A`).

Named `orientation_dual`, not `dual`, to avoid colliding with `TensorKit.dual` (which operates
on spaces, not on wire labels) once both are in scope together.
"""
orientation_dual(w::PenroseLabel{Normal}) = PenroseLabel{Dual}(w.family, w.index)
orientation_dual(w::PenroseLabel{Dual}) = PenroseLabel{Normal}(w.family, w.index)

function Base.:(==)(a::PenroseLabel, b::PenroseLabel)
    return typeof(a) == typeof(b) && a.family == b.family && a.index == b.index
end
Base.hash(w::PenroseLabel{O}, h::UInt) where {O} = hash(O, hash(w.index, hash(w.family, h)))

# SECTION -  Leg: a PenroseLabel bridged to a concrete AbstractIx

"""
    Leg{I<:AbstractIx}

A named wire ([`PenroseLabel`](@ref)) paired with the [`AbstractIx`](@ref) that bridges it to a
concrete `TensorKit.ElementarySpace` - i.e. the abstract Coecke & Kissinger system-type together
with the chosen orthonormal basis (Coecke & Kissinger Definition 5.5) that turns it into
something with array-positional matrix entries (Definition 5.16). `TIx` alone already plays that
bridging role (see its own docstring); `Leg` adds the abstract name on top, for contexts that
need named-wire identity rather than purely positional leg matching.

This is a standalone, additive type: [`QProcess`](@ref)'s `outputs`/`inputs` remain plain
`Tuple{Vararg{TIx}}` as before - nothing in the existing process layer constructs a `Leg`, so
code that never needs named-wire reasoning (e.g. an MPS canonicalization sweep, where only one
bond leg is ever matched at a time) pays no cost for this type existing.

# Fields

$(Glossaries.Field{Core}()([:penrose_label, :bridging_ix]))

# Examples

```jldoctest
julia> ℓ = Leg(PenroseLabel(:A), TIx(4));

julia> dim(ℓ)
4

julia> space(ℓ) == TensorKit.ComplexSpace(4)
true
```
"""
struct Leg{I<:AbstractIx}
    label::PenroseLabel
    ix::I
end

dim(leg::Leg) = dim(leg.ix)
space(leg::Leg) = space(leg.ix)
symmetry_structure(leg::Leg) = symmetry_structure(leg.ix)
entanglement_structure(leg::Leg) = entanglement_structure(leg.ix)

Base.:(==)(a::Leg, b::Leg) = a.label == b.label && a.ix == b.ix
Base.hash(leg::Leg, h::UInt) = hash(leg.ix, hash(leg.label, h))
