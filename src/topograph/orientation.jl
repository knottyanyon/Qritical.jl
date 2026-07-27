# SECTION - leg orientation

"""
    LegOrientation

Abstract supertype for the direction a leg's arrow points: [`Incoming`](@ref) or [`Outgoing`](@ref).

!!! note "Orientation vs. attachment"
    [`Attachment`](@ref) and `LegOrientation` classify two different things. Attachment is a property of a whole *wire*: how many of its two ends are attached. Orientation is a property of one *leg*, one particular end of a wire: which way the arrow points through it.

!!! design-note "Design note"
    The graph-level notion of orientation, a direction chosen for each wire, is Joyal's own ([`Oriented`](@ref), [joyal_street_1991](@cite)). `Incoming`/`Outgoing` are this project's own names for the two roles a wire's direction assigns to its ends, chosen deliberately over the category-theoretic `source`/`target`: "source" in category theory means *domain*, but a tensor sitting at a wire's source has that leg in its *codomain*, a direct collision this project's own index convention could not afford.
"""
abstract type LegOrientation end

"""
    Incoming <: LegOrientation

The leg sits at a wire's finish, `γ(1)`.

```text
────────▶●  tensor
  (the arrow points INTO the tensor: an Incoming leg)
```

!!! tip
    An incoming leg belongs to its tensor's domain, `TIx{Upper}`.

# Examples

```jldoctest
julia> orientation(TIx{Upper}(:σ, 2))
Incoming()
```
"""
struct Incoming <: LegOrientation end

"""
    Outgoing <: LegOrientation

The leg sits at a wire's start, `γ(0)`.

```text
tensor  ●────────▶
  (the arrow points OUT of the tensor: an Outgoing leg)
```

!!! tip
    An outgoing leg belongs to its tensor's codomain, `TIx{Lower}`.

# Examples

```jldoctest
julia> orientation(TIx{Lower}(:σ, 2))
Outgoing()
```
"""
struct Outgoing <: LegOrientation end

"""
    orientation(ix::TIx) -> LegOrientation

Read off the orientation an elementary index carries. An `Upper` index is incoming (domain), a `Lower` index is outgoing (codomain), following the same variance convention `TIx` already uses everywhere else in Qritical.
"""
orientation(::TIx{Upper}) = Incoming()
orientation(::TIx{Lower}) = Outgoing()

"""
    leg_orientation(ℓ::Leg) -> LegOrientation

Read off the orientation of a `Leg` fetched from a graph's leg table, e.g. `g.legs[id]`.

A plain lookup like that returns a value typed as the unparameterized `Leg`, since a `Dict{LegId,Leg}` cannot know the concrete variance `V` of the entry it stores until the value is actually read. Calling `orientation(ℓ.ix)` straight from such a lookup would force Julia to resolve the right method at runtime, on every single call. `leg_orientation` pays that cost exactly once: it hands `ℓ.ix` to `_leg_orientation`, which sees a concrete `TIx{V}` and specializes on it, so everything downstream of that point is stable again.
"""
leg_orientation(ℓ::Leg) = _leg_orientation(ℓ.ix)
_leg_orientation(ix::TIx{V}) where {V} = orientation(ix)
