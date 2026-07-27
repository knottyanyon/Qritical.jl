"""
    WireId

Identifies one wire in a generalized topological graph, a 1-manifold component of the graph in Joyal's sense (`G − G₀`). A wire is what carries a shared index between tensors: a contracted bond, an open physical leg, or a bare identity, depending on how many of its two ends are attached (see [`attachment`](@ref)).

!!! note "\"Wire\", not \"edge\""
    Joyal & Street's own word for this is **edge** ([joyal_street_1991](@cite)); "wire" is the graphical-calculus term for the same thing, following the string-diagram convention surveyed in [selinger_2010](@cite), and is the more natural word once the edge is carrying an actual tensor index rather than an abstract combinatorial relation.

Kept as its own struct rather than a bare `Int` because it indexes a different set from [`LegId`](@ref) and [`NodeId`](@ref): conflating them would let a leg id be passed where a wire id is expected, silently.

# Examples

```jldoctest
julia> WireId(1) == WireId(1)
true

julia> WireId(1) == WireId(2)
false
```
"""
struct WireId
    n::Int
end

"""
    LegId

Identifies one leg, one end of one wire, as seen by the tensor whose axis it occupies. Unlike [`WireId`](@ref) and [`NodeId`](@ref), a leg is not one of Joyal's primitives (nodes and edges, [joyal_street_1991](@cite)); it is Qritical's own bookkeeping device for the normalised relational layout described in the topograph design: a `Wire` records its two ends as `LegId`s, and each `Leg` records its own wire back by `WireId`, so neither object holds the other by value.

# Examples

```jldoctest
julia> LegId(1) == LegId(1)
true

julia> LegId(1) == LegId(2)
false
```
"""
struct LegId
    n::Int
end

"""
    NodeId

Identifies one node, a point of `G₀`, where wires attach. A node owns an ordered set of legs (one per axis of the tensor sitting there).

!!! definition "Node"
    Joyal's generalized topological graph is a space `G` with a discrete closed subset `G₀`, its **nodes**, such that `G − G₀` is a 1-manifold without boundary ([joyal_street_1991](@cite), p. 61).

# Examples

```jldoctest
julia> NodeId(1) == NodeId(1)
true

julia> NodeId(1) == NodeId(2)
false
```
"""
struct NodeId
    n::Int
end

# `==`/`hash` are defined explicitly (rather than relying on the default structural fallback) so that WireId/LegId/NodeId are safe Dict/Set keys

Base.:(==)(a::WireId, b::WireId) = a.n == b.n
Base.hash(a::WireId, h::UInt) = hash(a.n, hash(:WireId, h))

Base.:(==)(a::LegId, b::LegId) = a.n == b.n
Base.hash(a::LegId, h::UInt) = hash(a.n, hash(:LegId, h))

Base.:(==)(a::NodeId, b::NodeId) = a.n == b.n
Base.hash(a::NodeId, h::UInt) = hash(a.n, hash(:NodeId, h))
