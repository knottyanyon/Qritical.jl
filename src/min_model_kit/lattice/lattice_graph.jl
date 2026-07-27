# SECTION - LatticeGraph, the ordinary-graph realisation of a layout

"""
    LinkId

Identifies one bond of a [`LatticeGraph`](@ref). Distinct from [`WireId`](@ref): a lattice bond has no ends to attach and no space of its own, it is nothing more than an index into the edges found in the underlying layout.

# Examples

```jldoctest
julia> LinkId(1) == LinkId(1)
true

julia> LinkId(1) == LinkId(2)
false
```
"""
struct LinkId
    n::Int
end

Base.:(==)(a::LinkId, b::LinkId) = a.n == b.n
Base.hash(a::LinkId, h::UInt) = hash(a.n, hash(:LinkId, h))

"""
    LatticeGraph{L<:AbstractLayout} <: UndirectedGraph

The ordinary graph a [`AbstractLayout`](@ref) realises as: sites become nodes, bonds become pinned links, nothing dangles. `LatticeGraph` owns no mutable tables of its own, unlike [`TensorNetwork`](@ref); its whole structure is derived on demand from `layout`, since a layout's sites and bonds are already a complete graph description.

!!! note "Stops at Ungraded"
    A lattice bond carries no orientation and a node has no leg order, so [`graph_trait`](@ref) is left at its default, [`Ungraded`](@ref), rather than overridden to [`Oriented`](@ref) or [`Polarised`](@ref).

# Examples

```jldoctest
julia> g = LatticeGraph(Chain(4));

julia> nodes(g)
4-element Vector{Int64}:
 1
 2
 3
 4

julia> degree(g, 1)
1
```
"""
struct LatticeGraph{L<:AbstractLayout} <: UndirectedGraph
    layout::L
end

"""
    nodes(g::LatticeGraph) -> Vector{Int}

The sites of `g`'s layout, in the same order [`sites`](@ref) returns them.
"""
nodes(g::LatticeGraph) = collect(sites(g.layout))

"""
    links(g::LatticeGraph) -> Vector{LinkId}

All bond ids of `g`'s layout, one per entry of [`bonds`](@ref), in the same order.
"""
links(g::LatticeGraph) = [LinkId(k) for k in 1:length(bonds(g.layout))]

"""
    ends(g::LatticeGraph, ℓ::LinkId) -> Set{Int}

The two nodes bond `ℓ` connects, unordered. A lattice bond has no `start`/`finish`, [§6](@ref) reserves that vocabulary for [`Wire`](@ref); the pair is symmetric here because nothing about a lattice bond distinguishes its two nodes.
"""
function ends(g::LatticeGraph, ℓ::LinkId)
    (i, j) = bonds(g.layout)[ℓ.n]
    return Set((i, j))
end

"""
    incident(g::LatticeGraph, n::Int) -> Vector{LinkId}

The bonds touching node `n`.
"""
function incident(g::LatticeGraph, n::Int)
    [LinkId(k) for (k, (i, j)) in enumerate(bonds(g.layout)) if i == n || j == n]
end

"""
    degree(g::LatticeGraph, n::Int) -> Int

The number of bonds touching node `n`, i.e. `n`'s degree in the graph.
"""
degree(g::LatticeGraph, n::Int) = length(incident(g, n))

"""
    attachment(g::LatticeGraph, ℓ::LinkId) -> Pinned

Always [`Pinned`](@ref): every bond of an [`OrdinaryGraphNetwork`](@ref) is pinned by construction. Still resolves `ℓ` against `g`'s own bonds via [`ends`](@ref), so a `ℓ` that is not actually one of `g`'s links throws the same `BoundsError` [`ends`](@ref)/[`incident`](@ref)/[`degree`](@ref) would, rather than being silently accepted.
"""
function attachment(g::LatticeGraph, ℓ::LinkId)
    ends(g, ℓ)
    return Pinned()
end

# A bulk-degree node is one whose degree matches the maximum over the whole graph, i.e. one
# that is not missing any bonds a translation-invariant interior node would have. Used only by
# boundary below, not exported: a lattice-specific notion, unlike degree itself.
_bulk_degree(g::LatticeGraph) = maximum(degree(g, n) for n in nodes(g))

"""
    boundary(g::LatticeGraph) -> Vector{Int}

The nodes of `g` whose degree falls short of the bulk degree ([joyal_street_1991](@cite) has no notion of a lattice boundary, this is this project's own reasoning, not a cited result). Open boundary conditions mean fewer bonds at the two ends, never a dangling half-bond: `boundary` is a predicate on nodes derived from [`degree`](@ref), not an edge classification like [`attachment`](@ref) is for a [`TensorNetwork`](@ref).

# Examples

```jldoctest
julia> boundary(LatticeGraph(Chain(4)))
2-element Vector{Int64}:
 1
 4

julia> boundary(LatticeGraph(Chain(4, true)))
Int64[]
```
"""
function boundary(g::LatticeGraph)
    bulk = _bulk_degree(g)
    return [n for n in nodes(g) if degree(g, n) < bulk]
end

"""
    compactify(g::LatticeGraph) -> LatticeGraph

The identity on `g`. [`OrdinaryGraphNetwork`](@ref) guarantees every bond is already [`Pinned`](@ref), there is no half-loose or loose end for [`compactify`](@ref) to repair, unlike the [`TensorNetwork`](@ref) case this function is named after.
"""
compactify(g::LatticeGraph) = g
