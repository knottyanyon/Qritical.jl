# SECTION - TensorNetwork

"""
    TensorNetwork

The oriented, polarised implementation of the generalized topological graph interface: a container of nodes, wires, and the legs that join them, laid out as the normalised relational tables described in the topograph design. Nothing here is a `QTensor` yet, wiring an actual tensor network's numerical data through `TensorNetwork` is M-D's job. `TensorNetwork` only owns the graph shape: which nodes exist, which wires exist, and which leg of which node sits on which end of which wire.

# Fields

  - `wires :: Dict{WireId,Wire}`
  - `legs :: Dict{LegId,Leg}`
  - `node_legs :: Dict{NodeId,Vector{LegId}}`, the legs owned by each node, one entry per axis in axis order. This is what makes `legs(g, n)` an O(deg) lookup rather than a scan over every leg in the network.

Construct an empty network with `TensorNetwork()`, then populate it with [`add_node!`](@ref), [`add_wire!`](@ref), and [`add_leg!`](@ref).

# Examples

```jldoctest
julia> g = TensorNetwork();

julia> n = add_node!(g)
NodeId(1)

julia> w = add_wire!(g, 4; label=:bond)
WireId(1)

julia> nodes(g)
1-element Vector{NodeId}:
 NodeId(1)

julia> degree(g, n)
0
```
"""
mutable struct TensorNetwork <: AbstractGenTopoGraph
    wires::Dict{WireId,Wire}
    legs::Dict{LegId,Leg}
    node_legs::Dict{NodeId,Vector{LegId}}
    _next_wire::Int
    _next_leg::Int
    _next_node::Int
end

function TensorNetwork()
    TensorNetwork(
        Dict{WireId,Wire}(), Dict{LegId,Leg}(), Dict{NodeId,Vector{LegId}}(), 1, 1, 1
    )
end

# TensorNetwork carries a direction on every wire (see orientation.jl) and an axis order on every node's legs
graph_trait(::Type{TensorNetwork}) = Polarised()

"""
    add_node!(g::TensorNetwork) -> NodeId

Reserve a fresh node in `g` and return its id. A node starts with no legs, `add_leg!` populates them by adding one axis at a time.
"""
function add_node!(g::TensorNetwork)
    id = NodeId(g._next_node)
    g._next_node += 1
    g.node_legs[id] = LegId[]
    return id
end

"""
    add_wire!(g::TensorNetwork, space; kwargs...) -> WireId

Create a fresh, unattached wire carrying `space` and return its id. Keyword arguments are forwarded to the [`Wire`](@ref) constructor (`label`, `dof`, `spectrum`), a freshly added wire always starts loose, `start`/`finish` are set later by [`pin!`](@ref) and cannot be passed here.
"""
function add_wire!(g::TensorNetwork, space; kwargs...)
    id = WireId(g._next_wire)
    g._next_wire += 1
    g.wires[id] = Wire(id, space; kwargs...)
    return id
end

"""
    add_leg!(g::TensorNetwork, A, owner::NodeId, wire::WireId) -> LegId

Build the leg for the next free axis of node `owner` from tensor `A`'s indices, using [`make_leg`](@ref), and register it in `g`'s leg table and in `owner`'s ordered leg list. The axis is derived from how many legs `owner` already has, callers add a node's legs in axis order, matching the order of `A`'s own indices.

When `wire`'s space is a bare `Int`, this also checks that it matches the new leg's dimension, catching a mismatched bond before it reaches an `@tensor` contraction rather than after. Once `wire`'s space is a graded TensorKit space the two are not comparable this way yet, that bridge is M10's job, so the check is skipped for anything that is not an `Int`.
"""
function add_leg!(g::TensorNetwork, A, owner::NodeId, wire::WireId)
    axis = length(g.node_legs[owner]) + 1
    ℓ = make_leg(A, axis, wire, owner)
    space = g.wires[wire].space
    if space isa Int && space != dim(ℓ.ix)
        throw(
            ArgumentError(
                "leg dimension $(dim(ℓ.ix)) does not match wire $(wire) space $(space)"
            ),
        )
    end
    return _register_leg!(g, owner, ℓ)
end

# Shared by add_leg! and compactify's _add_boundary_leg!: both build a Leg by different routes (from a tensor's own indices, or from a bare TIx with no backing tensor), but the bookkeeping of registering it into g's tables is identical either way.
function _register_leg!(g::TensorNetwork, owner::NodeId, ℓ::Leg)
    id = LegId(g._next_leg)
    g._next_leg += 1
    g.legs[id] = ℓ
    push!(g.node_legs[owner], id)
    return id
end

"""
    nodes(g::TensorNetwork) -> Vector{NodeId}

All node ids currently registered in `g`, in no particular order.
"""
nodes(g::TensorNetwork) = collect(keys(g.node_legs))

"""
    wires(g::TensorNetwork) -> Vector{WireId}

All wire ids currently registered in `g`, in no particular order.
"""
wires(g::TensorNetwork) = collect(keys(g.wires))

"""
    legs(g::TensorNetwork, n::NodeId) -> Vector{LegId}

The leg ids owned by node `n`, in axis order. An O(deg) table lookup, not a scan over every leg in `g`, this is the whole point of storing `node_legs` alongside `legs` rather than deriving it by filtering.
"""
legs(g::TensorNetwork, n::NodeId) = g.node_legs[n]

"""
    ends(g::TensorNetwork, w::WireId) -> Set{NodeId}

The node ids that wire `w` touches, unordered, per the settled interface rule that `ends` returns node ids at level 0 and never exposes a `Leg` directly. Reading from zero elements (a loose wire or a circle) to two (a pinned bond) depending on [`attachment`](@ref); a self-loop or trace collapses to a single-element set since both ends share the same owner.
"""
function ends(g::TensorNetwork, w::WireId)
    wire = g.wires[w]
    result = Set{NodeId}()
    wire.start !== nothing && push!(result, g.legs[wire.start].owner)
    wire.finish !== nothing && push!(result, g.legs[wire.finish].owner)
    return result
end

"""
    incident(g::TensorNetwork, n::NodeId) -> Vector{WireId}

The wires touching node `n`, found by asking each of `n`'s legs which wire it belongs to. This is the natural counterpart of [`ends`](@ref): `ends` goes wire to nodes, `incident` goes node to wires.
"""
incident(g::TensorNetwork, n::NodeId) = [g.legs[ℓ].wire for ℓ in g.node_legs[n]]

"""
    degree(g::TensorNetwork, n::NodeId) -> Int

The number of wires touching node `n`, i.e. `n`'s valence. Equal to `length(incident(g, n))`, which for a `TensorNetwork` node is also just how many legs the node has, since every leg sits on exactly one wire.
"""
degree(g::TensorNetwork, n::NodeId) = length(incident(g, n))
