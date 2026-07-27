# SECTION - AbstractGenTopoGraph and its traits

"""
    AbstractGenTopoGraph

Abstract root of the generalized topological graph hierarchy ([joyal_street_1991](@cite)). Every concrete graph type in the topograph layer is a subtype, [`TensorNetwork`](@ref) is the first (an oriented, polarised implementor), an ordinary graph which also includes lattice graphs is another example.

A concrete subtype can add structure in stages, each one building on the one before it:

  1. A bare graph: nodes and wires exist, with no further structure attached to either.
  2. An **oriented** graph adds a direction to every wire.
  3. A **polarised** graph adds, on top of that, a linear order to the legs at every node.

A type can stop after any stage, but reaching the third always means it has already reached the second: nothing can be polarised without also being oriented. [`GraphTrait`](@ref) and [`graph_trait`](@ref) record which stage a given subtype has reached.
"""
abstract type AbstractGenTopoGraph end

"""
    GraphTrait

Abstract supertype for how far a concrete [`AbstractGenTopoGraph`](@ref) subtype has progressed through the stages described on [`AbstractGenTopoGraph`](@ref): [`Ungraded`](@ref) (the bare graph, no further structure), [`Oriented`](@ref) (wires have a direction), [`Polarised`](@ref) (node legs also have an order). [`graph_trait`](@ref) maps a graph type to one of these three singletons; [`is_oriented`](@ref)/[`is_polarised`](@ref) are the queries built on top of it.

!!! design-note "Design note"
    This is the "Holy traits" pattern: rather than asking at runtime whether a graph "is oriented", `graph_trait` returns a singleton value whose *type* answers the question, so dispatch on it is resolved at compile time. A new graph type opts into a stage by adding one method, `graph_trait(::Type{MyGraph}) = Oriented()`, rather than by editing a central `if`/`elseif` chain that every graph type would otherwise have to share.
"""
abstract type GraphTrait end

"""
    Ungraded <: GraphTrait

The bare graph: nodes and wires exist, but no direction and no leg order have been added. The default for any [`AbstractGenTopoGraph`](@ref) subtype that does not override [`graph_trait`](@ref).
"""
struct Ungraded <: GraphTrait end

"""
    Oriented <: GraphTrait

!!! definition "Oriented graph"
    "An **oriented graph** is a graph together with a choice of orientation for each of its edges and circles" (p. 62, [joyal_street_1991](@cite)).

Every wire carries a direction, on top of the bare graph's nodes and wires.
"""
struct Oriented <: GraphTrait end

"""
    Polarised <: GraphTrait

!!! definition "Polarised graph"
    "A **polarised graph** is an oriented graph together with a choice of linear order on each `in(x)` and `out(x)`" (p. 62, [joyal_street_1991](@cite)).

Every node's legs also carry a linear order (the array axis), on top of an oriented graph's directions. Implies [`Oriented`](@ref): [`is_oriented`](@ref) is `true` for a `Polarised` type too, per the stages described on [`AbstractGenTopoGraph`](@ref).
"""
struct Polarised <: GraphTrait end

"""
    graph_trait(::Type{<:AbstractGenTopoGraph}) -> GraphTrait

Which of the stages described on [`AbstractGenTopoGraph`](@ref) a concrete graph type has reached. Defaults to [`Ungraded`](@ref); a concrete subtype overrides this with its own method to claim a later stage, [`TensorNetwork`](@ref) claims [`Polarised`](@ref).
"""
graph_trait(::Type{<:AbstractGenTopoGraph}) = Ungraded()

"""
    is_oriented(g) -> Bool

`true` if `g`'s type has reached at least the oriented stage ([`Oriented`](@ref) or [`Polarised`](@ref)).
"""
is_oriented(g) = _is_oriented(graph_trait(typeof(g)))
_is_oriented(::Ungraded) = false
_is_oriented(::Oriented) = true
_is_oriented(::Polarised) = true

"""
    is_polarised(g) -> Bool

`true` if `g`'s type has reached the polarised stage ([`Polarised`](@ref)).
"""
is_polarised(g) = _is_polarised(graph_trait(typeof(g)))
_is_polarised(::Ungraded) = false
_is_polarised(::Oriented) = false
_is_polarised(::Polarised) = true
