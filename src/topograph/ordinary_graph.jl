# SECTION - the ordinary-graph branch of AbstractGenTopoGraph

"""
    OrdinaryGraphNetwork <: AbstractGenTopoGraph

A graph in which every edge is already attached at both of its ends, with no unattached ends and no closed loops left over as circles.

!!! definition "Graph, ordinary graph"
    "A **graph** is a generalised graph in which all the edges are pinned. It is called an **ordinary graph** when it has no circles" ([joyal_street_1991](@cite), p. 61).

!!! note "A type-level guarantee, not a convention"
    Because every edge of an `OrdinaryGraphNetwork` is pinned by construction, [`attachment`](@ref) never actually has to inspect one of its edges to answer the question, the answer is always [`Pinned`](@ref).

!!! note "Not a TensorNetwork"
    A [`TensorNetwork`](@ref) is not an `OrdinaryGraphNetwork`: it can have half-loose legs left open, and it can carry circles.
"""
abstract type OrdinaryGraphNetwork <: AbstractGenTopoGraph end

"""
    UndirectedGraph <: OrdinaryGraphNetwork

An [`OrdinaryGraphNetwork`](@ref) whose edges carry no direction: no choice of orientation has been made for any of them.

!!! definition "Oriented graph, for contrast"
    Joyal & Street define the opposite case: "An **oriented graph** is a graph together with a choice of orientation for each of its edges and circles" ([joyal_street_1991](@cite), p. 62). An `UndirectedGraph` is the case where that choice has simply not been made, Joyal has no separate name for it.

!!! note
    A lattice bond joins two sites with no start or finish, unlike a [`Wire`](@ref)'s `start`/`finish` pair.
"""
abstract type UndirectedGraph <: OrdinaryGraphNetwork end
