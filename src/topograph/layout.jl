# SECTION - AbstractLayout and the Chain layout

"""
    AbstractLayout

Abstract supertype for lattice layouts: a specification of an arrangement, parameters plus the rule for generating sites and bonds, not a graph by itself. A layout answers exactly two questions: which sites exist, and which pairs of sites are joined by a bond. That is all a Hamiltonian builder needs, it does not care whether the underlying arrangement is a chain, a square lattice, or a torus. Concrete subtypes only need to implement [`sites`](@ref) and [`bonds`](@ref).

The same layout can be turned into either a plain graph of sites and bonds (a [`LatticeGraph`](@ref)) or into an actual tensor network (a [`TensorNetwork`](@ref)) carrying whatever physical legs a chosen representation needs. Neither of those two things is stored on the layout itself, the layout only describes the arrangement.

!!! tip "One layout, many uses"
    The same `Chain` layout can drive very different constructions built on top of it: a plain 1D lattice model, an MPS, or an MPO, all sharing the same sites and bonds. The same idea generalises beyond `Chain`: a `Square` or hexagonal layout would describe a 2D lattice model, and a tree-shaped layout could describe a MERA tensor network, all through the same `AbstractLayout` interface.

Current concrete layout: [`Chain`](@ref) (1D open/periodic chain).

See also: [`Chain`](@ref), [`sites`](@ref), [`bonds`](@ref), [`LatticeGraph`](@ref)
"""
abstract type AbstractLayout end

"""
    Chain <: AbstractLayout

A one-dimensional chain of `L` sites with either open or periodic boundary conditions.

This is the workhorse layout for 1D condensed-matter models: Heisenberg, Ising, XXZ, Hubbard chains, and so on. Physically it represents a linear arrangement of sites ``1, 2, \\ldots, L``, with nearest-neighbour bonds ``(1,2), (2,3), \\ldots``. Under periodic boundary conditions the extra bond ``(L, 1)`` wraps the chain into a ring.

# Fields

  - `L::Int` — number of sites; must be positive.
  - `periodic::Bool` — `true` for periodic boundary conditions (ring), `false` for open boundary conditions (chain with two ends). Defaults to `false`.

# Constructors

```julia
Chain(L)              # open chain (OBC), L sites
Chain(L, true)        # periodic chain (PBC), L sites
```

# Examples

```jldoctest
julia> g = Chain(4);

julia> g.L
4

julia> g.periodic
false

julia> Chain(4, true).periodic
true
```
"""
struct Chain <: AbstractLayout
    L::Int
    periodic::Bool
end
Chain(L::Int) = Chain(L, false)

"""
    sites(g::Chain) -> UnitRange{Int}

Return the range of site indices ``1, 2, \\ldots, L`` for the chain `g`.

Every Hamiltonian or observable construction that needs to loop over lattice sites starts here. The return type is a `UnitRange`, so it is allocation-free and plays nicely with Julia's broadcasting and comprehension syntax.

# Examples

```jldoctest
julia> collect(sites(Chain(4)))
4-element Vector{Int64}:
 1
 2
 3
 4
```
"""
sites(g::Chain) = 1:g.L

"""
    bonds(g::Chain) -> Vector{Tuple{Int,Int}}

Return the list of nearest-neighbour bond pairs ``(i, j)`` for the chain `g`.

Each pair ``(i, j)`` with ``i < j`` represents a bond between sites ``i`` and ``j``. The ordering matters for MPO construction because the FSM channels are opened at site ``i`` and closed at site ``j``.

**Open boundary conditions** (`g.periodic == false`): returns ``(1,2), (2,3), \\ldots, (L-1, L)``, a total of ``L-1`` bonds.

**Periodic boundary conditions** (`g.periodic == true`): returns the same ``L-1`` bulk bonds plus the wrap-around bond ``(L, 1)``, giving ``L`` bonds in total and closing the chain into a ring.

# Examples

```jldoctest
julia> bonds(Chain(4))
3-element Vector{Tuple{Int64, Int64}}:
 (1, 2)
 (2, 3)
 (3, 4)

julia> bonds(Chain(4, true))
4-element Vector{Tuple{Int64, Int64}}:
 (1, 2)
 (2, 3)
 (3, 4)
 (4, 1)
```
"""
function bonds(g::Chain)
    L = g.L
    if g.periodic
        [(i, mod1(i + 1, L)) for i in 1:L]
    else
        [(i, i + 1) for i in 1:(L - 1)]
    end
end
