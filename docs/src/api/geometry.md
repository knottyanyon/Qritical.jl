# Geometry

## Physics motivation

A **geometry** answers exactly two questions: which sites exist, and which sites are
connected by bonds. This minimal interface is all that the Hamiltonian builder (§5/§7)
and the MPO constructor (§6) consume, so swapping a 1D open chain for a periodic chain
or a 2D lattice requires no changes anywhere else.

The 1D `Chain` is the workhorse for the course: finite MPS and DMRG are formulated for
open boundary conditions, where the bipartition structure is cleanest and the bond
dimension grows monotonically from each boundary. Periodic boundaries are supported but
inflate bond dimension and break the standard left/right canonical structure of an MPS —
use them with awareness of the extra cost.

The `sites` / `bonds` seam is what future geometries (`Square`, `Torus`, general
`Lattice{V,E}`) plug into: implement those two methods and every Hamiltonian constructor
works unchanged.

---

## Quick Reference

**Types:** [`AbstractGeometry`](@ref) · [`Chain`](@ref)

**Functions:** [`sites`](@ref) · [`bonds`](@ref)

---

## Types

```@docs
AbstractGeometry
Chain
```

## Functions

```@docs
sites
bonds
```
