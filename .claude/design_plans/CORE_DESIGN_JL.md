# Modeling quantum systems on lattices (Julia design)

## Implementation milestones

| Milestone | Target exercises | Status | Components |
|---|---|---|---|
| 1 | Ex 1–5 | **complete (v0.1–v0.3)** | `AbstractDoF` · `hilbert_space` · `TIx` / `MultiIx` / `Partition` / `Bipartition` · `IndexedTensor` · `Bond` · `TensorSVD` · `tensor_svd` + `AbstractTruncation` · `FiniteMPS` · `CanonicalForm` · `ArbitraryForm` |
| 2 | Ex 6–9, Final | **complete (v0.4–v0.6)** | `FiniteMPO` + `IdL`/`IdR` · `to_vidal` / `to_canonical` · `VidalForm` · `CompositeDoF` · `apply_gate!` · `trotter_step!` · `local_expectation` · `entanglement_spectrum` |
| 3 | Ex 10 | **complete (v0.7)** | `DenseHamiltonian` (SparseArrays / KrylovKit) |
| Future | — | deferred | `Lattice{V,E}` + MetaGraphsNext.jl · `:tensorkit` backend dispatch · `TIx` contraction engine |

Milestones 1, 2, and 3 are complete. Items in the Future milestone are specced here for design continuity but are not implemented until an algorithm actually needs them.

---

## Aspects of physical modeling

This section establishes three Milestone 1 building blocks:

1. The `AbstractDoF` hierarchy (`Spin{S}`, `Fermionic`, `HardCoreBoson`) as the compile-time representation of local physics. The type parameter `D` on `AbstractSite{D}` encodes the degree of freedom at each site; the compiler specialises all dispatch on it.
2. Two concrete site types encoding the state-vs-operator distinction at the type level: `StateSite{D}` (one open physical leg, for MPS/PEPS) and `OperatorSite{D}` (two open physical legs bra + ket, for MPO/PEPO).
3. The `hilbert_space` interface as the single bridge between `AbstractDoF` types and TensorKit's space system. Every subsequent type that needs a local Hilbert space goes through this one function.

`IndexedTensor` is the vertex type for abstract tensor network work without physical content. It is specced in section 2 and needs no `Site` wrapper when there is no physics to attach.

The topology model — `Lattice{V, E}`, where sites become typed vertices and contracted bonds become typed edges — is correct long-term architecture but is deferred to a future milestone. For 1D chain exercises (Ex 1–9, Final), topology is fully implicit in the `FiniteMPS`/`FiniteMPO` tensor ordering. The `Lattice` design is preserved at the end of this section for reference.

### Physical lattice sites: `AbstractSite{D}`, `StateSite{D}`, and `OperatorSite{D}`

- The root of the site hierarchy is `AbstractSite{D <: AbstractDoF}`, where `D` is a degree of freedom type. `AbstractDoF` is the abstract type at the root of the DoF hierarchy:
    ```julia
    abstract type AbstractDoF end
    abstract type AbstractSite{D <: AbstractDoF} end
    ```
- The constraint `D <: AbstractDoF` on `AbstractSite` makes the parameterization semantic: it says the type parameter _must_ be a real physical degree of freedom, which rules out meaningless instantiations like `AbstractSite{Int}` or a "no-physics" placeholder.

- `Spin`, `Fermionic`, and `HardCoreBoson` are the concrete DoF types where the local Hilbert space is determined by the physics: 
    ```julia
    struct Spin{S} <: AbstractDoF end       # the DoF variable is a spin residing on the site with magnitude S = 1//2, 1, 3//2, ...
    struct Fermionic <: AbstractDoF end     # the DoF variable is a fermion (i.e the fermionic anti-commutation rules must be obeyed)
    struct HardCoreBoson <: AbstractDoF end # the DoF variable is a hardcore boson (i.e a boson but only one boson can occupy a site at once)
    ```

- The same `Lattice{V, E}` graph infrastructure works regardless of the vertex type. For physical models the only thing that changes is `D` — one usually switches DoF types when solving for the ground state via transformations like Jordan-Wigner. For abstract TN work the vertex type switches from `AbstractSite{D}` to `IndexedTensor` entirely.

Two concrete `AbstractSite{D}` subtypes capture the state-vs-operator distinction at the type level:

```julia
struct StateSite{D <: AbstractDoF}    <: AbstractSite{D} end  # MPS/PEPS: 1 open physical leg
struct OperatorSite{D <: AbstractDoF} <: AbstractSite{D} end  # MPO/PEPO: 2 open physical legs (bra + ket)
```

Because `D` now always encodes a real physical degree of freedom and the state-vs-operator role is encoded in the concrete subtype, the compiler knows the full open-leg structure at specialisation time with no runtime dispatch: a `StateSite{Spin{1//2}}` always has exactly one open leg of dimension 2; an `OperatorSite{Spin{1//2}}` always has exactly two open legs of dimension 2.

**Connecting to TensorKit: the `hilbert_space` interface**

Each `AbstractDoF` subtype implements a single interface function that returns the local Hilbert space as a TensorKit `ElementarySpace`:

```julia
hilbert_space(::Spin{S})       = Rep{SU2}(S => 1)
hilbert_space(::Fermionic)     = Rep{FermionParity}(0 => 1, 1 => 1)
hilbert_space(::HardCoreBoson) = ComplexSpace(2)
```

This one function is the bridge between the `AbstractDoF` hierarchy and TensorKit's space system. The physical leg on any `AbstractSite{D}` gets its TensorKit space from `hilbert_space(dof)`, so symmetry structure flows automatically from the DoF type into the tensor, with no extra wiring.

The draft docstrings below capture the theoretical reasoning behind each method. When the actual source files are written these can be extended with argument tables, type annotations, and return-type documentation; the physics commentary should be kept as-is.

```julia
"""
    hilbert_space(dof::AbstractDoF) -> TensorKit.ElementarySpace

Return the local Hilbert space associated with degree of freedom `dof` as a
TensorKit `ElementarySpace`.

The local Hilbert space at a lattice site is the vector space in which the
quantum state of the entity residing at that site lives. For physical degrees
of freedom (spin, fermion, hard-core boson) this space carries the
representation theory of a symmetry group: states are labelled by quantum
numbers (irrep labels), and the space decomposes into a direct sum of
irreducible representation spaces. For abstract TN work without a physical DoF,
the space is a plain finite-dimensional complex vector space ℂ^d obtained directly
from the `TIx` dimension, with no symmetry structure.

In TensorKit, `ElementarySpace` is the base type for all one-index spaces.
The two concrete subtypes relevant here are `ComplexSpace` (unstructured) and
`Rep{G}` (representation space of a group G, carrying sector decomposition).

# TODO: add @arg dof, @return annotation, @example block when implementing
"""
function hilbert_space end

"""
    hilbert_space(::Spin{S}) -> Rep{SU2}

The local Hilbert space of a spin-S site is the (2S+1)-dimensional irreducible
representation (irrep) of SU(2). For half-integer S (e.g. S = 1/2) the
representation is spinorial; for integer S it is tensorial.

In Dirac notation the basis states are |S, mₛ⟩ with mₛ ∈ {-S, -S+1, ..., S},
giving 2S+1 states in total. The irrep is labelled by the spin quantum number S
itself (the highest weight), not by the dimension.

TensorKit encodes this as `Rep{SU2}(S => 1)`: one copy of the irrep with
highest-weight label S. The `1` is the multiplicity -- the number of times
this irrep appears in the space. A single site always has multiplicity 1; the
multiplicity grows when taking tensor products of multiple sites.

Examples:
- Spin-1/2 (qubit, two-level system): `Rep{SU2}(1//2 => 1)`, dimension 2
- Spin-1 (three-level system, e.g. NV center ground state): `Rep{SU2}(1 => 1)`, dimension 3
- Spin-3/2: `Rep{SU2}(3//2 => 1)`, dimension 4

# TODO: add @arg, @return, @example when implementing
"""
hilbert_space(::Spin{S}) where {S}

"""
    hilbert_space(::Fermionic) -> Rep{FermionParity}

The local Hilbert space of a spinless fermionic site has two basis states:
|0⟩ (empty, particle-number parity 0) and |1⟩ (occupied, parity 1). The
relevant symmetry is fermionic parity ℤ₂: fermionic systems obey a
superselection rule that forbids superpositions of states with different
particle-number parity. This is a physical constraint, not a gauge choice.

TensorKit encodes this as `Rep{FermionParity}(0 => 1, 1 => 1)`: one
one-dimensional space in each ℤ₂ sector (even parity and odd parity). The
block-sparse structure this imposes on `TensorMap`s is what enforces parity
conservation in all tensor operations automatically -- no explicit parity
checking in algorithm code is needed.

Note: a spinful (spin-1/2) fermion site has four states (|0⟩, |↑⟩, |↓⟩,
|↑↓⟩) and requires a product of fermionic parity and SU(2) spin symmetry.
`Fermionic` here models the spinless (or spin-polarized) case only.

# TODO: add @arg, @return, @example when implementing
"""
hilbert_space(::Fermionic)

"""
    hilbert_space(::HardCoreBoson) -> ComplexSpace(2)

The local Hilbert space of a hard-core boson site is two-dimensional: |0⟩
(empty) and |1⟩ (occupied), identical in dimension to a spinless fermion.
The key difference from `Fermionic` is that bosons are not subject to
fermionic parity superselection. Superpositions of |0⟩ and |1⟩ are physically
allowed, so no ℤ₂ block structure is imposed.

TensorKit encodes this as `ComplexSpace(2)`: a plain 2D complex vector space
with no symmetry sectors. The distinction from `Fermionic` only matters in `:tensorkit` mode,
where `Fermionic` gives block-sparse tensors (ℤ₂ sector structure) and `HardCoreBoson` does not.

# TODO: add @arg, @return, @example when implementing
"""
hilbert_space(::HardCoreBoson)

```

In `:native` mode (plain arrays), `hilbert_space` can return a bare `ComplexSpace(d)` -- just the dimension, no symmetry sectors. In `:tensorkit` mode, it returns the full representation space carrying quantum number information. The DoF types themselves don't need to know which backend is active; the backend switch (via `ScopedValues` from ADR 0003) determines which behavior applies.

The **overall Hilbert space** of a model is the tensor product of all site spaces, $\bigotimes_i \operatorname{hilbert\_space}(D_i)$. For a homogeneous model (all sites with the same DoF), TensorKit computes this automatically and organizes the full space by symmetry sector.

The real payoff from symmetry-aware spaces is in the **bond spaces** (virtual indices between sites). A bond carrying only `ComplexSpace(\chi)` is non-symmetric. Once the site DoF types carry their symmetry and that information propagates into the bond spaces (constraining bonds to decompose into representations of the same group), all tensors become block-sparse `TensorMap`s. TensorKit then handles contractions entirely within blocks, which is dramatically cheaper at large bond dimensions. This is the deferred upgrade to virtual bond legs mentioned in the `IndexedTensor` backend section — `TIx{Upper/Lower}` carry only `ndim` today; they will need to carry `ElementarySpace` objects once the `:tensorkit` backend is activated.

`VectorInterface.jl` and `MatrixAlgebraKit.jl` fit into the picture at the tensor operation level (not the DoF level): `VectorInterface.jl` gives the vector arithmetic interface for quantum states, and `MatrixAlgebraKit.jl` provides SVD and eigendecompositions used in MPS normalization and DMRG sweeps. Both operate on `IndexedTensor` objects regardless of whether the underlying spaces carry symmetry or not.

### Future milestone: topology via `Lattice{V, E}`

The subsections below spec the `Lattice{V, E}` graph model and its MetaGraphsNext.jl backing. This is the correct long-term architecture for non-1D topologies (PEPS, tree TN, arbitrary geometry), but it has no implementation dependency in Milestones 1–3. Kept here for design continuity; implement only when a 2D algorithm or non-trivial topology is actually next.

#### Why tensor networks are NOT ordinary graphs, and how to reconcile that

This is an important mathematical point. In an ordinary graph, every edge connects exactly two vertices. A tensor network has two kinds of legs:

- **Contracted (closed) legs**: a virtual bond connecting two tensors. One leg on each tensor, paired together and summed over. These DO behave like graph edges.
- **Open (boundary) legs**: a physical index, or a virtual leg at the edge of the network, with only one end attached to a tensor. These do NOT fit into standard graph theory at all.

The nLab page on tensor networks ([BCJ10: Biamonte, Clark, Jaksch 2011](https://ncatlab.org/nlab/show/tensor+network#BCJ10)) formalizes this: tensor networks are string diagrams in a symmetric monoidal category. Closed wires are morphism compositions (contraction), and boundary/open wires are the free indices of the resulting composite morphism. Trying to force open legs into graph edges breaks the mathematical structure.

The reconciliation is clean: **split the representation along exactly this line.**

- The **graph** (`Lattice{V, BondLeg}`, backed by MetaGraphsNext.jl) represents only the **closed wires**: contracted bonds between sites. Every edge in this graph is a `BondLeg` (a contracted pair of legs). This is a proper graph and MetaGraphsNext.jl handles it perfectly.
- The **open legs** belong to the vertex, not the graph. For a `StateSite{Spin{1//2}}` vertex, the open leg is the physical spin index of dimension 2. For an `IndexedTensor` vertex (abstract TN without physics), the open legs are whichever `TIx` entries remain uncontracted in the tensor's index tuple.

Together the graph (closed wires) and the per-site open legs (boundary wires) make a complete string diagram. Nothing is lost, and the graph part stays a proper graph.

To make this concrete with an MPS example:

- A tensor at an interior abstract MPS site (no physics) has 2 contracted virtual bonds (graph edges to left and right neighbors) and 1 open physical leg. The vertex is an `IndexedTensor`; the open leg is a free `TIx` in its index tuple, NOT a graph edge.
- A tensor at a boundary abstract MPS site has 1 contracted virtual bond (graph edge) and 2 open legs: the physical leg and the boundary virtual leg (both free `TIx` entries on the `IndexedTensor`).
- A spin-1/2 site in a Heisenberg chain uses `StateSite{Spin{1//2}}` as the vertex type. It has interaction terms connecting to 2 neighbors (graph edges via `InteractionTerm`) and 1 open physical spin index of dimension 2, determined by `D = Spin{1//2}` at compile time.

#### Lattice implementation via MetaGraphsNext.jl

The `Lattice{V, E}` type is backed by [`MetaGraphsNext.jl`](https://github.com/JuliaGraphs/MetaGraphsNext.jl). Making it parametric over vertex data `V` and edge data `E` keeps both use cases (tensor network topology and physical geometry) under the same type:

```julia
struct Lattice{V, E}
    graph::MetaGraph{Int, SimpleGraph{Int}, V, E, Nothing}
end
```

For a physical Heisenberg chain (MPS), `V = StateSite{Spin{1//2}}` and `E = BondLeg`. For a physical lattice with explicit Hamiltonian terms, `E = InteractionTerm` carrying the relevant parameters (interaction type, coupling strength, distance, etc.). For a Heisenberg MPO, `V = OperatorSite{Spin{1//2}}` and `E = BondLeg`. For an abstract MPS without physical content, `V = IndexedTensor{T, N, D}` and `E = BondLeg`. Graph algorithms (neighbor queries, path-finding, boundary detection for open vs. periodic boundary conditions, connected components) work the same way for any `V` and `E`.

**Enumeration is already solved.** MetaGraphsNext.jl assigns each vertex an integer code internally. That code locates any site in the `Lattice`. No custom indexing logic needed.

**`Block` as a vertex subset.** A `Block` is just a `Set{Int}` of vertex codes within a parent `Lattice`. Multiple `Block`s can overlap and share the same `Lattice` without copying any graph data.

**Graph algorithms for free.** MetaGraphsNext.jl subtypes `Graphs.jl`'s `AbstractGraph`, so path-finding, neighbor lookups, connected component detection, etc., all work on any `Lattice{V, E}` directly.

**References**

- [Graphs.jl](https://juliagraphs.org/Graphs.jl/dev/): the core graph interface and algorithm library (the NetworkX equivalent in Julia).
- [MetaGraphsNext.jl](https://juliagraphs.org/MetaGraphsNext.jl/dev/): type-stable graphs with typed vertex and edge metadata, the backing store for `Lattice`.
- [nLab: Tensor network](https://ncatlab.org/nlab/show/tensor+network#BCJ10): the categorical foundation for why tensor networks are string diagrams in a monoidal category, not ordinary graphs. BCJ10 (Biamonte, Clark, Jaksch 2011) is the key reference.

---

> **Detail files** (read on demand, not auto-loaded):
> - `CORE_DESIGN_JL_math.md` — index types (`AbstractIndex`, `TIx`, `MultiIx`), SVD design (`Bipartition`, `group_legs`, `TensorSVD`), `IndexedTensor` + backends
> - `CORE_DESIGN_JL_mps.md` — `FiniteMPS`, `FiniteMPO`, `AbstractMPSForm`, design decisions comparison table
