# Modeling quantum systems on lattices (Julia design)

## Implementation milestones

| Milestone | Target exercises | Status | Components |
|---|---|---|---|
| 1 | Ex 1–5 | **complete (v0.1–v0.3)** | `AbstractDoF` · `hilbert_space` · `TIx` / `MultiIx` / `Partition` / `Bipartition` · `IndexedTensor` · `Bond` · `TensorSVD` · `tensor_svd` + `AbstractTruncation` · `FiniteMPS` · `CanonicalForm` · `ArbitraryForm` |
| 2 | Ex 6–9, Final | **next (v0.4–v0.5)** | `FiniteMPO` + `IdL`/`IdR` · `to_vidal` / `to_canonical` · `VidalForm` · `CompositeDoF` · `apply_gate!` · `trotter_step!` |
| 3 | Ex 10 | not started (v0.7) | `DenseHamiltonian` (SparseArrays / KrylovKit) |
| Future | — | deferred | `Lattice{V,E}` + MetaGraphsNext.jl · `:tensorkit` backend dispatch · `TIx` contraction engine |

Milestone 1 is complete. Milestone 2 is the next priority (final assignment deadline: 02.09.2025). Items in the Future milestone are specced here for design continuity but are not implemented until an algorithm actually needs them.

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


## Aspects of mathematical modeling

- To connect the abstract notions of physical sites on lattices to concrete mathematical objects, we need the idea of indices.
- Like Penrose, we use the word "valence" or "order" instead of "rank" when referring to the number of indices on a tensor, since "rank" is already used in the context of SVD and the truncation of singular values. A tensor with $n$ indices up and $m$ down is called a "valence-$(n, m)$ tensor". It can also be called a "valence-$k$ tensor" for $k = n + m$.
  - The $k$-particle quantum state of a system residing in a `Lattice` with $k$ sites can be described as an order-$k$ / valence-$k$ tensor.
  - I also want to have something called a `Leg` to denote the valence legs of a tensor. Each valence leg has an index associated with it, and an order-$N$ tensor will have $N$ legs, one for each index. The number of distinct values that a valence leg can take gives the `ndim` of the index associated with that leg.
  - A `Leg` can remain "open" (one end is free, the other end connected to a site) or "contracted" (both ends connected to sites).
  - If both ends of a `Leg` are connected to the same `Site`, that implies a "trace" operation.
  - A `Leg` needs a label, usually a Greek letter:
    - Letters like $\alpha, \beta$ are used for bond legs (also called virtual legs).
    - Letters like $\sigma$ are used for physical legs.
    - A `Leg` should have some field that captures its orientation. A `Leg` can be thought of as going `In` to a `Site` or `Out` from a `Site`. The orientation lets us draw arrow directions on the `Leg`, and these arrow directions are important because they determine whether the index corresponding to the `Leg` is covariant or contravariant. Getting this right makes contractions much cleaner and makes the notion of left/right normalization in the canonical form of a tensor network much more intuitive.
  - A `Bond` captures the idea of entanglement between two `Site`s, and a `BondLeg` then denotes a contraction over the indices corresponding to the `Leg`s of those sites.
    - A `BondLeg` is a pair of `Leg`s. (T.B.D.: whether it holds references or copies of those `Leg`s.)
    - The orientation of the `BondLeg` is determined by which of the two `Leg`s is considered the "from" end and which is the "to" end.
    - The arrow direction implies the following:
      - The `Site` to which the from-leg is attached has that `Leg` going outward from it, and the corresponding index on a tensor belonging to that `Site` is a lower (down) index.
      - The `Site` to which the to-leg is attached has that `Leg` coming inward to it, and the corresponding index on a tensor belonging to that `Site` is an upper (up) index.
      - This also clearly shows how the Einstein summation convention in covariant notation can be enforced, i.e., only a pair of one upper and one lower index can be contracted.
    - Similar to how a `Leg` belongs to a `Site`, a `BondLeg` belongs to a `Bond`.
    - Since the purpose of a `Bond` is to capture entanglement between two sites, it also carries details about any cut-off dimension used when doing SVD on that link. This helps track truncation errors and related quantities.

- I feel that the ability to attach indices directly to a tensor object, separately from the `Site`s that capture the physics, is important for two main reasons:
  - It lets me do tensor calculations without being tied to any specific physical system.
  - It helps separate different tensor representations of the same system, since different tensor network structures correspond to different ways of bookkeeping entanglement.
- The `Leg` / `BondLeg` concepts then serve as the connecting bridge between the physical picture of `Site`s / `Bond`s and the mathematical side of indexed tensors.
- With the help of indices, we can now build a complete picture of quantum states, operators, and observables using indexed tensors.

### Index types

In Julia, the natural way to model a family of index objects is through **abstract types and parametric structs**. This captures everything that a Python metaclass hierarchy was trying to enforce, but in a way that integrates cleanly with Julia's type system and multiple dispatch.

**The interface requirements: `ndim` and `label`**

The root of the index hierarchy is an abstract type:

```julia
abstract type AbstractIndex end
```

Any concrete index type is a subtype of `AbstractIndex`. Two functions form the required interface:

- `ndim(::AbstractIndex) → Int` — the number of distinct values the index can take
- `label(::AbstractIndex) → Symbol` — the index's name (e.g. `:σ`, `:αL`)

Both are expressed as fallback methods that throw an error when a subtype has not implemented them. This is Julia's standard pattern for defining an interface: you do not enforce it at the struct definition level, you enforce it at first use through dispatch.

`label` is used by `MultiIx` to auto-generate composite labels and by `group_legs` when constructing `MultiIx` objects from `Partition` contents — so any new `AbstractIndex` subtype must implement it.

We only use covariant indexing in this project and skip generic (non-covariant) indexing altogether, since everything can be stated cleanly in covariant form anyway.

**Covariant index types**

We follow the standard physics convention: upper indices are contravariant and lower indices are covariant. To illustrate: if we have a vector $\Psi_\alpha$ and expand it in a basis $\{\varphi_i\}$, the covariant convention puts the vector label at the bottom and the expansion coefficient at the top, $\Psi_\alpha = \psi_\alpha^i \varphi_i$, where $\psi_\alpha^i$ is the basis expansion coefficient.

The key design choice here is to encode direction as a **type parameter** rather than using two separate concrete struct definitions for up and down indices. Something like:

```julia
abstract type IndexLoc end
struct Upper <: IndexLoc end
struct Lower <: IndexLoc end

struct TIx{L <: IndexLoc} <: AbstractIndex
    label::Symbol   # :σ, :αL, :αR, etc. -- identifies this leg by name
    ndim::Int
    function TIx{L}(label::Symbol, ndim::Int) where {L<:IndexLoc}
        ndim > 0 || throw(ArgumentError("TIx ndim must be positive, got $ndim"))
        new{L}(label, ndim)
    end
end
```

The inner constructor enforces the invariant that `ndim > 0`; a zero- or negative-dimensional index is physically meaningless and caught at construction rather than silently producing wrong results downstream.

For everyday use, prefer the named constructor helpers over the parametric form directly:

```julia
upper(label::Symbol, ndim::Int) = TIx{Upper}(label, ndim)
lower(label::Symbol, ndim::Int) = TIx{Lower}(label, ndim)

# batch form for creating several same-position indices in one call
uppers(pairs::Pair{Symbol,Int}...) = Tuple(TIx{Upper}(p.first, p.second) for p in pairs)
lowers(pairs::Pair{Symbol,Int}...) = Tuple(TIx{Lower}(p.first, p.second) for p in pairs)

# site-indexed label helper for MPS/MPO construction
bond_label(base::Symbol, site::Int) = Symbol(base, site)
```

**Site-indexed labels and the deferred `Label` struct**

`bond_label` generates labels like `:α1`, `:α2` for MPS virtual bonds. It is a plain `Symbol`-returning function — no new type is introduced. This is the deliberate choice for now; the full implications are documented below for when the decision comes up again.

In canonical sweep algorithms, each virtual bond gets a unique label via `bond_label`:

```julia
upper(bond_label(:χ, i - 1), χL)   # left virtual leg of site i  — TIx{Upper}, inward
lower(bond_label(:χ, i),     χR)   # right virtual leg of site i — TIx{Lower}, outward
```

**Virtual bond direction convention (settled in v0.3):** a right virtual leg goes *outward* from site `i` → `TIx{Lower}`; a left virtual leg comes *inward* to site `i` → `TIx{Upper}`. Physical legs (`σ`) are lower (covariant). This matches the Penrose from/to arrow convention documented below.

**The `Label` struct decision (2026-06-03):** `bond_label` with `Symbol` is the current implementation. A `Label{base,site}` struct was considered for v0.3 sweep code but deferred — `Symbol` parsing was not required in practice. Revisit if a contraction engine needs to recover the base label from a bond label programmatically.

`TIx{Upper}` is the contravariant (upper) index, and `TIx{Lower}` is the covariant (lower) index. Any behavior that differs between the two is handled by multiple dispatch on the type parameter, not by creating two separate types. This avoids having two nearly identical struct definitions for what is really the same structure with a position tag attached.

The `label` field is required, not optional. Its role is contraction matching: when two tensors are contracted, the engine pairs a `TIx{Upper}` on one tensor with a `TIx{Lower}` on the other that carries the same label. This is exactly how QSpace identifies legs via its `itag` field -- purely positional matching is fragile as soon as tensors have more than two legs. The label also makes tensor expressions readable: `αL` and `αR` immediately tell you which leg is the left bond and which is the right bond of an MPS site tensor, without having to count index positions.

For tensor transposition in covariant notation: $(A^T)^i_j = A^i_j$. Note that not just the up/down position of an index matters but also the order in which the indices appear. The full set of indices on a valence-$N$ tensor is treated as an ordered $N$-tuple, so `A[Lower(j), Upper(i)]` is considered distinct from `A[Upper(i), Lower(j)]`.

**How `Leg` orientation connects to index direction**

- If a `Leg` is going outward from the `Site` it is attached to, the corresponding index on that `Site`'s tensor is a down index.
- If a `Leg` is coming inward to the `Site` it is attached to, the corresponding index on that `Site`'s tensor is an up index.

Raising and lowering indices uses the metric tensor for the chosen basis. In the orthonormal basis we work with in quantum mechanics, the metric tensor is just the Kronecker delta.

**Grouped indices**

Grouping comes in handy when doing SVD on a tensor of order $k > 2$: we pick which indices to treat as the "row" direction and which to treat as the "column" direction to reshape the valence-$k$ tensor into a matrix. Once grouped, each of the two groups becomes a single combined index, and the reshaped tensor looks like a matrix with those two combined indices as its two legs. This lets us apply SVD to tensors that are not already matrices.

To make this work, we need a `MultiIx` type. Like any other index, a `MultiIx` carries a label (a Greek letter or a combined label like $(\alpha, \beta)$ assigned to the new reshaped leg). It holds a tuple of constituent `AbstractIndex` values, and its `ndim` is not stored directly but computed on demand as the product of the constituent `ndim`s. In Julia this looks like:

```julia
struct MultiIx <: AbstractIndex
    label::Symbol
    indices::Tuple{Vararg{AbstractIndex}}
end

label(g::MultiIx) = g.label
ndim(g::MultiIx)  = prod(ndim, g.indices; init=1)  # init=1: empty product is the scalar dimension
```

This fits cleanly into the `AbstractIndex` interface since `ndim` is already expected to be a function rather than a stored field. The `MultiIx` just computes it from the reshaping instead of carrying it explicitly.

The `label` argument is optional. When omitted, the label is auto-generated by concatenating the constituent index labels:

```julia
_autolabel(indices) = isempty(indices) ? :scalar : Symbol(join(String.(label.(indices))))

MultiIx(indices::Tuple{Vararg{AbstractIndex}}) = MultiIx(_autolabel(indices), indices)
MultiIx(indices::AbstractIndex...)             = MultiIx(indices)   # varargs convenience
```

So `MultiIx(upper(:α, 2), lower(:β, 3))` produces `MultiIx(:αβ, ...)` automatically, and `MultiIx(())` produces `MultiIx(:scalar, ())`.

### SVD on indexed tensors

Nearly every tensor network algorithm in the course reduces to the same pattern: take a tensor sitting at a site (or spanning two sites connected by a bond), pick a bipartition of its indices, SVD the reshaped tensor, keep some singular values according to a truncation rule, and update the bond dimension. Canonicalization sweeps apply this left-to-right and then right-to-left; TEBD applies it after each two-site gate; variational MPS optimization (DMRG-lite) applies it after each site update to restore canonical form. Because the underlying operation is always the same, a single well-designed `tensor_svd` interface covers all of these.

SVD on an `IndexedTensor` of order $k > 2$ is a three-step process, and conveniently, each step maps directly to a design concept already in place.

**Step 1: Bipartition and `group_legs`**

Pick a bipartition of the $k$ indices into a "left" group and a "right" group. Encode each group as a `Partition` (an ordered `Vector{AbstractIndex}`) and combine them into a `Bipartition`. The `Bipartition` constructor eagerly checks that the two partitions share no indices; the subsequent `group_legs` call checks that together they cover all legs of the tensor — a partial bipartition would otherwise surface as a confusing `permutedims` error rather than a clear message.

`group_legs(A, bp::Bipartition)` performs the permute + reshape and returns a 2-leg `IndexedTensor` whose row axis is an auto-labeled `MultiIx` over `bp.left.indices` and whose column axis is an auto-labeled `MultiIx` over `bp.right.indices`. No numerical data moves during the permute (only strides change); the reshape is a view. The full validation sequence inside `group_legs` is:

```julia
# 1. resolve positions — throws ArgumentError with the missing index name if not found
left_pos  = [_resolve(idx, A) for idx in bp.left.indices]
right_pos = [_resolve(idx, A) for idx in bp.right.indices]

# 2. coverage check — clear error listing uncovered indices
covered = length(left_pos) + length(right_pos)
covered == ndims(A) || throw(ArgumentError(
    "bipartition covers $covered/$(ndims(A)) tensor indices — uncovered: $(setdiff(A.indices, vcat(bp.left.indices, bp.right.indices)))"
))
```

Convenience constructors avoid manually building both `Partition`s when one side is implicit:

```julia
complement(p::Partition, A::IndexedTensor)        # all indices in A not in p
bipartition(left::Partition, A::IndexedTensor)    # pairs left with complement(left, A)
```

**Step 2: SVD with truncation — the full factorisation**

The SVD of $A$ is the three-tensor factorisation

$$A^i{}_j = U^i{}_\lambda \; \Sigma^\lambda{}_{\lambda'} \; (V^\dagger)^{\lambda'}{}_j$$

where $\Sigma$ is diagonal.  There are **two distinct bonds**:
- **Bond₁** — $U$'s lower leg $\lambda$ contracts with $\Sigma$'s upper leg $\lambda$
- **Bond₂** — $\Sigma$'s lower leg $\lambda'$ contracts with $V^\dagger$'s upper leg $\lambda'$

Because a lower index on one tensor and an upper index on the same tensor implies a *trace*, the two bond labels must be **distinct**.  Qritical derives them from the bipartition's `MultiIx` autolabels, prefixed with `:χ`:

```
λ_label  = Symbol(:χ, label(M.indices[1]))   # Bond₁  e.g. :χvLσ
λ′_label = Symbol(:χ, label(M.indices[2]))   # Bond₂  e.g. :χvR
```

`Σ` is returned as `IndexedTensor{real(T), 2, Diagonal{real(T), Vector{real(T)}}}` — a genuine two-leg tensor backed by Julia's `Diagonal` (compact storage, no dense allocation).  Singular values are always real even when the input tensor has complex entries.

**Truncation strategies** (implemented, `AbstractTruncation` hierarchy):

```julia
abstract type AbstractTruncation end
struct KeepFirst     <: AbstractTruncation; r::Int    end
struct KeepAbove     <: AbstractTruncation; atol::Float64  end
struct KeepRelative  <: AbstractTruncation; rtol::Float64  end
struct KeepMachineEps <: AbstractTruncation end   # sqrt(eps(T)) * σ₁; self-calibrating
```

The `normalize` keyword (default `false`) divides the diagonal of `Σ` by `‖A‖_F` to yield Schmidt coefficients.  `ε` (the 2-norm of discarded singular values) is always returned in raw form and stored in `TensorSVD.ε`.

**Step 3: Re-indexing and `TensorSVD`**

`tensor_svd` returns a typed result struct (not a named tuple):

```julia
struct TensorSVD{Element, SingularElement, UOrder, VdOrder, UData, VdData}
    U          :: IndexedTensor{Element,         UOrder,  UData}
    Σ          :: IndexedTensor{SingularElement, 2,       Diagonal{SingularElement,Vector{SingularElement}}}
    Vd         :: IndexedTensor{Element,         VdOrder, VdData}
    ε          :: SingularElement
    normalized :: Bool
end
```

`SingularElement = real(Element)` is enforced in the inner constructor.  Supports both named `(; U, Σ, Vd, ε) = F` and positional `U, Σ, Vd, ε = F` destructuring via `Base.iterate`.

Index layout after the factorisation:
- `U.indices`  = `(bp.left.indices..., lower(λ_label, r))`   — original left legs + outward bond leg
- `Σ.indices`  = `(upper(λ_label, r), lower(λ′_label, r))`  — upper Bond₁, lower Bond₂
- `Vd.indices` = `(upper(λ′_label, r), bp.right.indices...)` — inward bond leg + original right legs

**`Bond` struct**

`Bond` records a specific contraction — the pair of `TIx` legs that are summed over — plus optional SVD metadata:

```julia
struct Bond
    lower :: TIx{Lower}           # the from-leg (outward, on the U-side tensor)
    upper :: TIx{Upper}           # the to-leg   (inward,  on the Vd-side tensor)
    trunc :: Union{AbstractTruncation, Nothing}
    ε     :: Float64
end
```

`Bond` is **not** an `AbstractIndex`; it is a record of a contraction, not a leg type.  Virtual legs on site tensors are plain `TIx{Upper}` (inward) or `TIx{Lower}` (outward).

**Truncation error accumulation**

`ε = ‖discarded singular values‖₂` is returned in `TensorSVD.ε`.  Accumulating ε across a sweep gives the total truncation error for an MPS compression — the standard diagnostic for checking whether bond dimension was large enough.

**Backend note.** `LinearAlgebra.svd` calls LAPACK via Julia's standard bindings.  When the `:tensorkit` backend is introduced the entry point will dispatch via `Val(current_backend())` — algorithm code never changes.  TensorKit v0.15+ uses MatrixAlgebraKit internally; the relevant API is `svd_compact` / `svd_trunc`.

### The `IndexedTensor` type and backends

The central type that brings everything together is `IndexedTensor`. It combines two things: a tuple of `AbstractIndex` values (the index metadata from the type hierarchy above) and a numerical backing array (the actual numbers). The point is to keep these two concerns completely separate.

The key design choice is to make `IndexedTensor` parametric over its backing store type, along the lines of:

```julia
struct IndexedTensor{T, N, D <: AbstractArray{T, N}} <: AbstractArray{T, N}
    data::D
    indices::NTuple{N, AbstractIndex}
end
```

The type parameter `D` is what makes backends possible. The algorithm code describes contractions in terms of the index algebra. The backend decides how those contractions are actually executed. Swapping `D` changes the execution engine without touching a single line of algorithm code.

**The two backends**

There are two backends, selected by a symbol: `:native` and `:tensorkit`.

- In `:native` mode, `D` is a plain `Array{T, N}`. This is the default. It is easy to inspect, easy to debug, and the `@tensor` macro from [`TensorOperations.jl`](https://github.com/QuantumKitHub/TensorOperations.jl) works out of the box. Good for learning and checking index algebra.
- In `:tensorkit` mode, `D` becomes a [`TensorKit.TensorMap`](https://jutho.github.io/TensorKit.jl/stable/man/tensors/). TensorKit stores data in a block-sparse format organized by symmetry sector. It is also what `TensorOperations.jl` was originally built for, so `@tensor` keeps working with the same syntax. This is the production path.

The goal is that the same contraction code runs in both modes. You write your algorithm once and choose the backend separately.

**Scoped context, not a global flag**

The active backend is held in a [`ScopedValue`](https://docs.julialang.org/en/v1/base/scopedvalues/) from Julia's standard library (available since 1.11). The interface looks like:

```julia
with_backend(:tensorkit) do
    # everything in here uses the TensorKit-backed path
end
# back to :native here
```

A global flag would be simpler to implement, but it breaks as soon as you want to benchmark one backend against the other in the same session, or run two independent pieces of code with different backends. The scoped approach handles that cleanly. It follows the same pattern as [JAX's `with jax.default_device(...)`](https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices). The `:native` default means existing code never breaks; switching to `:tensorkit` is always an explicit opt-in.

**The bridge from index metadata to TensorKit**

The connection between `AbstractIndex` values and TensorKit spaces is built into the index types themselves:

- A `PhysicalIndex` stores a reference to its `StateSite` or `OperatorSite`, and each `AbstractSite{D}` carries the Hilbert space for that site as a [`TensorKit.ElementarySpace`](https://jutho.github.io/TensorKit.jl/stable/man/spaces/). So the conversion is just `tensorkit_space(i::PhysicalIndex) = i.site.space`. No computation needed, just unwrapping. For abstract TN work (`IndexedTensor` vertices with no physical site), there is no `PhysicalIndex` — open legs are plain `TIx` values with explicit `ndim`, and the TensorKit space is `ComplexSpace(ndim(i))` directly.
- A virtual bond leg (`TIx{Upper/Lower}`) maps to `TensorKit.ComplexSpace(ndim(i))`. This is enough for non-symmetric computations; symmetry-carrying bonds require `ElementarySpace` objects (deferred).

These two conversions are all you need to construct a TensorKit `TensorMap` from an `IndexedTensor` in `:tensorkit` mode.

**What is deferred**

Two things are intentionally left for later:

- Sector information on `BondIndex`: to actually get block-sparse contractions you need to know which quantum numbers live at which bond. Adding this is a breaking change to `BondIndex` and only makes sense once there is a specific symmetry group and a concrete physical system to simulate with it.
- The full [`TensorOperations.jl`](https://github.com/QuantumKitHub/TensorOperations.jl) contraction interface for native-backed `IndexedTensor` (the `tensorcontract!`, `tensortrace!`, `tensoradd!` methods). This is a real implementation effort and deserves its own plan when the time comes.

**References**

- [JAX default device context](https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices): the direct precedent for the `with_backend(...) do` scoped pattern.
- [PennyLane device model](https://docs.pennylane.ai/en/stable/introduction/circuits.html): prior art for the "write algorithm once, choose backend separately" idea.
- [Julia `ScopedValues` stdlib](https://docs.julialang.org/en/v1/base/scopedvalues/): the mechanism used to thread the backend choice through the call stack without explicit parameter passing.
- [Julia `IOContext`](https://docs.julialang.org/en/v1/base/io-network/#Base.IOContext): Julia's own prior art for scoped ambient context; confirms this is an established Julia pattern.
- [Flux.jl `gpu()`/`cpu()` model transfer](https://fluxml.ai/Flux.jl/stable/guide/gpu/): the Julia ecosystem precedent for making a model parametric over its backing array type.
- [TensorKit.jl](https://jutho.github.io/TensorKit.jl/stable/man/tensors/): block-sparse `TensorMap`, the likely production backing type for the `:tensorkit` backend.
- [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl): goes all-in on TensorKit and never uses `@tensor` for inner products. Confirms there is no existing Julia ecosystem precedent for the hybrid `:native`/`:tensorkit` approach this design is trying to build.
- [MatrixAlgebraKit.jl](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl): sits between LinearAlgebra/LAPACK and TensorKit, providing `svd_compact`, `svd_trunc`, and the `TruncationScheme` vocabulary. TensorKit v0.15+ uses it internally, so both backends share the same SVD API.


## Matrix product states and operators

This section brings together the two layers defined above — the physical site types (`StateSite{D}`, `OperatorSite{D}`) and the mathematical objects (`IndexedTensor`, `BondLeg`, `tensor_svd`) — into the concrete data structures that MPS and MPO algorithms operate on.

### `AbstractMPS` base type

Both `FiniteMPS` and `FiniteMPO` share a common interface: querying length, reading and writing individual site tensors by index, and contracting with environments. This shared behaviour lives in a single abstract type:

```julia
abstract type AbstractMPS end
```

`FiniteMPS{D, T} <: AbstractMPS` and `FiniteMPO{D, T} <: AbstractMPS`, so methods like `Base.length`, `Base.getindex`, and `Base.setindex!` can be defined once on `AbstractMPS` and inherited by both. This mirrors both [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl) and [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl), which each define the same `AbstractMPS` base for exactly the same reason.

### `FiniteMPS{D, T}`: finite matrix product state

A finite MPS on a physical chain is a sequence of $L$ rank-3 tensors, one per site, connected by virtual bond legs. Each site tensor carries three legs: `vL` (left virtual bond), `σ` (physical index), and `vR` (right virtual bond). The two virtual legs are contracted and appear as edges in the `Lattice{StateSite{D}, BondLeg}` graph; the physical leg is open and its dimension is fixed by `hilbert_space(D)`.

```julia
mutable struct FiniteMPS{D <: AbstractDoF, T <: Number} <: AbstractMPS
    L::Int
    tensors::Vector{IndexedTensor{T, 3}}   # B[i]: legs (vL, σ, vR), 1-indexed
    bond_svs::Vector{Vector{real(T)}}      # S[i]: L+1 entries; S[1] = S[L+1] = [1.0]
    form::AbstractMPSForm                  # current gauge; see AbstractMPSForm hierarchy below
end
```

**Type parameter `D`**

The DoF type `D` is a compile-time type parameter, not a stored field. The Julia compiler specialises all `FiniteMPS{Spin{1//2}, ComplexF64}` code as a distinct compilation unit from `FiniteMPS{Fermionic, ComplexF64}`. Dispatch on DoF type is therefore resolved at compile time, with no runtime inspection of tags or stored instances.

Contrast with [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl), where the MPS struct has no type parameters at all:

```julia
# ITensorMPS.jl source — for comparison, not for use here
mutable struct MPS <: AbstractMPS
    data::Vector{ITensor}
    llim::Int
    rlim::Int
end
```

The physical degree of freedom in ITensorMPS is encoded in string tags on `Index` objects (`siteind("S=1/2")`), which are inspected at runtime. The flexibility is real but the compile-time information is lost.

**Canonical form: `AbstractMPSForm` type hierarchy**

Different algorithms impose different gauge structures on the MPS tensors, and those gauges have meaningfully different semantics. Encoding the current gauge as bare integers or as per-site float tuples works, but it forces algorithms to branch on values at runtime and makes precondition errors invisible until results are wrong. Qritical.jl instead represents gauge structure as a typed sum — each concrete subtype of `AbstractMPSForm` is dispatchable, so algorithms can state their preconditions in their signatures:

```julia
abstract type AbstractMPSForm end

struct CanonicalForm <: AbstractMPSForm  # A···A · center · B···B
    llim::Int
    rlim::Int
end

struct VidalForm <: AbstractMPSForm end  # Γ-Λ-Γ-Λ: tensors are Γ, Λ lives in bond_svs

struct ArbitraryForm <: AbstractMPSForm end  # no orthogonality guarantees
```

**`CanonicalForm(llim, rlim)`: the standard sweep form**

The pair `(llim, rlim)` inside `CanonicalForm` encodes orthogonality as an integer interval:

- `tensors[i]` is left-canonical ($A_i^\dagger A_i = \mathbb{1}$) for all `i ≤ llim`
- `tensors[i]` is right-canonical ($B_i B_i^\dagger = \mathbb{1}$) for all `i ≥ rlim`
- The orthogonality center occupies sites `llim+1` through `rlim-1`; a complete sweep reduces this to a single site

State lifecycle: a freshly allocated MPS starts as `ArbitraryForm()`. After a full left-to-right sweep: `CanonicalForm(L, L+1)`. After a full right-to-left sweep: `CanonicalForm(0, 1)`. During a DMRG sweep, moving the center one site leftward decrements `rlim`; rightward increments `llim`.

This is the pattern used directly by ITensorMPS.jl, where `llim` and `rlim` are bare `Int` fields on the struct. The `CanonicalForm` wrapper adds no storage overhead but makes the gauge dispatchable.

**`VidalForm()`: the TEBD-native gauge**

`CanonicalForm` can only describe configurations shaped like `A···A · center · B···B`. It cannot represent the Vidal **Γ-Λ-Γ form**:

```
Λ₀ — Γ₁ — Λ₁ — Γ₂ — Λ₂ — Γ₃ — Λ₃
```

where every site tensor is a Γ matrix (no singular values absorbed) and the Λ arrays live on the bonds in `bond_svs`. This is the natural form for TEBD: the two-site gate update preserves it by construction. Contracting $\Theta_{i,i+1} = \Lambda_{i-1} \Gamma_i \Lambda_i \Gamma_{i+1} \Lambda_{i+1}$, applying the gate, and SVD-splitting produces new $\tilde{\Gamma}_i, \tilde{\Lambda}_i, \tilde{\Gamma}_{i+1}$ — all still in Γ form. No re-canonicalization is needed between gates, so the cost per gate stays $O(\chi^3)$ rather than $O(L\chi^3)$.

`VidalForm()` is the tag that says: the authoritative Λ arrays are in `bond_svs`; tensors in `tensors[i]` are Γ matrices. The `bond_svs` field was designed with this use in mind — its presence in the struct makes both `CanonicalForm` and `VidalForm` first-class representations using the same data, distinguished only by the `form` tag.

**Why not per-site tuples (TeNPy approach)?**

TeNPy stores a `(λL, λR)` float pair per site representing what fraction of the neighboring Λ matrices has been absorbed from each side:

| Tuple | Name | Meaning |
|---|---|---|
| `(1, 0)` | A | left-canonical |
| `(0, 1)` | B | right-canonical |
| `(0, 0)` | Γ | Vidal — no SVs absorbed |
| `(0.5, 0.5)` | symmetric | half-Λ on each side |
| `(1, 1)` | Θ | full SVs on both sides |

Per-site tuples are strictly more expressive: they can represent configurations like `A A Γ Γ Γ B B` (after a TEBD sweep starting from a canonical state) or a single `Θ` tensor at the center bond during a 2-site update. `AbstractMPSForm` as defined above cannot represent all of these.

The trade-off is correctness guarantee versus bookkeeping cost. A `Vector{NTuple{2,Float64}}` of length $L$ must be updated consistently every time any tensor changes. A metadata error — forgetting to mark site $i$ as Γ after a gate — silently corrupts the form and produces wrong energies or overlaps with no crash. The typed sum covers the three gauges that arise in the course exercises (`CanonicalForm`, `VidalForm`, `ArbitraryForm`), with no per-site bookkeeping to maintain. If a future algorithm needs the symmetric Θ form or a partially-canonical intermediate state, a new concrete subtype of `AbstractMPSForm` can be added without touching existing subtypes or their dispatch.

MPSKit.jl takes a third path: it stores all four representations (AL, AR, AC, C) simultaneously with lazy `Union{Missing, A}` fields, so the question "which form am I in?" never arises — you request the form you want and the library recomputes it if stale. This is the most robust design at the cost of significant implementation complexity, and is deferred to the `InfiniteMPS` era.

**Why `bond_svs` is stored explicitly**

ITensorMPS.jl drops singular values from the MPS struct entirely — they appear only as return values from SVD. MPSKit.jl absorbs them into the `C` bond gauge matrices (so the singular value array and the C matrix are effectively the same object).

Qritical.jl stores them in `bond_svs` because two course algorithms need direct access to Λ:

- **TEBD (Exercise 3)** operates in the Vidal Γ-Λ-Γ form. Each site tensor splits as $B_i = \Gamma_i \Lambda_i$, and a two-site gate is applied as $B_i B_{i+1} \to \tilde{B}_i \tilde{B}_{i+1}$ via SVD on the merged tensor. Extracting Λ from an implicit C matrix requires an extra SVD; having it in `bond_svs` makes the update direct.
- **Truncation error tracking** (throughout): each `bond_svs[i]` stores the full retained singular value spectrum, so the truncation error at bond `i` is always $\varepsilon_i = \lVert \Sigma_{\text{discarded}} \rVert_2$, computable without re-running the SVD.

Boundary conditions: `bond_svs[1]` and `bond_svs[L+1]` are trivially `[1.0]` — they are the left and right boundary vectors of a finite open chain.

Note on element type: singular values are always real even when tensors are complex (they are the square roots of eigenvalues of the positive-semidefinite matrix $A^\dagger A$). Storing them as `Vector{real(T)}` — i.e. `Float64` when `T = ComplexF64` — halves their memory and removes the conceptual confusion of a complex singular value.

### `FiniteMPO{D, T}`: finite matrix product operator

An MPO site tensor has four legs: `vL` (left virtual), `σ` (ket physical leg), `σ*` (bra physical leg), and `vR` (right virtual). The two physical legs carry `hilbert_space(D)` in domain and codomain respectively, which is exactly what an operator $\hat{O} : \mathcal{H} \to \mathcal{H}$ requires — the same local Hilbert space appears on both sides.

```julia
mutable struct FiniteMPO{D <: AbstractDoF, T <: Number} <: AbstractMPS
    L::Int
    tensors::Vector{IndexedTensor{T, 4}}        # W[i]: legs (vL, σ, σ*, vR)
    IdL::Int                                     # virtual bond index: "identity applied left of i"
    IdR::Int                                     # virtual bond index: "identity applied right of i"
end
```

**No canonical form**

MPOs representing physical Hamiltonians have no canonical form that simplifies computations. There is no `llim`/`rlim` on `FiniteMPO`. The virtual bond dimension of a Hamiltonian MPO is determined by the number of independent terms in $H$ (for a Heisenberg chain $H = \sum_i \vec{S}_i \cdot \vec{S}_{i+1}$, the virtual bond dimension is 5), not by any truncation.

**`IdL` and `IdR`**

These are integer indices into the virtual bond space of the MPO that label the "only identity operators applied so far" state of the finite state machine underlying a Hamiltonian MPO. They are needed to correctly contract the MPO with an MPS at the left and right boundaries — the left environment is initialized as the row vector corresponding to `IdL`, and the right environment is initialized as the column vector corresponding to `IdR`. TeNPy and ITensorMPS.jl both carry these as explicit fields; leaving them out forces callers to guess the boundary index convention each time.

**MPS vs MPO struct distinction**

In ITensorMPS.jl, `MPS` and `MPO` are literally the same struct — they differ only in the number of physical legs on the `ITensor` objects stored inside. In MPSKit.jl the same pattern holds: there is one `MPO{O, V}` type and `FiniteMPO`/`InfiniteMPO` are type aliases differing only in whether `V` is `Vector` or `PeriodicVector`.

Qritical.jl uses separate types because `FiniteMPS` and `FiniteMPO` carry structurally different fields: `bond_svs` and `form::AbstractMPSForm` only make sense for states, and `(IdL, IdR)` only makes sense for operators. Merging them into one struct would require dummy fields on one side, which obscures the design intent.

### Connection to the lattice and site types

The relationship between these structs and the `Lattice{V, E}` graph defined earlier:

- `FiniteMPS{D, T}` corresponds to a 1D chain `Lattice{StateSite{D}, BondLeg}`. The lattice captures the topology; the MPS struct carries the tensor data.
- `FiniteMPO{D, T}` corresponds to a 1D chain `Lattice{OperatorSite{D}, BondLeg}`.
- For abstract tensor network work (Track 1), there is no MPS/MPO struct — tensors are vertex data on a `Lattice{IndexedTensor, BondLeg}` directly.

For a finite 1D chain the graph topology is trivially a path, so the `Lattice` object is implicit in the struct layout rather than stored as a field. The `L` field plus the `tensors` vector ordering encodes the chain completely. For non-1D or non-trivial topologies (e.g. PEPS on a 2D grid) a full `Lattice{V, E}` would be needed as an explicit field.

### Future extensions

**`InfiniteMPS{D, T}`**: For infinite, translationally invariant systems on a unit cell of length `L_uc`. The tensor storage becomes a `PeriodicVector` instead of `Vector` (wrapping indices modulo `L_uc`), and the canonical form management upgrades to the four-representation `(AL, AR, AC, C)` design from MPSKit.jl and the VUMPS literature. This extension does not affect `FiniteMPS` at all.

**`CompositeDoF{D1, D2}`**: For TEBD two-site gates (Exercise 3), two adjacent site tensors must be merged, SVD'd with a gate applied, and split back. The merged tensor has a single `d²`-dimensional physical leg representing the product space $\mathcal{H}_{D_1} \otimes \mathcal{H}_{D_2}$. The Julia equivalent of TeNPy's `GroupedSite`:

```julia
struct CompositeDoF{D1 <: AbstractDoF, D2 <: AbstractDoF} <: AbstractDoF end

hilbert_space(::CompositeDoF{D1, D2}) where {D1, D2} =
    hilbert_space(D1()) ⊗ hilbert_space(D2())
```

A `StateSite{CompositeDoF{Spin{1//2}, Spin{1//2}}}` then has a single four-dimensional physical leg — the merged index for a 2-site update step.

### Design decisions summary

| Aspect | ITensorMPS.jl | MPSKit.jl | Qritical.jl |
|---|---|---|---|
| Type parameters | None | `{A <: GenericMPSTensor, B <: MPSBondTensor}` | `{D <: AbstractDoF, T <: Number}` |
| MPS vs MPO | Same struct, different tensors inside | Same type alias pattern | Separate types — different fields |
| Canonical form | `(llim, rlim)` interval | AL/AR/AC/C four representations (lazy) | `AbstractMPSForm` hierarchy: `CanonicalForm(llim,rlim)`, `VidalForm()`, `ArbitraryForm()` |
| Singular values | Not stored in struct | Absorbed into C bond matrices | Stored explicitly in `bond_svs` |
| Physics encoding | Runtime string tags on `Index` | TensorKit `ElementarySpace` at construction | Type parameter `D <: AbstractDoF` |
| Mutability | `mutable struct` | Immutable `struct` (return new object) | `mutable struct` |
| Scope | Finite + infinite unified | Finite + infinite unified | Finite first; infinite deferred |

**References**

- [ITensorMPS.jl source: mps.jl](https://github.com/ITensor/ITensorMPS.jl/blob/main/src/mps.jl): the `(data, llim, rlim)` canonical form design and the `AbstractMPS` base type.
- [MPSKit.jl source: finitemps.jl](https://github.com/QuantumKitHub/MPSKit.jl/blob/master/src/states/finitemps.jl): the `(AL, AR, AC, C)` four-representation design with lazy `Union{Missing, A}` fields.
- [MPSKit.jl source: infinitemps.jl](https://github.com/QuantumKitHub/MPSKit.jl/blob/master/src/states/infinitemps.jl): `PeriodicVector` and the iMPS extension of the same four-representation design.
- [MPSKit.jl source: mpo.jl](https://github.com/QuantumKitHub/MPSKit.jl/blob/master/src/operators/mpo.jl): the minimal `MPO{O, V}` wrapper and the `FiniteMPO`/`InfiniteMPO` type alias pattern.
- [TeNPy MPS documentation](https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.html): per-site `(λL, λR)` form tuples and the `_S` singular value array — the direct precedent for `bond_svs`.
