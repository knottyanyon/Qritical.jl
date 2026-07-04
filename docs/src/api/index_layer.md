# Index Layer

## Physics motivation

Every tensor in quantum many-body physics carries more information than just its numerical entries: each leg has a *direction* and a *physical meaning*. In a matrix product state, the left virtual leg flows *into* a site tensor while the right virtual leg flows *out*; the physical leg ``\sigma`` labels the local Hilbert-space basis and flows out toward the ket. These distinctions are not cosmetic — they are what makes contraction rules unambiguous, what allows block-sparse symmetry sectors to be tracked automatically, and what prevents silently mis-wiring a bra where a ket is expected.

The index layer makes this precise. Each leg is a `TIx{L}` where `L ∈ {Upper, Lower}` records the *variance* — whether the leg is contravariant (inward, domain) or covariant (outward, codomain) in the sense of von Delft's arrow convention. This is not a label one attaches by convention: it follows directly from the geometric orientation of the leg in the tensor-network diagram.

The settled bond convention throughout Qritical is: left virtual leg inward → `TIx{Upper}`; right virtual leg outward → `TIx{Lower}`; physical legs ``\sigma`` outward → `TIx{Lower}`. An Einstein contraction then pairs exactly one `Upper` with one `Lower` of the same label, conserving quantum numbers automatically once symmetry sectors are attached to the space.

A **partition** (`Bipartition`) is orthogonal to variance: it is a per-operation choice of which legs are grouped as the matrix rows vs. columns for an SVD. For a *state*, all physical legs share `Lower` variance, so the bipartition is a free Schmidt cut. For an *operator*, the bra (`Upper`) and ket (`Lower`) physical legs have opposite variance, which pins the partition. `group_legs` mechanises this distinction so that neither state nor operator decompositions ever need to specify the row/column split by hand.

The `MulTIx` type represents a fused (grouped) leg — the result of `group_legs` collapsing several legs into one. Its dimension is the product of its children's dimensions, and it carries both faces of the bond so that re-gauging (moving the orthogonality centre) is a swap, not a rebuild.

---

## Types

### Abstract types and variance tags

```@docs
AbstractIx
IxLoc
Upper
Lower
```

### Single index

```@docs
TIx
```

### Multi-index (fused leg)

```@docs
MulTIx
```

### Partitions

```@docs
Partition
Bipartition
```

## Functions

### Accessors

```@docs
dim
label
which_space
```

### Index manipulation

```@docs
flip
upper
lower
```

### Batch constructors

```@docs
uppers
lowers
uppers_range
lowers_range
```

### Partition utilities

```@docs
complement
bipartition
bond_label
```
