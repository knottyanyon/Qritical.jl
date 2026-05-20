# IndexedTensor

When I first drew a tensor network diagram on paper for the course exercises, the correspondence between the diagram and the code felt obvious — each node is a tensor, each line is an index, arrows show direction. Then I opened a Julia file and stared at `rand(2, 4, 4)`. The array has three slots, sure, but it has no idea that slot 1 is the physical spin degree of freedom and slots 2 and 3 are bond legs with arrows pointing in specific directions. Naming that `A` doesn't help: `A[2, 3, 1]` is just a number.

`IndexedTensor` is what closes that gap. It pairs the backing array with a named, directed index for each leg — one index per dimension, in order — so the diagram and the code say the same thing.

## Defining indices

Every leg is described by an `AbstractIndex`. There are two kinds:

| Type | What it represents |
|:-----|:-------------------|
| `PhysicalIndex` | local Hilbert space at a lattice site; dimension comes from the site type |
| `BondIndex` | virtual/entanglement leg between two sites; dimension is set explicitly |

Each index carries a **direction** (`UpIndex` or `DownIndex`) that matches the superscript/subscript notation from the [Notation](../notation.md) page:

| Direction    | Notation position | Arrow in diagram |
|:-------------|:------------------|:-----------------|
| `UpIndex`    | superscript       | incoming → •     |
| `DownIndex`  | subscript         | outgoing • →     |

Contractions always pair one `UpIndex` leg with one `DownIndex` leg — same as the Einstein summation convention where you sum over a repeated up-down pair.

### Physical indices

```jldoctest indexed_tensor
julia> using HalfIntegers: half

julia> σ = PhysicalIndex(:σ, SpinSite(half(1), 1), UpIndex);

julia> local_hilbert_dim(σ)   # spin-1/2: 2S+1 = 2
2
```

### Bond indices

A bond index lives on a directed bond from site `from` to site `to`. The convention (see [ADR 0002](../adr/0002-bond-index-arrow-orientation.md)) is:

- the `from` site holds the leg with `DownIndex` (arrow leaves, outgoing)
- the `to` site holds the leg with `UpIndex` (arrow arrives, incoming)

```jldoctest indexed_tensor
julia> α_out = BondIndex(:α, 1, 2, 4, DownIndex);   # bond α leaves site 1

julia> α_in  = BondIndex(:α, 1, 2, 4, UpIndex);     # same bond, enters site 2

julia> local_hilbert_dim(α_out)   # bond dimension
4
```

### Dual indices

`dual(i)` returns a copy of `i` with the direction flipped. The postfix `'` is also available. Two indices form a valid contraction pair when they are duals of each other:

```jldoctest indexed_tensor
julia> isdual(α_out, α_in)        # opposite directions, same label and bond
true

julia> isdual(α_out, dual(α_out))
true

julia> isdual(α_out, α_out)       # same direction — not a valid pair
false
```

## Building a tensor

Pass the backing `Array` and a tuple of indices to `IndexedTensor`. The tuple must have exactly as many entries as the array has dimensions, in the same order:

```jldoctest indexed_tensor
julia> β = BondIndex(:β, 2, 3, 3, DownIndex);

julia> A = IndexedTensor(ones(2, 4), (σ, α_out))
2×4 IndexedTensor{Float64, 2, Matrix{Float64}}:
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0

julia> size(A)
(2, 4)

julia> ndims(A)
2
```

This represents the site tensor $A^{\sigma}_{\alpha}$ at site 1 — a physical leg $\sigma$ up (superscript) and bond leg $\alpha$ down (subscript).

## Element access

`IndexedTensor` behaves as a plain array for indexing and iteration. Everything from `Base.AbstractArray` works out of the box:

```jldoctest indexed_tensor
julia> A[1, 2]
1.0

julia> A[2, end]
1.0
```

## Contracting tensors

The named indices pay off when contracting. The `@tensor` macro from `TensorOperations.jl` uses the index labels in the expression to decide which legs to pair, and the `checkcontractible` rules enforce the pairing: the two legs must have opposite directions, the same label, and the same dimension.

```jldoctest indexed_tensor
julia> using TensorOperations

julia> B = IndexedTensor(ones(4, 3), (α_in, β));

julia> @tensor C[s, b] := A[s, a] * B[a, b];

julia> size(C)
(2, 3)
```

Here `a` contracts $\alpha_{\text{out}}$ (from `A`) with $\alpha_{\text{in}}$ (from `B`) — a valid up-down pair. The result is a `2×3` tensor $C^{\sigma}_{\beta}$.

If the directions don't form a valid pair, the error fires immediately at contraction time:

```julia
julia> B_wrong = IndexedTensor(ones(4, 3), (α_out, β));   # α_out on both sides

julia> @tensor D[s, b] := A[s, a] * B_wrong[a, b]
ERROR: ArgumentError: Both legs on index a are DownIndex — contraction requires a dual pair
```

You get the error before any computation happens. This is the main advantage of carrying directions on the index objects rather than relying on convention.

## Kronecker delta

`kronecker_delta(i, j)` returns the identity tensor for a dual index pair. It is useful for inserting identities and for index relabelling:

```jldoctest indexed_tensor
julia> δ = kronecker_delta(α_out, α_in)
4×4 IndexedTensor{Float64, 2, Matrix{Float64}}:
 1.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  1.0  0.0
 0.0  0.0  0.0  1.0

julia> δ.indices
(BondIndex(:α, 1, 2, 4, DownIndex), BondIndex(:α, 1, 2, 4, UpIndex))
```

Passing two indices that are not a dual pair throws:

```jldoctest indexed_tensor
julia> kronecker_delta(α_out, α_out)
ERROR: ArgumentError: kronecker_delta requires a dual index pair — got DownIndex and DownIndex
[...]
```
