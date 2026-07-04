# Indices and QTensor

When you draw a tensor network on paper, the correspondence between the diagram and the maths is obvious — each node is a tensor, each line is a leg with an index, and arrow directions encode covariant vs contravariant position. Opening a Julia file and staring at `rand(2, 4, 4)` breaks that correspondence: the array has three slots but no idea that slot 1 is a physical spin leg and slots 2–3 are left and right bond legs pointing in specific directions.

`QTensor` closes that gap. It pairs a backing array with a named, directed index for each leg, so the diagram and the code say the same thing.

## Index types

Every leg is described by a concrete subtype of [`AbstractIx`](@ref). Two interface methods must be implemented by every subtype:

| Method | Returns | Meaning |
|:-------|:--------|:--------|
| `ndim(i)` | `Int` | Number of distinct values the index can take |
| `label(i)` | `Symbol` | Name of the leg (e.g. `:σ`, `:αL`) |

### `TIx` — a single named index

[`TIx{L}`](@ref) is the workhorse index type. The type parameter `L` is either `Upper` or `Lower`, encoding whether the index sits in superscript (contravariant) or subscript (covariant) position. This is a **type-level** distinction: `TIx{Upper}` and `TIx{Lower}` are different types, so matching them up for contraction can be resolved by the compiler.

Use the constructor helpers `upper` and `lower` instead of the parametric form directly:

```jldoctest tix
julia> using Qritical

julia> σ = upper(:σ, 2)       # physical spin-1/2: 2 values
TIx{Upper}(:σ, 2)

julia> αL = lower(:αL, 4)     # left bond: dimension 4
TIx{Lower}(:αL, 4)

julia> ndim(σ)
2

julia> label(αL)
:αL
```

Upper and lower indices with the same label are distinct — index position is semantically meaningful:

```jldoctest tix
julia> upper(:α, 4) == lower(:α, 4)
false

julia> upper(:α, 4) == upper(:α, 4)
true
```

The inner constructor rejects non-positive dimensions immediately:

```jldoctest
julia> using Qritical

julia> TIx{Upper}(:σ, 0)
ERROR: ArgumentError: TIx ndim must be positive, got 0
[...]
```

To build several indices of the same position at once, use [`uppers`](@ref) or [`lowers`](@ref):

```jldoctest
julia> using Qritical

julia> vL, vR = uppers(:vL => 1, :vR => 4)
(TIx{Upper}(:vL, 1), TIx{Upper}(:vR, 4))

julia> ndim(vL)
1
```

A dimension-1 virtual index appears at the open boundaries of a finite MPS, where the bond space is trivial.

### `MulTIx` — a composite index

[`MulTIx`](@ref) bundles several constituent indices into one combined index whose dimension is the product of its parts. This is what you need when reshaping a high-order tensor into a matrix for SVD: the "row" group and "column" group each become one `MulTIx`.

```jldoctest multiix
julia> using Qritical

julia> α, β = upper(:α, 2), lower(:β, 3)
(TIx{Upper}(:α, 2), TIx{Lower}(:β, 3))

julia> g = MulTIx((α, β))
MulTIx((TIx{Upper}(:α, 2), TIx{Lower}(:β, 3)))

julia> dim(g)   # product: 2 × 3
6

julia> label(g)
:αβ
```

## `QTensor`

`QTensor` (see the [QTensor API documentation](../api/qtensor.md)) pairs a backing `AbstractArray` with an `NTuple` of `AbstractIx` values, one per dimension:

```jldoctest it
julia> using Qritical

julia> vL, σ, vR = upper(:vL, 1), lower(:σ, 2), upper(:vR, 4);

julia> A = QTensor(rand(1, 2, 4), (vL, σ, vR));

julia> size(A)
(1, 2, 4)

julia> A.indices
(TIx{Upper}(:vL, 1), TIx{Lower}(:σ, 2), TIx{Upper}(:vR, 4))
```

`QTensor` subtypes `AbstractArray`, so all standard array operations work — indexing, slicing, broadcasting, `size`, `ndims`:

```jldoctest it
julia> ndims(A)
3

julia> size(A, 2)
2

julia> length(A.indices) == ndims(A)
true
```

### Scalar tensors

An order-0 `QTensor` is a scalar: empty index tuple, 0-dimensional array:

```jldoctest
julia> using Qritical

julia> s = QTensor(fill(3.14), ());

julia> ndims(s)
0

julia> s[]
3.14
```

## Partition and Bipartition

A [`Partition`](@ref) is an ordered list of indices representing one set of tensor legs. A [`Bipartition`](@ref) is an ordered pair of non-overlapping `Partition`s — the left one maps to matrix rows, the right to columns.

```jldoctest bp
julia> using Qritical

julia> vL, σ, vR = upper(:vL, 2), lower(:σ, 2), upper(:vR, 3);

julia> A = QTensor(rand(2, 2, 3), (vL, σ, vR));

julia> left  = Partition(vL, σ)
Partition(AbstractIx[TIx{Upper}(:vL, 2), TIx{Lower}(:σ, 2)])

julia> right = Partition(vR)
Partition(AbstractIx[TIx{Upper}(:vR, 3)])

julia> bp = Bipartition(left, right)
Bipartition(Partition(AbstractIx[TIx{Upper}(:vL, 2), TIx{Lower}(:σ, 2)]), Partition(AbstractIx[TIx{Upper}(:vR, 3)]))
```

The constructor checks for overlap at construction time — two partitions sharing an index would make the bipartition ambiguous:

```jldoctest bp
julia> Bipartition(Partition(vL, σ), Partition(σ, vR))
ERROR: ArgumentError: Bipartition partitions overlap on: AbstractIx[TIx{Lower}(:σ, 2)]
[...]
```

### `bipartition` convenience constructor

When one side of the split is implicit (all legs not in the left partition), use [`bipartition`](@ref) with [`complement`](@ref):

```jldoctest bp
julia> bp2 = bipartition(Partition(vL, σ), A)   # right = complement = {vR}
Bipartition(Partition(AbstractIx[TIx{Upper}(:vL, 2), TIx{Lower}(:σ, 2)]), Partition(AbstractIx[TIx{Upper}(:vR, 3)]))

julia> bp2 == bp
true
```

## `group_legs`

[`group_legs`](@ref) permutes and reshapes a `QTensor` into a matrix according to a `Bipartition`. Each axis of the output is tagged with an auto-labeled `MulTIx`:

```jldoctest bp
julia> M = group_legs(A, bp);

julia> size(M)
(4, 3)

julia> ndims(M)
2

julia> label(M.indices[1])   # auto-label from vL and σ
:vLσ
```

`group_legs` validates full index coverage — every leg in `A` must appear in exactly one side of the bipartition:

```jldoctest bp
julia> group_legs(A, Bipartition(Partition(vL), Partition(σ)))
ERROR: ArgumentError: bipartition covers 2/3 tensor indices — uncovered: AbstractIx[TIx{Upper}(:vR, 3)]
[...]
```
