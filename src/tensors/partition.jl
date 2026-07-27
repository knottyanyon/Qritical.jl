# ========================= Partition & Bipartition ===========================

"""
    Partition

An ordered group of tensor legs, represented as `Vector{AbstractIx}`.

A `Partition` names the subset of a tensor's legs that will be collected
along one axis (rows or columns) when the tensor is matricised for an SVD or
contraction. The order within the partition determines the reshape order of
the underlying array.

Use [`Bipartition`](@ref) to pair two complementary `Partition`s, and
[`group_legs`](@ref) to apply the split to a `QTensor`.

# Examples

```jldoctest
julia> vL = upper(:vL, 2);
       σ = upper(:σ, 3);

julia> p = Partition([vL, σ]);

julia> length(p)
2

julia> p[1] == vL
true
```
"""
const Partition = Vector{AbstractIx}   # `const` = assign once; `Vector{AbstractIx}` = mutable resizable array of AbstractIx elements. this is a TYPE ALIAS, not a newtype — Partition IS Vector{AbstractIx}, no wrapping

"""
    Bipartition

A split of a tensor's legs into two ordered groups:

  - **`left`** — the legs that become the row axis (first index) after
    reshaping
  - **`right`** — the legs that become the column axis (second index) after
    reshaping

The constructor verifies that the two groups are **disjoint**: no single
`AbstractIx` value may appear in both `left` and `right`. Coverage — that the
two groups together account for every leg of the target tensor — is checked by
[`group_legs`](@ref) at the point of use, not here, because a `Bipartition`
may be built before a tensor is chosen.

# Fields

  - `left  :: Partition`
  - `right :: Partition`

# Examples

```jldoctest
julia> vL = upper(:vL, 2);
       σ = upper(:σ, 3);
       vR = lower(:vR, 4);

julia> bp = Bipartition(Partition([vL, σ]), Partition([vR]));

julia> length(bp.left)
2

julia> bp.right[1] == vR
true
```
"""
struct Bipartition   # immutable struct: left and right Partitions cannot be swapped after construction
    left::Partition    # the legs that will form the row axis in the SVD reshape; `Partition` = `Vector{AbstractIx}`
    right::Partition   # the legs that will form the column axis

    function Bipartition(left::Partition, right::Partition)   # inner constructor: validates disjointness before calling `new`
        for ix in left   # iterate over each leg in the left partition 
            ix ∈ right && throw(   # `∈` = Unicode `in` operator. `&&` short-circuits: only throw if the condition is true
                ArgumentError(
                    "Bipartition: leg '$(label(ix))' (dim=$(dim(ix))) appears in both " *   # `*` = string concatenation. `$(...)` = string interpolation 
                    "the left and right partitions — each leg must belong to exactly one side.",
                ),
            )
        end
        new(left, right)   # allocate and initialize the struct fields (only callable inside an inner constructor)
    end
end

"""
    complement(p::Partition, indices) -> Partition

Return the legs in `indices` that are **not** present in partition `p`,
preserving their original order.

`indices` may be any iterable of `AbstractIx` values — typically a tuple of
legs taken from a `QTensor` or a plain `Vector{AbstractIx}`.

Matching is by index equality (`==`), so two legs with the same label and
dimension but different variance (e.g. `upper(:σ, 2)` vs `lower(:σ, 2)`) are
treated as distinct.

See also: [`bipartition`](@ref), [`Bipartition`](@ref), [`complement(::Partition, ::QTensor)`](@ref)

# Examples

```jldoctest
julia> vL = upper(:vL, 2);
       σ = upper(:σ, 3);
       vR = lower(:vR, 4);

julia> complement(Partition([vL, σ]), [vL, σ, vR])
1-element Vector{AbstractIx}:
 TIx{Lower}(:vR, 4)

julia> complement(Partition([]), [vL, σ])
2-element Vector{AbstractIx}:
 TIx{Upper}(:vL, 2)
 TIx{Upper}(:σ, 3)
```
"""
complement(p::Partition, indices) = Partition([ix for ix in indices if ix ∉ p])   # comprehension: collect all `ix` from `indices` that are NOT in partition `p`; `∉` = Unicode `not in`. preserves original order from `indices`

"""
    bipartition(left::Partition, indices) -> Bipartition

Construct a [`Bipartition`](@ref) whose right side is automatically
`complement(left, indices)`.

This is the most convenient way to describe a Schmidt cut: name the legs you
want on the left (row) side and let the library fill in the right (column)
side.

# Examples

```jldoctest
julia> vL = upper(:vL, 2);
       σ = upper(:σ, 3);
       vR = lower(:vR, 4);

julia> bp = bipartition(Partition([vL, σ]), [vL, σ, vR]);

julia> bp.right[1] == vR
true
```
"""
bipartition(left::Partition, indices) = Bipartition(left, complement(left, indices))   # convenience: right side = everything not in left; calls complement to compute the right Partition, then constructs a Bipartition
