using LinearAlgebra
using TensorOperations

# Naming deviates from Julia convention ({T,N}) intentionally:
#   Element — the scalar field (e.g. ComplexF64), not constrained to <:Number.
#   Order   — number of legs; the term "rank" is reserved for the SVD
"""
    QTensor{Element, Order, D<:AbstractArray{Element,Order}}

An `Order`-index tensor with a named, covariant/contravariant index attached to each leg.
Subtyping `AbstractArray` makes it transparent to `@tensor` contractions.

The backing store `D` is a free type parameter: `:native` mode uses plain
`Array{Element,Order}` from Julia Base; future backends may substitute other `AbstractArray`
subtypes without touching algorithm code. The goal is to facilitate the use of efficient
sparse array descriptions if one wishes to implement symmetry details to take advantage
of the block structure of the Hamiltonians that we usually deal with.

# Constructor
```julia
QTensor(data, indices)
```

`data` must be an `AbstractArray` encoding the numerical data of the tensor and `indices`
an `NTuple` of `AbstractIx` values. The shared `Order` parameter in both types enforces
that the number of indices equals the array rank at the type level.

# Fields
- `data::D` — the backing data array
- `indices::NTuple{Order,AbstractIx}` — named, variance-tagged legs

# Examples
```jldoctest
julia> data = [1.0 2.0; 3.0 4.0]
2×2 Matrix{Float64}:
 1.0  2.0
 3.0  4.0

julia> i = upper(:i, 2); j = lower(:j, 2);

julia> t = QTensor(data, (i, j))
QTensor{Float64, 2, ...}

julia> t[1, 1]
1.0
```
"""
struct QTensor{Element,Order,D<:AbstractArray{Element,Order}} <:
       AbstractArray{Element,Order}
    data::D
    indices::NTuple{Order,AbstractIx}

    function QTensor(
        data::D, indices::NTuple{Order,AbstractIx}
    ) where {Element,Order,D<:AbstractArray{Element,Order}}
        for k in 1:Order
            size(data, k) == dim(indices[k]) || throw(
                ArgumentError(
                    "leg $k: array size $(size(data,k)) ≠ index dim $(dim(indices[k]))"
                ),
            )
        end
        new{Element,Order,D}(data, indices)
    end
end

# ── AbstractArray interface ───────────────────────────────────────────────────
# All data operations delegate to `t.data`; `QTensor` adds no storage
# overhead and is fully transparent to Julia's array machinery, broadcasting,
# and `@tensor` contractions.

"""
    Base.size(t::QTensor) -> Tuple{Int,...}
    Base.size(t::QTensor, dim::Integer) -> Int

Return the dimensions of `t` as a tuple of `Int`s, or — when `dim` is given — the
length along that single dimension.

Delegates to `size(t.data)`.  The one-argument form returns `(d₁, d₂, …, dₙ)` where
`n == Order` and each `dₖ == dim(t.indices[k])`.  The two-argument form is equivalent
to `size(t)[dim]` but does not materialise the full tuple.

`Base.size` is the primary query in Julia's `AbstractArray` interface: `ndims`, `length`,
`axes`, and `CartesianIndices` all derive from it.

# Examples
```jldoctest
julia> t = QTensor([1.0 2.0; 3.0 4.0], (upper(:i,2), lower(:j,2)));

julia> size(t)
(2, 2)

julia> size(t, 1)
2
```
"""
Base.size(t::QTensor) = size(t.data)

"""
    Base.getindex(t::QTensor, i...) -> Element or Array

Return the element (or sub-array) of `t` at index `i...`, delegating to
`getindex(t.data, i...)`.

`i...` follows all standard Julia indexing rules:
- **Scalar indices** (one integer per dimension) return a single element of type `Element`.
- **Range / colon indices** (e.g. `t[1:2, :]`) return a newly-allocated plain `Array`
  (not a `QTensor` — the selected sub-array carries no index metadata).
- **Boolean arrays / `BitArray`** filter elements where the mask is `true`.
- **`CartesianIndex`** can be used in place of a tuple of integers.

To index multiple elements without allocating a copy, use `view(t.data, i...)` directly.

!!! note
    Indexing with ranges returns a raw `Array`, not a `QTensor`.  Use explicit leg
    slicing helpers (TBD) when you need to preserve index metadata.

# Examples
```jldoctest
julia> t = QTensor([1.0 2.0; 3.0 4.0], (upper(:i,2), lower(:j,2)));

julia> t[1, 2]
2.0

julia> t[1, :]
2-element Vector{Float64}:
 1.0
 2.0
```
"""
Base.getindex(t::QTensor, i...) = getindex(t.data, i...)

"""
    Base.setindex!(t::QTensor, v, i...) -> t

Store value `v` at index `i...` in the backing array and return `t`, delegating to
`setindex!(t.data, v, i...)`.

All standard Julia indexing forms are accepted:
- A single integer per dimension writes one element.
- Range / colon indices broadcast `v` (or copy from `v` when `v` is an array) into the
  selected slice, following the same rules as `Base.setindex!` on a plain `Array`.

Mutates the backing store in-place; the `QTensor` wrapper and its index metadata are
unchanged.

# Examples
```jldoctest
julia> t = QTensor(zeros(2,2), (upper(:i,2), lower(:j,2)));

julia> t[1, 1] = 3.14; t[1, 1]
3.14
```
"""
Base.setindex!(t::QTensor, v, i...) = setindex!(t.data, v, i...)

"""
    Base.IndexStyle(::Type{<:QTensor{E,O,D}}) -> IndexStyle(D)

Report the indexing style of `QTensor` by delegating to the style of the backing store `D`.

Julia's `AbstractArray` interface distinguishes two styles:
- **`IndexLinear()`** — the array can be addressed with a single integer `i ∈ 1:length(A)`.
  Julia dispatches generic iteration and many higher-order functions through the
  linear index when this is declared, typically giving better performance.
- **`IndexCartesian()`** — the natural index is a `CartesianIndex` (one integer per
  dimension).  Required for arrays whose internal layout is not a flat memory buffer
  (e.g. sparse arrays, `TensorMap`s).

For the default `:native` backend (`D = Array`) this returns `IndexLinear()`, meaning
`QTensor` elements are stored contiguously and can be iterated linearly.  If a future
backend substitutes an `IndexCartesian` store, this delegation ensures the correct
style is advertised automatically.

!!! note
    Declaring `IndexLinear` also satisfies the requirements for `strides` and
    `unsafe_convert`, enabling direct BLAS dispatch on the backing buffer.
"""
Base.IndexStyle(::Type{QTensor{E,O,D}}) where {E,O,D} = IndexStyle(D)

"""
    Base.strides(t::QTensor) -> Tuple{Int,...}

Return the memory strides of the backing array, delegating to `strides(t.data)`.

The *stride* of dimension `k` is the number of elements (not bytes) in the flat backing
buffer separating two adjacent entries along that dimension.  For a column-major
`Array{T,N}` (Julia's default layout), `strides(A) == (1, size(A,1), size(A,1)*size(A,2), …)`.

Together with `Base.unsafe_convert`, strides enable BLAS/LAPACK routines and external C
libraries to operate directly on `QTensor` data via a raw pointer + stride descriptor,
with no intermediate copy.  Julia's `LinearAlgebra` (including `mul!` / `BLAS.gemm!`)
queries `strides` internally whenever it dispatches to a native BLAS call.

`strides` is only defined for `IndexLinear` arrays with a strided memory layout.  For
non-strided backends (e.g. sparse or TensorKit), this method must not be called.

# Examples
```jldoctest
julia> t = QTensor(ones(3, 4), (upper(:i,3), lower(:j,4)));

julia> strides(t)
(1, 3)
```
"""
Base.strides(t::QTensor) = strides(t.data)

"""
    Base.unsafe_convert(::Type{Ptr{Element}}, t::QTensor{Element}) -> Ptr{Element}

Return a raw C pointer (`Ptr{Element}`) to the first element of the backing array,
delegating to `unsafe_convert(Ptr{Element}, t.data)`.

`Base.unsafe_convert` is part of Julia's C-interop protocol: it is called by `ccall`
and by BLAS/LAPACK wrappers in `LinearAlgebra` (e.g. `BLAS.gemm!`, `LAPACK.gesvd!`)
whenever they need a raw buffer address.  Together with `strides`, it makes `QTensor`
transparent to any routine that accepts a strided pointer + descriptor — no copy is made.

**The `unsafe` contract (caller's responsibility):**
1. The array must remain reachable (not garbage-collected) for the duration of the call.
   Pin it with `GC.@preserve t … end` when passing to external C code.
2. Accesses must stay within the declared bounds (`size(t)`, `strides(t)`).
3. The backing store must be mutable and contiguous (valid for the default `Array` backend;
   do not call on sparse or non-strided backends).

Violating any of these conditions is undefined behaviour — hence the `unsafe` prefix.
"""
function Base.unsafe_convert(::Type{Ptr{Element}}, t::QTensor{Element}) where {Element}
    return Base.unsafe_convert(Ptr{Element}, t.data)
end

# ── Partition / Bipartition convenience overloads ────────────────────────────
# These accept a QTensor as the second argument so callers don't have to
# extract A.indices manually.

"""
    complement(p::Partition, A::QTensor) -> Partition

Return the legs of `A` that are not in partition `p`, in the order they appear
in `A.indices`.  Delegates to `complement(p, A.indices)`.

# Examples
```jldoctest
julia> vL = upper(:vL, 2);  σ = lower(:σ, 3);  vR = lower(:vR, 4);

julia> A = QTensor(rand(2, 3, 4), (vL, σ, vR));

julia> complement(Partition([vL, σ]), A)
1-element Vector{AbstractIx}:
 TIx{Lower}(:vR, 4)
```
"""
complement(p::Partition, A::QTensor) = complement(p, A.indices)

"""
    bipartition(left::Partition, A::QTensor) -> Bipartition

Construct a [`Bipartition`](@ref) for tensor `A` whose right side is
`complement(left, A)`.

# Examples
```jldoctest
julia> vL = upper(:vL, 2);  σ = lower(:σ, 3);  vR = lower(:vR, 4);

julia> A = QTensor(rand(2, 3, 4), (vL, σ, vR));

julia> bp = bipartition(Partition([vL, σ]), A);

julia> bp.right[1] == vR
true
```
"""
bipartition(left::Partition, A::QTensor) = bipartition(left, A.indices)

# ── group_legs ────────────────────────────────────────────────────────────────

"""
    group_legs(A::QTensor, bp::Bipartition) -> QTensor

Permute and reshape `A` into a rank-2 `QTensor` by grouping its legs according
to the bipartition `bp`.

The legs listed in `bp.left` are collected into the **first** (row) axis and
wrapped in a [`MulTIx`](@ref); the legs in `bp.right` become the **second**
(column) axis, also wrapped in a `MulTIx`.  Legs within each group may appear
in any order relative to `A` — the array axes are permuted to match the order
specified in the partition before reshaping.

# Errors
- `ArgumentError` if a leg named in the bipartition is not found in `A`.
- `ArgumentError` if any leg of `A` is not covered by `bp.left ∪ bp.right`.

# Examples
```jldoctest
julia> σ  = lower(:σ,  2);  vL = upper(:vL, 3);  vR = lower(:vR, 4);

julia> A  = QTensor(rand(2, 3, 4), (σ, vL, vR));

julia> bp = Bipartition(Partition([σ, vL]), Partition([vR]));

julia> M  = group_legs(A, bp);

julia> size(M)
(6, 4)

julia> M.indices[1] isa MulTIx
true
```
"""
function group_legs(A::QTensor, bp::Bipartition)
    all_ix = collect(A.indices)

    # locate each partition leg in A.indices (by equality)
    _find(ix) = findfirst(==(ix), all_ix)
    left_pos  = _find.(bp.left)
    right_pos = _find.(bp.right)

    any(isnothing, left_pos) && throw(
        ArgumentError(
            "group_legs: one or more left-partition legs were not found in the tensor."
        ),
    )
    any(isnothing, right_pos) && throw(
        ArgumentError(
            "group_legs: one or more right-partition legs were not found in the tensor."
        ),
    )

    covered = Set{Int}([left_pos..., right_pos...])
    length(covered) < length(all_ix) && throw(
        ArgumentError(
            "group_legs: tensor has $(length(all_ix)) leg(s) but the bipartition " *
            "covers only $(length(covered)) — every leg must appear on exactly one side."
        ),
    )

    # permute axes so left legs come first (in bp.left order), then right legs
    perm = [left_pos..., right_pos...]
    data_p = isempty(perm) ? A.data : permutedims(A.data, perm)

    # reshape into a 2D matrix
    left_dim  = prod(dim, bp.left;  init=1)
    right_dim = prod(dim, bp.right; init=1)
    data_2d   = reshape(data_p, left_dim, right_dim)

    # attach fused index metadata
    left_ix  = MulTIx(Tuple(bp.left))
    right_ix = MulTIx(Tuple(bp.right))

    QTensor(data_2d, (left_ix, right_ix))
end

# ── TensorOperations.jl interface ────────────────────────────────────────────

"""
    TensorOperations.tensorstructure(t::QTensor, i::Int, conjA::Bool) -> AbstractIx

Return the `AbstractIx` metadata for leg `i` of tensor `t`, as required by the
`TensorOperations.jl` contraction interface.

`TensorOperations.jl` calls this function during allocation and compatibility checking
when building a `@tensor` contraction. For a plain `AbstractArray` the default returns
`size(A, i)` — just an integer. Overriding it to return the full `AbstractIx` exposes
the label, direction (`Upper`/`Lower`), and dimension to the contraction engine, enabling
future leg-compatibility checks beyond a bare dimension match.

The `conjA` flag (`true` when the tensor appears conjugated in a `@tensor` expression)
is forwarded by the interface but does not affect index identity — the same `AbstractIx`
is returned regardless.
"""
TensorOperations.tensorstructure(t::QTensor, i::Int, ::Bool) = t.indices[i]

# ==== State utilities =========================================================

"""
    bipartition_matrix(A::QTensor, bp::Bipartition) -> QTensor

Reshape the tensor `A` into a matrix (order-2 `QTensor`) by grouping its legs
according to the bipartition `bp`.

This is a physics-vocabulary alias for [`group_legs`](@ref): left-partition legs
become the row index (a `MulTIx`), right-partition legs become the column index.
The returned tensor satisfies:

```math
\\text{size}(M, 1) = \\prod_{\\ell \\in \\text{left}} \\dim(\\ell), \\qquad
\\text{size}(M, 2) = \\prod_{\\ell \\in \\text{right}} \\dim(\\ell)
```

# See also
[`group_legs`](@ref), [`do_svd`](@ref)
"""
bipartition_matrix(A::QTensor, bp::Bipartition) = group_legs(A, bp)

"""
    as_state(v::AbstractVector, dof_dims::AbstractVector{Int}) -> QTensor

Reshape the coefficient vector `v` into an order-`L` tensor with one physical
leg per site.  `dof_dims[i]` is the local Hilbert-space dimension at site `i`.

The result satisfies `vec(ψ.data) === v` (no copy; the tensor shares memory with
the input vector).  Each leg is labelled `:σ1`, `:σ2`, … and typed `Lower`
(outgoing physical index, codomain convention).

# Arguments
- `v         :: AbstractVector` — full state vector of length `∏ dof_dims[i]`
- `dof_dims  :: AbstractVector{Int}` — local Hilbert-space dimensions, one per site

# Example
```julia
ψ = as_state(randn(8), [2, 2, 2])   # L=3 spin-½ chain
ndims(ψ.data)  # 3
dim(ψ.indices[1])  # 2
```

# See also
[`bipartition_matrix`](@ref), [`QTensor`](@ref)
"""
function as_state(v::AbstractVector, dof_dims::AbstractVector{Int})
    indices = Tuple(lower(Symbol(:σ, i), dof_dims[i]) for i in eachindex(dof_dims))
    data = reshape(v, dof_dims...)
    return QTensor(data, indices)
end

