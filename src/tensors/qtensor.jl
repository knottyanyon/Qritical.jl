using LinearAlgebra        # provides `svd`, `Diagonal`, `norm`, `mul!`, `BLAS.gemm!`, etc. 
using TensorOperations    # provides `@tensor` macro for Einstein-summation contractions 

# Naming deviates from Julia convention ({T,N}) intentionally:
#   Element — the scalar field (e.g. ComplexF64), not constrained to <:Number.
#   Valence — number of legs (Penrose/Biamonte term); "rank" is reserved for the SVD
"""
    QTensor{Element, Valence, D<:AbstractArray{Element,Valence}}

A `Valence`-index tensor with a named, covariant/contravariant index attached to each leg.
Subtyping `AbstractArray` makes it transparent to `@tensor` contractions.

The backing store `D` is a free type parameter: `:native` mode uses plain
`Array{Element,Valence}` from Julia Base; future backends may substitute other `AbstractArray`
subtypes without touching algorithm code. The goal is to facilitate the use of efficient
sparse array descriptions if one wishes to implement symmetry details to take advantage
of the block structure of the Hamiltonians that we usually deal with.

# Constructor

```julia
QTensor(data, indices)
```

`data` must be an `AbstractArray` encoding the numerical data of the tensor and `indices`
an `NTuple` of `AbstractIx` values. The shared `Valence` parameter in both types enforces
that the number of indices equals the array rank at the type level.

# Fields

  - `data::D` — the backing data array
  - `indices::NTuple{Valence,AbstractIx}` — named, variance-tagged legs

# Examples

```jldoctest
julia> data = [1.0 2.0; 3.0 4.0]
2×2 Matrix{Float64}:
 1.0  2.0
 3.0  4.0

julia> i = upper(:i, 2);
       j = lower(:j, 2);

julia> t = QTensor(data, (i, j))
QTensor{Float64, 2, ...}

julia> t[1, 1]   # parametric struct: `{Element, Valence, D}` are 3 type parameters; `D<:AbstractArray{Element,Valence}` constrains D to be an array of the right element type and rank; `<: AbstractArray{Element,Valence}` makes QTensor behave like a Julia array
1.0
```                              # the raw backing array; for native mode this is `Array{Element,Valence}` (e.g. `Array{ComplexF64,3}`); future backends may use TensorMap or BlockSparse
```
"""
struct QTensor{Element,Valence,D<:AbstractArray{Element,Valence}} <:   # parametric struct: `{Element, Valence, D}` are 3 type parameters; `D<:AbstractArray{Element,Valence}` constrains D to be an array of the right element type and rank; `<: AbstractArray{Element,Valence}` makes QTensor behave like a Julia array 
       AbstractArray{Element,Valence}
    data::D                              # the raw backing array; for native mode this is `Array{Element,Valence}` (e.g. `Array{ComplexF64,3}`); future backends may use TensorMap or BlockSparse
    indices::NTuple{Valence,AbstractIx}  # `NTuple{Valence, AbstractIx}` = a Tuple of exactly `Valence` elements of type AbstractIx; statically sized at compile time; Python: `tuple[AbstractIx, ...]` but length is fixed

    function QTensor(
        data::D, indices::NTuple{Valence,AbstractIx}
    ) where {Element,Valence,D<:AbstractArray{Element,Valence}}   # inner constructor; `where {Element,Valence,D<:...}` declares the type parameters; Python: `def __init__(self, data, indices)` with validation
        for k in 1:Valence # make sure that the local state space size of a leg matched the actual array dimensions
            size(data, k) == dim(indices[k]) || throw(   # `size(data, k)` = dimension of axis k (1-indexed); `dim(indices[k])` = declared dim of that leg; `||` short-circuit: if false, throw; physics: each array axis must have the same size as the corresponding index's Hilbert space dimension
                ArgumentError(
                    "leg $k: array size $(size(data,k)) ≠ index dim $(dim(indices[k]))",  # `$k` and `$(expr)` = string interpolation 
                ),
            )
        end
        new{Element,Valence,D}(data, indices)   # allocate and initialise the struct; must pass ALL type parameters explicitly inside inner constructors: `new{Element,Valence,D}(...)`
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
`n == Valence` and each `dₖ == dim(t.indices[k])`.  The two-argument form is equivalent
to `size(t)[dim]` but does not materialise the full tuple.

`Base.size` is the primary query in Julia's `AbstractArray` interface: `ndims`, `length`,
`axes`, and `CartesianIndices` all derive from it.

# Examples

```jldoctest
julia> t = QTensor([1.0 2.0; 3.0 4.0], (upper(:i, 2), lower(:j, 2)));

julia> size(t)
(2, 2)

julia> size(t, 1)
2
```
"""
Base.size(t::QTensor) = size(t.data)   # `Base.size` = extend Julia's built-in `size`. delegates to the backing array; the AbstractArray machinery derives `ndims`, `length`, `axes` from this

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
julia> t = QTensor([1.0 2.0; 3.0 4.0], (upper(:i, 2), lower(:j, 2)));

julia> t[1, 2]
2.0

julia> t[1, :]
2-element Vector{Float64}:
 1.0
 2.0   # `Base.getindex` = extend `[]` operator. `i...` = splat varargs; delegates to the backing array so `t[1,2]` works exactly like `t.data[1,2]`
```
"""
Base.getindex(t::QTensor, i...) = getindex(t.data, i...)   # `Base.getindex` = extend `[]` operator. `i...` = splat varargs; delegates to the backing array so `t[1,2]` works exactly like `t.data[1,2]`

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
julia> t = QTensor(zeros(2, 2), (upper(:i, 2), lower(:j, 2)));

julia> t[1, 1] = 3.14;
       t[1, 1]
3.14   # `Base.setindex!` = extend `[]=` mutation operator. `!` suffix = convention for mutating functions; writes through to backing array; `t[1,1] = 3.14` changes `t.data[1,1]` in place
```
"""
Base.setindex!(t::QTensor, v, i...) = setindex!(t.data, v, i...)   # `Base.setindex!` = extend `[]=` mutation operator. `!` suffix = convention for mutating functions; writes through to backing array; `t[1,1] = 3.14` changes `t.data[1,1]` in place

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
    `unsafe_convert`, enabling direct BLAS dispatch on the backing buffer.   # `Base.IndexStyle` = declare whether the array uses linear or cartesian indexing; `::Type{QTensor{E,O,D}}` dispatches on the TYPE itself (not an instance); `IndexStyle(D)` delegates to the backing array D — for plain `Array` this returns `IndexLinear()`, enabling BLAS dispatch
"""
Base.IndexStyle(::Type{QTensor{E,O,D}}) where {E,O,D} = IndexStyle(D)   # `Base.IndexStyle` = declare whether the array uses linear or cartesian indexing; `::Type{QTensor{E,O,D}}` dispatches on the TYPE itself (not an instance); `IndexStyle(D)` delegates to the backing array D — for plain `Array` this returns `IndexLinear()`, enabling BLAS dispatch

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
julia> t = QTensor(ones(3, 4), (upper(:i, 3), lower(:j, 4)));

julia> strides(t)
(1, 3)
```
"""
Base.strides(t::QTensor) = strides(t.data)   # `Base.strides` = return memory strides (elements between adjacent entries along each axis); for column-major Array: `(1, size1, size1*size2, ...)`; enables BLAS/LAPACK to operate on QTensor data directly via raw pointer

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
function Base.unsafe_convert(::Type{Ptr{Element}}, t::QTensor{Element}) where {Element}   # `Base.unsafe_convert` = return a raw C pointer to the first element; `Ptr{Element}` = typed raw pointer ; called internally by BLAS routines; `::Type{Ptr{Element}}` dispatches on the target pointer type; `where {Element}` = type parameter binding
    return Base.unsafe_convert(Ptr{Element}, t.data)   # delegate to the backing array's unsafe_convert; `t.data` is the actual `Array{Element,N}` whose pointer BLAS/LAPACK can use directly
end

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
julia> σ = upper(:σ, 2);
       vL = upper(:vL, 3);
       vR = lower(:vR, 4);

julia> A = QTensor(rand(2, 3, 4), (σ, vL, vR));

julia> bp = Bipartition(Partition([σ, vL]), Partition([vR]));

julia> M = group_legs(A, bp);

julia> size(M)
(6, 4)

julia> M.indices[1] isa MulTIx
true   # `collect(A.indices)` converts the NTuple to a Vector so we can use `findfirst`; `A.indices` is a fixed-length NTuple at compile time
```
"""
function group_legs(A::QTensor, bp::Bipartition)
    all_ix = collect(A.indices)   # `collect(A.indices)` converts the NTuple to a Vector so we can use `findfirst`; `A.indices` is a fixed-length NTuple at compile time

    # locate each partition leg in A.indices (by equality)
    _find(ix) = findfirst(==(ix), all_ix)   # `_find` is a local function. `findfirst(pred, arr)` = index of first element satisfying pred ` = partial application of `==`. returns `nothing` if not found
    left_pos = _find.(bp.left)    # broadcast `_find` over all left-partition legs; result is a Vector of Int (positions) or nothing; `.(...)` = element-wise 
    right_pos = _find.(bp.right)  # same for right-partition legs

    any(isnothing, left_pos) && throw(   # `any(pred, iter)` = Python `any(pred(x) for x in iter)`; if any left leg was not found → error
        ArgumentError(
            "group_legs: one or more left-partition legs were not found in the tensor."
        ),
    )
    any(isnothing, right_pos) && throw(   # same check for right legs
        ArgumentError(
            "group_legs: one or more right-partition legs were not found in the tensor."
        ),
    )

    covered = Set{Int}([left_pos..., right_pos...])   # `Set{Int}` = unordered set of integers ; `[a..., b...]` = concatenate two arrays. the Set deduplicates to find how many DISTINCT legs are covered
    length(covered) < length(all_ix) && throw(   # if not all legs are covered → error; `length(...)` = Python `len(...)`
        ArgumentError(
            "group_legs: tensor has $(length(all_ix)) leg(s) but the bipartition " *   # `*` = string concatenation 
            "covers only $(length(covered)) — every leg must appear on exactly one side.",
        ),
    )

    # permute axes so left legs come first (in bp.left order), then right legs
    perm = [left_pos..., right_pos...]   # permutation vector: left positions first, then right positions; e.g. for legs (σ, vL, vR) with left=[σ,vL] and right=[vR], perm=[1,2,3]
    data_p = isempty(perm) ? A.data : permutedims(A.data, perm)   # `permutedims(A, perm)` = reorder array axes ; uses ternary to avoid an error when perm is empty

    # reshape into a 2D matrix
    left_dim = prod(dim, bp.left; init=1)    # total row dimension = product of left-leg dims; `prod(f, iter; init=1)` = reduce product with function f ; `init=1` handles empty left partition
    right_dim = prod(dim, bp.right; init=1)  # total column dimension = product of right-leg dims
    data_2d = reshape(data_p, left_dim, right_dim)   # `reshape(A, m, n)` = reinterpret as m×n matrix  = first index varies fastest; this is the inverse of the partition ordering above

    # attach fused index metadata
    left_ix = MulTIx(Tuple(bp.left))    # `Tuple(bp.left)` converts Vector to Tuple ; then MulTIx outer constructor auto-generates a label from the constituent labels
    right_ix = MulTIx(Tuple(bp.right))  # same for right partition

    QTensor(data_2d, (left_ix, right_ix))   # construct the rank-2 QTensor with fused index metadata; `(left_ix, right_ix)` is a 2-tuple (NTuple{2, AbstractIx})
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
TensorOperations.tensorstructure(t::QTensor, i::Int, ::Bool) = t.indices[i]   # `TensorOperations.tensorstructure` = extension hook for the `@tensor` contraction engine; returns the AbstractIx metadata for leg `i`; `::Bool` matches the `conjA` argument but is ignored (unnamed `::Bool` = type-only dispatch, value discarded); by returning the full AbstractIx instead of just `size(t,i)`, we expose label, direction, and dim to the contraction engine

# ==== Adjoint & flips =========================================================

"""
    Base.adjoint(A::QTensor) -> QTensor
    A'

Return the Hermitian adjoint of `A` under the von Delft covariant convention:
**flip every leg's variance, reverse the leg order, and conjugate the data**,

```math
[A^{\\dagger}]^{\\beta}_{\\sigma\\alpha} := \\overline{A^{\\alpha\\sigma}_{\\beta}},
```

generalising ``(MN)^{\\dagger} = N^{\\dagger}M^{\\dagger}`` to arbitrary valence.
Labels are **not** primed — the daggering is carried entirely by the variance
and the leg positions. Because position `p` is array axis `p` (row = leftmost),
the leg reversal is mirrored on the backing array:
`(A').data == conj(permutedims(A.data, reverse(1:ndims(A))))`.

The adjoint is an involution, `(A')' == A`, and reproduces the familiar matrix
adjoint for valence-2 tensors: `(A').data == adjoint(A.data)`.

This is the same operation as TensorKit's `adjoint`: swap domain and codomain,
dualise every leg, reverse their order.

See also: [`dagger`](@ref), [`flip`](@ref)
"""
function Base.adjoint(A::QTensor)   # `Base.adjoint` = extend `A'` operator .T`
    N = length(A.indices)   # `length(A.indices)` = number of legs = Valence; `A.indices` is a fixed-length NTuple; physics: adjoint reverses leg order (row ↔ column swap generalised to N legs)
    data = conj(permutedims(A.data, ntuple(k -> N + 1 - k, N)))   # `ntuple(f, N)` = create a Tuple of length N with `f(k)` at position k ; `N+1-k` = reverse index order; `permutedims` = transpose. `conj` = complex conjugate of all elements; physics: adjoint = complex conjugate + reverse legs
    return QTensor(data, reverse(map(flip, A.indices)))   # `map(flip, A.indices)` = flip each leg's variance: Upper↔Lower ^T
end

"""
    dagger(A::QTensor) -> QTensor

Syntactic sugar for [`Base.adjoint`](@ref): `dagger(A) == A'`. Provided for
readability in physics-flavoured code where ``A^{\\dagger}`` is spelled out.
"""
dagger(A::QTensor) = adjoint(A)   # alias: `dagger(A) == A'`; provided for physics-readable code where A† is spelled out explicitly; `adjoint` is Julia's standard name

"""
    flip(A::QTensor, i::Integer) -> QTensor

Raise or lower leg `i` of `A`: return a `QTensor` whose `i`-th index has the
opposite variance, all other legs untouched. With the trivial metric of an
orthonormal basis this is numerically free, so **the backing array is reused
as-is** — no copy, no conjugation, no permutation.

Contrast with the adjoint, which flips *every* leg, reverses their order, and
conjugates the data. Flipping a single end of a contracted bond breaks the
upper-with-lower pairing rule; flip bonds at **both** ends (this is what a
centre-move across a bond does).

See also: [`flip(::TIx)`](@ref), [`Base.adjoint(::QTensor)`](@ref)
"""
function flip(A::QTensor, i::Integer)   # flip a SINGLE leg's variance; `i::Integer` = integer. different from `Base.adjoint` which flips ALL legs, reverses order, and conjugates data
    1 <= i <= length(A.indices) ||   # bounds check: `1 <= i <= N` is Julia's chained comparison. `||` short-circuit: throw if out of range
        throw(
            ArgumentError(
                "flip: leg $i out of range for a valence-$(length(A.indices)) tensor."
            ),
        )
    indices = ntuple(k -> k == i ? flip(A.indices[k]) : A.indices[k], length(A.indices))   # `ntuple(f, N)` = build a tuple of length N; `k == i ? flip(...) : ...` = ternary: flip only leg i; all other legs unchanged; Python: `tuple(flip(ix) if k==i else ix for k, ix in enumerate(indices, 1))`
    return QTensor(A.data, indices)   # reuse A.data WITHOUT copying — the metric is identity in orthonormal basis, so flipping variance is purely syntactic; physics: raising/lowering an index with δᵢⱼ metric leaves the components unchanged
end

# ==== State utilities =========================================================

"""
    bipartition_matrix(A::QTensor, bp::Bipartition) -> QTensor

Reshape the tensor `A` into a matrix (valence-2 `QTensor`) by grouping its legs
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
bipartition_matrix(A::QTensor, bp::Bipartition) = group_legs(A, bp)   # physics-vocabulary alias for `group_legs`; "bipartition matrix" is the standard term in MPS literature for the reshaped tensor used in the Schmidt decomposition

"""
    as_state(v::AbstractVector, dof_dims::AbstractVector{Int}) -> QTensor

Reshape the coefficient vector `v` into a valence-`L` tensor with one physical
leg per site.  `dof_dims[i]` is the local Hilbert-space dimension at site `i`.

The result satisfies `vec(ψ.data) === v` (no copy; the tensor shares memory with
the input vector).  Each leg is labelled `:σ1`, `:σ2`, … and typed `Upper`: the
stored array is the ket-expansion **coefficient** tensor ``A^{\\sigma_1 \\ldots \\sigma_L}`` of ``|\\psi\\rangle = A^{\\vec{\\sigma}}|\\vec{\\sigma}\\rangle``,
whose physical indices are contravariant (incoming arrows, domain convention).

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

[`bipartition_matrix`](@ref), [`QTensor`](@ref)   # reshape a flat state vector into a rank-L tensor; `AbstractVector` = any 1D array; `AbstractVector{Int}` = any 1D array of integers; physics: |ψ⟩ = Σ_{σ₁...σ_L} A^{σ₁...σ_L}|σ₁...σ_L⟩ stored as an L-legged tensor
"""
function as_state(v::AbstractVector, dof_dims::AbstractVector{Int})   # reshape a flat state vector into a rank-L tensor; `AbstractVector` = any 1D array; `AbstractVector{Int}` = any 1D array of integers; physics: |ψ⟩ = Σ_{σ₁...σ_L} A^{σ₁...σ_L}|σ₁...σ_L⟩ stored as an L-legged tensor
    L = length(dof_dims)   # number of sites = number of physical legs; `length` = Python `len`
    indices = Tuple(upper(Symbol(:σ, i), dof_dims[i]) for i in 1:L)   # `Tuple(generator)` = collect a generator into a Tuple; `upper(Symbol(:σ, i), d)` creates e.g. upper(:σ1, 2); `Symbol(:σ, i)` = :σ1, :σ2, ... (Python has no Symbol; closest is string interpolation)
    # Kron product ordering: site 1 is most significant (changes slowest).
    # Julia reshape is column-major: first index varies fastest.
    # So reshape(v, d_L,...,d_1) gives data_raw[σ_L,...,σ_1] = v[kron_index], then
    # permutedims reverses to data[σ_1,...,σ_L] = v[kron_index(σ_1,...,σ_L)].
    data = permutedims(reshape(v, reverse(dof_dims)...), L:-1:1)   # CRITICAL: Julia is column-major (first index fastest), but kron convention has site 1 slowest; `reverse(dof_dims)` = reversed order dims; `...` = splat: expand the array as separate arguments to reshape; `L:-1:1` = reverse range ; this is the same pattern used in `apply_gate` — see that file's comments for a detailed explanation
    return QTensor(convert(Array{ComplexF64}, data), indices)   # `convert(Array{ComplexF64}, data)` = ensure element type is ComplexF64 ; needed because v might be real and physics calculations expect complex state vectors
end

