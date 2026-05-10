using LinearAlgebra

"""
    Bisection(left, right)
    Bisection(left, ndims)
    Bisection(tensor::IndexedTensor, left_indices)
    Bisection(tensor::IndexedTensor, T::Type{<:AbstractIndex})

A partition of a tensor's indices into two disjoint sets `left` and `right`.

The integer forms take explicit dimension positions; `ndims` infers `right` as
the complement of `left` within `1:ndims`. The `IndexedTensor` forms resolve
positions automatically: pass a vector of `AbstractIndex` objects to specify
the left set by identity, or pass an index type (`PhysicalIndex`/`BondIndex`)
to put all legs of that kind on the left.

# Examples
```jldoctest
julia> b = Bisection([1, 2], 5)
Bisection([1, 2], [3, 4, 5])

julia> b.left
2-element Vector{Int64}:
 1
 2

julia> b.right
3-element Vector{Int64}:
 3
 4
 5

julia> Bisection([1], [1, 2])
ERROR: ArgumentError: Indices [1] appear in both partitions.
[...]
```
"""
struct Bisection
    left::Vector{Int}
    right::Vector{Int}

    function Bisection(left::Vector{Int}, right::Vector{Int})
        shared = intersect(left, right)
        isempty(shared) || throw(ArgumentError("Indices $shared appear in both partitions."))
        length(unique(left)) == length(left) || throw(ArgumentError("left indices contain duplicates."))
        length(unique(right)) == length(right) || throw(ArgumentError("right indices contain duplicates."))
        all(>(0), left) || throw(ArgumentError("left indices must be positive integers."))
        all(>(0), right) || throw(ArgumentError("right indices must be positive integers."))
        new(left, right)
    end
end

Bisection(left::AbstractVector{Int}, right::AbstractVector{Int}) = Bisection(collect(left), collect(right))
Bisection(left::AbstractVector{Int}, ndims::Int) = Bisection(collect(left), setdiff(1:ndims, left))

function Bisection(tensor::IndexedTensor, left::AbstractVector{<:AbstractIndex})
    left_positions = map(left) do idx
        pos = findfirst(==(idx), tensor.indices)
        isnothing(pos) && throw(ArgumentError("Index $idx not found in tensor"))
        pos
    end
    Bisection(left_positions, ndims(tensor))
end

function Bisection(tensor::IndexedTensor, ::Type{T}) where {T <: AbstractIndex}
    left_positions = findall(i -> i isa T, collect(tensor.indices))
    isempty(left_positions) && throw(ArgumentError("No indices of type $T found in tensor"))
    Bisection(left_positions, ndims(tensor))
end

"""
    factorize_with_svd(tensor; discard_below_threshold=true, threshold=1e-3)

Compute the SVD of `tensor` and optionally truncate singular values below a
given threshold, returning the filtered factors.

# Arguments
- `tensor`: A matrix (or 2D array) to decompose. Must be a valid input for
  `LinearAlgebra.svd`.
- `discard_below_threshold`: If `true` (default), singular values below
  `threshold` are discarded and the corresponding columns of `U` and rows of
  `Vt` are removed. If `false`, the full untruncated factors are returned.
- `threshold`: Cutoff value for discarding singular values. Defaults to
  `1e-3`. Only used when `discard_below_threshold=true`.

# Returns
A 3-tuple `(U, Σ, Vt)` where:
- `U`  — `Matrix` of left singular vectors, shape `(m × k)`.
- `Σ`  — `Diagonal` matrix of singular values, shape `(k × k)`.
- `Vt` — `Matrix` of right singular vectors (already transposed), shape `(k × n)`.

Where `k` is the number of singular values kept after truncation (or
`min(m, n)` when `discard_below_threshold=false`).

# Logs
Emits two `@info` messages when `discard_below_threshold=true`:
- The threshold being applied.
- The Schmidt rank before and after truncation.

# Examples
```julia-repl
julia> M = rand(4, 8);
julia> U, Σ, Vt = factorize_with_svd(M; threshold=1e-2);
julia> size(U)    # (4, k) where k ≤ 4
julia> size(Vt)   # (k, 8)

# Reconstruct (approximate if truncated)
julia> U * Σ * Vt ≈ M

# No truncation
julia> U, Σ, Vt = factorize_with_svd(M; discard_below_threshold=false);
julia> size(Σ)    # (4, 4) — full rank
```

# Notes
- Singular values are sorted in descending order by `LinearAlgebra.svd`, so
  the mask discards the smallest values at the tail.
- The reconstruction `U * Σ * Vt` will not exactly equal the input when
  truncation is applied — the relative error grows with the magnitude of the
  discarded singular values.
- When `discard_below_threshold=false` the returned `Σ` is still a
  `Diagonal` matrix (not a plain vector), consistent with the truncated case.
"""
function factorize_with_svd(tensor; discard_below_threshold=true, threshold=1e-3)
    M_svd = LinearAlgebra.svd(tensor)
    U = M_svd.U
    Σ_diagonal = M_svd.S
    Vt = M_svd.Vt

    if discard_below_threshold
        @info "Discarding singular values below $threshold"
        mask_keep = Σ_diagonal .> threshold

        @info "Schmidt rank" before = length(Σ_diagonal) after = count(mask_keep)

        U_cleaned = U[:, mask_keep]
        Σ_cleaned = Diagonal(Σ_diagonal[mask_keep])
        Vt_cleaned = Vt[mask_keep, :]
        return U_cleaned, Σ_cleaned, Vt_cleaned
    else
        return U, Diagonal(Σ_diagonal), Vt
    end
end

"""
    validate_bipartition_indices(indices, ndims_tensor)

Validate the `indices` argument for a tensor bipartition.

# Arguments
- `indices`: A `Vector` of 1 or 2 `Tuple`s of positive integers.
- `ndims_tensor`: Number of dimensions of the target tensor.

# Returns
A `Tuple` of two `Vector{Int}` `(left_indices, right_indices)` where
`right_indices` is either taken from the second tuple or inferred as
the complement of the first.

# Throws
- `ArgumentError` on any of the following:
  - more than 2 tuples
  - any index out of range `[1, ndims_tensor]`
  - repeated index within a tuple
  - repeated index across tuples
  - combined index count exceeds `ndims_tensor`
"""
function validate_bipartition_indices(indices, ndims_tensor)
    if length(indices) > 2
        throw(ArgumentError("indices must contain 1 or 2 tuples, got $(length(indices))."))
    end

    for (t_idx, tup) in enumerate(indices)
        for idx in tup
            if idx < 1 || idx > ndims_tensor
                throw(
                    ArgumentError(
                        "Index $idx in tuple $t_idx is out of range. " *
                        "Tensor has $ndims_tensor dimensions, so valid indices are 1–$ndims_tensor.",
                    ),
                )
            end
        end
    end

    for (t_idx, tup) in enumerate(indices)
        if length(tup) != length(unique(tup))
            throw(ArgumentError("Tuple $t_idx contains repeated indices: $tup."))
        end
    end

    if length(indices) == 2
        shared = intersect(indices[1], indices[2])
        if !isempty(shared)
            throw(
                ArgumentError(
                    "Index/indices $shared appear in both tuples. " *
                    "Each dimension must belong to exactly one side of the bipartition.",
                ),
            )
        end
    end

    left_indices = Int[i for i in indices[1]]
    if length(indices) == 1
        right_indices = setdiff(1:ndims_tensor, left_indices)
    else
        right_indices = Int[i for i in indices[2]]
    end

    if length(left_indices) + length(right_indices) > ndims_tensor
        throw(
            ArgumentError(
                "Total number of indices ($(length(left_indices) + length(right_indices))) " *
                "exceeds the number of tensor dimensions ($ndims_tensor).",
            ),
        )
    end

    return left_indices, right_indices
end

"""
    reshape_tensor_for_bipartition(tensor, indices)

Non-mutating version. Returns a new reshaped matrix without modifying `tensor`.
See `reshape_tensor_for_bipartition!` for the mutating version.
"""
function reshape_tensor_for_bipartition(tensor, indices)
    left_indices, right_indices = validate_bipartition_indices(indices, ndims(tensor))

    perm = vcat(left_indices, right_indices)
    permuted_tensor = permutedims(tensor, perm)

    sz = size(tensor)
    left_dim = prod(sz[i] for i in left_indices)
    right_dim = prod(sz[i] for i in right_indices)

    return reshape(permuted_tensor, left_dim, right_dim)
end

function reshape_tensor_for_bipartition(tensor, b::Bisection)
    sz = size(tensor)
    perm = vcat(b.left, b.right)
    permuted_tensor = permutedims(tensor, perm)
    left_dim = prod(sz[i] for i in b.left)
    right_dim = prod(sz[i] for i in b.right)
    return reshape(permuted_tensor, left_dim, right_dim)
end

"""
    reshape_tensor_for_bipartition!(tensor, indices)

Mutating version. Overwrites `tensor` in-place with the reshaped matrix and
returns it. `tensor` must be a `Vector` or a 1D-contiguous `Array` for the
in-place `reshape` to succeed, or Julia will throw a `DimensionMismatch` at
runtime. If your tensor is a general N-d array, prefer the non-mutating
`reshape_tensor_for_bipartition` instead.

See `reshape_tensor_for_bipartition` for the non-mutating version.
"""
function reshape_tensor_for_bipartition!(tensor, indices)
    left_indices, right_indices = validate_bipartition_indices(indices, ndims(tensor))

    perm = vcat(left_indices, right_indices)
    permuted_tensor = permutedims(tensor, perm)

    sz = size(tensor)
    left_dim = prod(sz[i] for i in left_indices)
    right_dim = prod(sz[i] for i in right_indices)

    return reshape!(tensor, permuted_tensor, left_dim, right_dim)
end

"""
    get_schmidt_coefficients(tensor; discard_below_threshold=true, threshold=1e-3)

Compute the Schmidt coefficients of a bipartite state from its matrix
representation.

The Schmidt coefficients `λ_i` are the normalised singular values of the
reshaped state matrix:

    λ_i = σ_i / ‖ψ‖,    ‖ψ‖ = √(Σ_j σ_j²)

where the norm is always computed from the **full** set of singular values
before any truncation, so that the truncation error is reflected in the
returned coefficients (i.e. `Σ λ_i² ≤ 1` after truncation, with equality
only when no values are discarded).

# Arguments
- `tensor`: A matrix representing the bipartite state. Must already be
  reshaped for the desired bipartition via `reshape_tensor_for_bipartition`
  before calling this function.
- `discard_below_threshold`: If `true` (default), Schmidt coefficients whose
  corresponding singular value is below `threshold` are discarded.
- `threshold`: Cutoff for discarding singular values. Defaults to `1e-3`.

# Returns
A `Vector{Float64}` of Schmidt coefficients `λ_i`, sorted in descending
order, satisfying `Σ λ_i² ≤ 1` (with equality when no truncation occurs or
the state is exactly represented by the kept terms).

# Logs
When `discard_below_threshold=true`, emits two `@info` messages:
- The threshold being applied.
- The Schmidt rank before and after truncation.

# Examples
```julia-repl
julia> ψ_reshaped = reshape_tensor_for_bipartition(ψ, [(1,)]);  # (2, 512)
julia> λ = get_schmidt_coefficients(ψ_reshaped; threshold=1e-6);
julia> sum(λ .^ 2)   # ≈ 1.0 if no significant values were discarded
julia> length(λ)     # Schmidt rank
```

# Notes
- Uses `LinearAlgebra.svdvals` rather than the full `svd`, which is more
  efficient when the left/right singular vectors are not needed.
- For a normalised state `‖ψ‖ ≈ 1`, so `λ_i ≈ σ_i` directly.
"""
function get_schmidt_coefficients(tensor; discard_below_threshold=true, threshold=1e-3)
    singular_vals = LinearAlgebra.svdvals(tensor)

    # compute full norm before any truncation — must not be recomputed after
    norm_ψ = sqrt(sum(singular_vals .^ 2))

    if discard_below_threshold
        @info "Discarding Schmidt coefficients below $threshold"
        mask_keep = singular_vals .> threshold
        @info "Schmidt rank" before = length(singular_vals) after = count(mask_keep)
        singular_vals = singular_vals[mask_keep]
    end

    return singular_vals ./ norm_ψ
end

"""
    get_entanglement_entropy(schmidt_coeffs)

Compute the von Neumann entanglement entropy from a vector of Schmidt
coefficients.

The entropy is defined as:

    S = -Σ_i  λ_i²  log(λ_i²)

where `p_i = λ_i²` are the eigenvalues of the reduced density matrix.
The result is in **nats** (natural logarithm). Divide by `log(2)` for bits.

# Arguments
- `schmidt_coeffs`: A `Vector` of Schmidt coefficients `λ_i` as returned by
  `get_schmidt_coefficients`. Must satisfy `Σ λ_i² ≈ 1`.

# Returns
A non-negative `Float64`. Returns `0.0` for a product state (single non-zero
Schmidt coefficient). Reaches its maximum of `log(d)` for a maximally
entangled state across a `d`-dimensional bipartition.

# Throws
- `AssertionError` if `Σ λ_i²` deviates from 1 by more than `1e-6`,
  indicating unnormalised input (e.g. raw singular values were passed instead
  of Schmidt coefficients).

# Examples
```julia-repl
julia> λ = get_schmidt_coefficients(ψ_reshaped; threshold=1e-6);
julia> S = get_entanglement_entropy(λ)

# convert nats → bits
julia> S / log(2)

# product state → zero entropy
julia> get_entanglement_entropy([1.0])
0.0
```

# Notes
- Zero-valued coefficients are excluded before taking the logarithm to avoid
  `NaN` from `0 * log(0)` (the limit is 0 by continuity).
- For the middle bipartition of a strongly entangled state, `S` will be
  significantly larger than for an edge bipartition, directly reflecting the
  entanglement structure of the state.
"""
function get_entanglement_entropy(schmidt_coeffs)
    @assert abs(sum(schmidt_coeffs .^ 2) - 1.0) < 1e-6 "schmidt_coeffs must be normalised: Σλ² must equal 1"

    p = schmidt_coeffs[schmidt_coeffs .> 0] .^ 2
    return -sum(p .* log2.(p))
end
