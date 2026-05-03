using LinearAlgebra

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