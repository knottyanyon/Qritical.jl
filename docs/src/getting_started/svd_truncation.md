# SVD and Truncation

SVD is the workhorse of every MPS algorithm: it appears in canonicalization sweeps, TEBD gate updates, and state compression. `tensor_svd` provides a single interface that handles all three, returning the decomposition together with an exact truncation error so you always know how much spectral weight was discarded.

## Truncation strategies

Choose a strategy by constructing the appropriate [`AbstractTruncation`](@ref) subtype:

| Type | Keep condition | Typical use |
|:-----|:---------------|:------------|
| [`KeepFirst(r)`](@ref) | Largest `r` singular values | Fixed bond dimension cap |
| [`KeepAbove(atol)`](@ref) | Every σ with `σ > atol` | Absolute noise floor |
| [`KeepRelative(rtol)`](@ref) | Every σ with `σ/σ_max > rtol` | Relative threshold |

## `tensor_svd`

[`tensor_svd`](@ref) takes an `IndexedTensor`, a `Bipartition` describing how to split the legs, and a truncation strategy. It returns a named tuple `(; U, S, Vd, ε)`:

| Field | Type | Meaning |
|:------|:-----|:--------|
| `U` | `IndexedTensor` | Left isometry; left-partition indices + `BondIndex` |
| `S` | `Vector{<:Real}` | Retained singular values, descending |
| `Vd` | `IndexedTensor` | Right isometry; `BondIndex` + right-partition indices |
| `ε` | `Real` | `‖discarded singular values‖₂` — exact truncation error |

```jldoctest svd
julia> using Qritical, LinearAlgebra

julia> vL, σ, vR = upper(:vL, 2), lower(:σ, 2), upper(:vR, 3);

julia> A = IndexedTensor(rand(2, 2, 3), (vL, σ, vR));

julia> bp = bipartition(Partition(vL, σ), A);

julia> (; U, S, Vd, ε) = tensor_svd(A, bp, KeepFirst(4));

julia> length(S)   # at most 4, capped at rank(A) = min(4, 3) = 3
3

julia> S == sort(S; rev=true)   # sorted descending
true

julia> norm(U.data' * reshape(U.data, 4, :) - I) < 1e-12   # U†U ≈ I
true
```

### Reconstruction and truncation error

With no truncation (`KeepFirst(r)` where `r ≥ rank(A)`) the factorization is exact:

```jldoctest svd
julia> (; U, S, Vd, ε) = tensor_svd(A, bp, KeepFirst(10));

julia> ε < 1e-14   # exact — nothing discarded
true

julia> U_mat  = reshape(U.data,  4, length(S));
julia> Vd_mat = reshape(Vd.data, length(S), 3);
julia> A_rec  = reshape(U_mat * Diagonal(S) * Vd_mat, 2, 2, 3);
julia> norm(A.data - A_rec) < 1e-12
true
```

### Re-indexing

`U` and `Vd` carry the original partition indices plus a new [`BondIndex`](@ref) whose `ndim` equals the number of retained singular values:

```jldoctest svd
julia> bond = U.indices[end];   # BondIndex is appended to left indices

julia> typeof(bond)
BondIndex

julia> ndim(bond) == length(S)
true
```

### Threshold strategies

`KeepAbove` keeps every singular value strictly above the threshold:

```jldoctest svd
julia> σ_vals = [1.0, 0.5, 0.1, 1e-8];

julia> A2 = IndexedTensor(diagm(σ_vals), (upper(:r, 4), lower(:c, 4)));

julia> bp2 = bipartition(Partition(upper(:r, 4)), A2);

julia> (; S) = tensor_svd(A2, bp2, KeepAbove(0.05));

julia> S
3-element Vector{Float64}:
 1.0
 0.5
 0.1
```

`KeepRelative` keeps values above a fraction of the largest singular value:

```jldoctest svd
julia> (; S) = tensor_svd(A2, bp2, KeepRelative(0.2));

julia> S   # keep σ/1.0 > 0.2: 1.0 and 0.5 qualify
2-element Vector{Float64}:
 1.0
 0.5
```

## Rank-deficient inputs

`KeepFirst(r)` automatically caps at the numerical rank — it never retains machine-epsilon singular values even if `r` exceeds the true rank:

```jldoctest
julia> using Qritical, LinearAlgebra

julia> A = IndexedTensor(rand(3, 2) * rand(2, 4), (upper(:r, 3), lower(:c, 4)));

julia> bp = bipartition(Partition(upper(:r, 3)), A);

julia> (; S, ε) = tensor_svd(A, bp, KeepFirst(10));

julia> length(S) == rank(A.data)   # capped at true rank, not 10
true

julia> ε < 1e-12   # nothing meaningful discarded
true
```
