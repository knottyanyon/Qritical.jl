# SVD and Truncation

SVD is the workhorse of every MPS algorithm: it appears in canonicalization sweeps, TEBD gate updates, and state compression. `tensor_svd` provides a single interface that handles all three, returning the decomposition together with an exact truncation error so you always know how much spectral weight was discarded.

## Truncation strategies

Choose a strategy by constructing the appropriate [`AbstractTrunc`](@ref) subtype:

| Type | Keep condition | Typical use |
|:-----|:---------------|:------------|
| [`MaxBondDimTrunc(r)`](@ref) | Largest `r` singular values | Fixed bond dimension cap |
| [`ValCutoffTrunc(atol)`](@ref) | Every σ with `σ > atol` | Absolute noise floor |
| [`NoTrunc()`](@ref) | All singular values | Exact (no truncation) |

## `do_svd`

[`do_svd`](@ref) takes a `QTensor`, a `Bipartition` describing how to split the legs, and a truncation strategy. It returns a [`FullSVD`](@ref) (exact) or [`ReducedSVD`](@ref) (truncated) with the following fields:

| Field | Type | Meaning |
|:------|:-----|:--------|
| `U` | `QTensor` | Left unitary; left-partition indices + `lower` bond leg |
| `Σ` | `QTensor` | Diagonal singular-value matrix; spectrum via `SingValSpectrum(Σ.data.diag)` |
| `V` | `QTensor` | Right unitary; `upper` bond leg + right-partition indices |
| `ε` | `Real` | Truncation error (0 for `NoTrunc`, nonzero for `ValCutoffTrunc`/`MaxBondDimTrunc`) |

```jldoctest svd
julia> using Qritical, LinearAlgebra

julia> i, σ, j = upper(:i, 2), lower(:σ, 2), upper(:j, 3);

julia> A = QTensor(rand(2, 2, 3), (i, σ, j));

julia> bp = bipartition(Partition(i, σ), A);

julia> result = do_svd(A, bp, MaxBondDimTrunc(4));

julia> length(result.Σ.data.diag)   # at most 4, capped at rank(A) = min(4, 3) = 3
3

julia> result.Σ.data.diag == sort(result.Σ.data.diag; rev=true)   # sorted descending
true

julia> result.ε  # truncation error
0.0
```

### Reconstruction and truncation error

With no truncation (`KeepFirst(r)` where `r ≥ rank(A)`) the factorization is exact:

```jldoctest svd
julia> (; U, Σ, Vd, ε) = tensor_svd(A, bp, KeepFirst(10));

julia> ε < 1e-14   # exact — nothing discarded
true

julia> svs   = Σ.data.diag;

julia> U_mat = reshape(U.data,  4, length(svs));

julia> V_mat = reshape(Vd.data, length(svs), 3);

julia> norm(A.data - reshape(U_mat * Diagonal(svs) * V_mat, 2, 2, 3)) < 1e-12
true
```

### Re-indexing

`U` and `Vd` carry the original partition indices plus a new `TIx` bond leg whose `ndim` equals the number of retained singular values:

```jldoctest svd
julia> bond = U.indices[end];   # TIx{Lower} bond leg appended to left indices

julia> typeof(bond)
TIx{Lower}

julia> ndim(bond) == length(Σ.data.diag)
true
```

### Threshold strategies

`KeepAbove` keeps every singular value strictly above the threshold:

```jldoctest svd
julia> σ_vals = [1.0, 0.5, 0.1, 1e-8];

julia> A2 = IndexedTensor(diagm(σ_vals), (upper(:r, 4), lower(:c, 4)));

julia> bp2 = bipartition(Partition(upper(:r, 4)), A2);

julia> (; Σ) = tensor_svd(A2, bp2, KeepAbove(0.05));

julia> Σ.data.diag
3-element Vector{Float64}:
 1.0
 0.5
 0.1
```

`KeepRelative` keeps values above a fraction of the largest singular value:

```jldoctest svd
julia> (; Σ) = tensor_svd(A2, bp2, KeepRelative(0.2));

julia> Σ.data.diag   # keep σ/1.0 > 0.2: 1.0 and 0.5 qualify
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

julia> (; Σ, ε) = tensor_svd(A, bp, KeepFirst(10));

julia> length(Σ.data.diag) == rank(A.data)   # capped at true rank, not 10
true

julia> ε < 1e-12   # nothing meaningful discarded
true
```
