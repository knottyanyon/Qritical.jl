# State Utilities & I/O

## Physics motivation

Before any tensor-network computation can begin, the physical state must be represented correctly as an indexed tensor. This small layer bridges raw numerical data (arrays loaded from disk, flat state vectors) to the `QTensor` and `FiniteMPS` types with the correct leg/space structure.

`load_array` is the single entry point for input data: it dispatches on file extension (`.jls` for Julia serialised course data, `.txt`/`.dat` for plain text, `.npy` for NumPy interop) and returns a raw dense array. The course exercises load matrix and state data through this path.

`as_state` reshapes a flat ``d^L`` state vector into a rank-``L`` tensor ``(\underbrace{d, d, \ldots, d}_{L})``. The convention used throughout Qritical follows Julia's column-major (Fortran) ordering: site 1 is the *fast* (least-significant bit) index. This is opposite to the physics ``|\sigma_1 \sigma_2 \cdots \sigma_L\rangle`` Kronecker ordering, where site 1 is the slowest index. The offset matters when comparing MPS expectation values to brute-force full-state contractions — the [`local_expectation`](@ref) docstring discusses the mapping explicitly.

`bipartition_matrix` reshapes the full state tensor into a ``(d^k \times d^{L-k})`` matrix at a chosen bond cut ``k``. This is exactly the matrix whose SVD gives the Schmidt decomposition of the state across that cut — the starting point for building an MPS by iterated SVD.

---

## Functions

```@docs
load_array
as_state
bipartition_matrix
```
