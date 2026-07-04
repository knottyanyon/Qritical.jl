# Exact Diagonalization

## Physics motivation

**Exact diagonalization (ED)** is the gold standard for small quantum systems: build the
full ``d^L \times d^L`` Hamiltonian matrix and find its eigenvalues exactly, no
approximation. Every MPS or TEBD result should be cross-validated against ED for small
system sizes to make sure the numerics are correct. Think of ED as your reference
implementation: when `EDResult.energy` agrees with the imaginary-time TEBD result, you
can trust the TEBD for larger systems where ED is no longer feasible.

The catch is cost. Storing the full matrix requires ``O(d^{2L})`` memory and dense
diagonalization costs ``O(d^{3L})`` — for a spin-½ chain (``d = 2``) that is ``4^L``
and ``8^L`` operations. This is feasible up to around ``L \approx 12`` on a laptop; at
``L = 20`` you have over a million states and the dense matrix takes gigabytes.

### Sparse Lanczos vs full diagonalization

Qritical offers two modes, selected by a symbol argument:

**`:ground` — Krylov-Lanczos via KrylovKit.** Instead of storing the full matrix, Krylov
methods build a Krylov subspace iteratively: starting from a random vector ``|v_0\rangle``,
each step applies ``H`` once and orthogonalizes. After ``m`` steps you have an
``m \times m`` tridiagonal matrix whose eigenvalues approximate the extremal eigenvalues
of ``H``. The cost per step is one matrix-vector product — ``O(d^L \cdot \text{nnz})``
where ``\text{nnz}`` is the number of nonzeros in the sparse Hamiltonian. The number of
Krylov steps ``m`` needed to converge is sublinear in ``d^L`` for well-separated ground
states, making Lanczos far cheaper than full diagonalization for large Hilbert spaces.

**`:full` — dense diagonalization.** Converts the sparse matrix to a dense
`Hermitian` array and calls Julia's `eigen`. This gives you the *entire spectrum*, which
is useful for studying level statistics, spectral functions, or the gap between the ground
state and first excited state. The cost is genuinely ``O(d^{3L})``, so use this only for
small systems.

```math
\text{cost ratio:} \quad \frac{\text{full}\ \text{diag}}{\text{Lanczos}} \sim \frac{d^{3L}}{m \cdot d^L} = \frac{d^{2L}}{m}
```

For ``L = 10``, ``d = 2``, that is a factor of roughly ``10^6 / m``.

### The ``2^{20}`` guard

`matrix_repr` refuses to build a Hilbert space larger than ``2^{20} \approx 10^6``
states. This is not an arbitrary limit — it reflects the point at which the dense matrix
would take several gigabytes of RAM and the full diagonalization would run for hours.
Beyond this scale you should be using an MPS algorithm. The guard makes this failure loud
(an `ArgumentError`) rather than silent (an OOM crash or a multi-hour runtime).

### Cross-validation pattern

The typical workflow is:

1. Fix a small system (``L \leq 10``).
2. Find the ground state with `solve(H, GroundState(), ExactDiagonalization(:ground))`.
3. Compare `EDResult.energy` against the power-method or imaginary-time TEBD result.
4. Once they agree, scale up with the MPS algorithm.

---

```@docs
GroundState
ExactDiagonalization
EDResult
solve(H::LatticeOperator, ::GroundState, ::ExactDiagonalization{:ground})
solve(H::LatticeOperator, ::GroundState, ::ExactDiagonalization{:full})
```
