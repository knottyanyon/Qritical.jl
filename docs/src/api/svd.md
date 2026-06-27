# SVD & Truncation

## Physics motivation

The Schmidt decomposition is the fundamental tool of tensor-network methods. For a bipartite quantum state ``|\psi\rangle \in \mathcal{H}_A \otimes \mathcal{H}_B``, the Schmidt decomposition reads

```math
|\psi\rangle = \sum_{i=1}^{r} \sigma_i \, |u_i\rangle_A \otimes |v_i\rangle_B
```

where ``\sigma_i \geq 0`` are the Schmidt values, ``r`` is the Schmidt rank, and ``\{|u_i\rangle\}``, ``\{|v_i\rangle\}`` are orthonormal bases. This is exactly a singular-value decomposition of the reshaped state tensor across the bipartition cut.

The Schmidt values carry the entanglement: the von Neumann entropy is ``S = -\sum_i \sigma_i^2 \log \sigma_i^2``. In a weakly entangled state — which includes the ground states of local gapped Hamiltonians by the area law — most Schmidt values are negligibly small. **Truncation** keeps only the top ``r`` values and discards the rest, controlled by `AbstractTrunc`. This is what makes MPS efficient: rather than storing ``d^L`` amplitudes, one stores ``L`` tensors each of bond dimension ``\chi \ll d^{L/2}``.

`do_svd` separates the *exact* case from the *approximate* case at the type level:

- `FullSVD` is returned when `NoTrunc()` is used — no error, no ``\varepsilon`` field, reconstruction is exact.
- `ReducedSVD` is returned for any truncating strategy — the discarded weight ``\varepsilon = \|A - U \Sigma V^\dagger\|_F`` is always present and impossible to forget because it lives in the type.

Before any truncation strategy sees the spectrum, LAPACK rounding noise is stripped using the Golub–Van Loan numerical-rank criterion (``\sigma_k > n \varepsilon_\text{mach} \sigma_1``). This prevents floating-point artifacts from being counted as genuine Schmidt values.

---

```@docs
AbstractTrunc
NoTrunc
MaxBondDimTrunc
ValCutoffTrunc
FullSVD
ReducedSVD
do_svd
```
