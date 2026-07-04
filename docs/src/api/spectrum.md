# Spectra & Entanglement

## Physics motivation

The singular values of an SVD are not just intermediate numerical results — they are physical observables that characterise the quantum state. Keeping them as a first-class object (rather than buried inside a factored matrix) makes entanglement analysis, convergence diagnostics, and canonical-form bookkeeping all natural.

The spectrum hierarchy in Qritical reflects three genuinely different physical situations:

**`SingValSpectrum`** is "just for a matrix": singular values from an SVD of an array with no further physical context. It carries the discarded weight ``\varepsilon`` and a `normalized` flag (whether ``\sum_i \sigma_i^2 = 1``). This is what lives at each bond of a `FiniteMPS`.

**`SchmidtSpectrum`** wraps a `SingValSpectrum` with physical meaning: it belongs to a *state bipartition*, records the `Bipartition` (which legs are the cut), and points to the `BondCenter` (which virtual bond it gauges). Entanglement entropy and Schmidt rank are only physically meaningful once both the cut and the bond location are specified — demanding a `SchmidtSpectrum` at a function boundary enforces this. A bare `SingValSpectrum` will not typecheck where a `SchmidtSpectrum` is expected.

**`EigValSpectrum`** stores eigenvalues from a diagonalisation (Hamiltonian, density operator, transfer matrix). Unlike singular values, eigenvalues may be signed or complex, and diagonalisation does not truncate — so there is no ``\varepsilon`` field.

The spectrum verbs — `schmidt_rank`, `spectral_gap`, `schmidt_values`, `entanglement_entropy`, `entanglement_spectrum` — are derived functions on `AbstractSpectrum`, not stored fields. The entanglement spectrum is ``-2 \log \sigma_i``; the von Neumann entropy is ``-\sum_i \sigma_i^2 \log \sigma_i^2`` with the ``0 \log 0 = 0`` convention. The spectral gap is ``\sigma_1 - \sigma_2``, a measure of how quickly entanglement falls off.

`Bond`, `OrthoCenter`, `BondCenter`, and `SiteCenter` provide the geometric language for describing where the orthogonality centre of an MPS sits:

- A **`BondCenter`** places the centre on a *link* — the link matrix is diagonal, positive, and descending (it *is* the Schmidt spectrum in the canonical gauge).
- A **`SiteCenter`** places the centre on a *tensor* — sites to the left are left-isometric, sites to the right are right-isometric, and the centre tensor is neither.

These types are kept distinct so that `BondCanonical` and `SiteCanonical` sweeps can never be confused with each other.

---

```@docs
Bond
OrthoCenter
BondCenter
SiteCenter
AbstractSpectrum
SingValSpectrum
EigValSpectrum
SchmidtSpectrum
schmidt_rank
spectral_gap
schmidt_values
entanglement_entropy
entanglement_spectrum
```
