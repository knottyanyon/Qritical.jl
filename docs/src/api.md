# API Reference

Complete reference for all public symbols exported by `Qritical`.

---

## Index layer

```@docs
IxLoc
Upper
Lower
TIx
MulTIx
dim
label
which_space
upper
lower
uppers
lowers
uppers_range
lowers_range
bond_label
```

---

## QTensor

```@docs
QTensor
Partition
Bipartition
complement
bipartition
group_legs
```

---

## SVD & truncation

```@docs
AbstractTrunc
NoTrunc
MaxBondDimTrunc
ValCutoffTrunc
FullSVD
ReducedSVD
do_svd
```

---

## Spectra & entanglement

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

---

## State utilities & I/O

```@docs
bipartition_matrix
as_state
load_array
```

---

## MPS & canonical forms

```@docs
AbstractMPSForm
CanonicalForm
VidalForm
ArbitraryForm
FiniteMPS
to_mps
add_mps
```

### Canonicalization

```@docs
CanonicalizeConfig
LeftCanonical
RightCanonical
BondCanonical
SiteCanonical
canonicalize
canonical_error
is_canonical
```

### Vidal (Γ–Λ) form

```@docs
to_vidal
to_canonical
```

### Observables & correlators

```@docs
overlap
local_expectation
two_point
```
