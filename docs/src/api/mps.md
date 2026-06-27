# MPS & Canonical Forms

## Physics motivation

A **matrix product state** (MPS) for a chain of ``L`` sites with local Hilbert-space dimension ``d`` is a compressed representation of the full ``d^L``-dimensional state vector:

```math
|\psi\rangle = \sum_{\sigma_1, \ldots, \sigma_L} A^{\sigma_1} A^{\sigma_2} \cdots A^{\sigma_L} \, |\sigma_1 \cdots \sigma_L\rangle
```

where each ``A^{\sigma_i}`` is a ``\chi_{i-1} \times \chi_i`` matrix (rank-3 tensor with legs ``(\text{vL}, \sigma, \text{vR})``). The bond dimensions ``\chi_i`` control the accuracy of the approximation: ``\chi = 1`` gives a product state, and ``\chi = d^{L/2}`` recovers the exact state. For ground states of local gapped Hamiltonians, the area law guarantees that ``\chi`` needed for a faithful approximation grows only polynomially in ``L`` — this is the theoretical foundation of MPS efficiency.

The **canonical form** is the key operational concept. A site tensor ``A_i`` is *left-isometric* if ``A_i^\dagger A_i = I`` (when reshaped as ``(\chi_L d \times \chi_R)``), and *right-isometric* if ``B_i B_i^\dagger = I`` (reshaped as ``(\chi_L \times d\chi_R)``). Canonical forms make expectation values computable in ``O(L)`` time: when bra and ket MPS are contracted site by site, the ``A_i^\dagger A_i`` pair at each left-canonical site collapses to the identity, leaving only the open right index. Without canonicalization, each contraction would cost ``O(d^L)``.

`FiniteMPS` stores the ``L`` site tensors, the ``L+1`` bond singular-value spectra (boundary spectra are ``[1.0]``), a form tag recording which sites are isometric, and the accumulated truncation error ``\varepsilon``.

`to_mps` decomposes a full state tensor into an MPS by iterated SVD (carry propagation). The left sweep produces a left-canonical MPS: at each bond the carry ``(\chi_{i-1}, d_i, \ldots, d_L)`` is reshaped to ``(\chi_{i-1} d_i \times d_{i+1}\cdots d_L)``, SVD'd, and the isometric factor ``U`` becomes the site tensor while ``\Sigma V^\dagger`` propagates rightward. Per-bond cost is ``O(\chi^2 d)`` — linear in ``L``, not exponential in ``d``.

`add_mps` implements ``a|\psi\rangle + b|\varphi\rangle`` via a block-diagonal direct-sum construction on the virtual bonds, followed by an optional recompression sweep. The result has bond dimension at most ``\chi_\psi + \chi_\varphi`` before truncation.

---

```@docs
AbstractMPSForm
CanonicalForm
VidalForm
ArbitraryForm
FiniteMPS
to_mps
add_mps
```

---

## Canonicalization

Re-gauging an existing MPS — moving the orthogonality centre without rebuilding from the full state tensor — uses carry-propagation sweeps directly on the ``L`` site tensors. The per-bond cost is ``O(\chi^2 d)``, the same as a single site contraction, making a full re-gauge of the chain ``O(L\chi^2 d)``.

The four `CanonicalizeConfig` subtypes cover the standard gauge choices:

| Config | Result form | Centre |
|---|---|---|
| `LeftCanonical` | `CanonicalForm(L, L+1)` | right boundary |
| `RightCanonical` | `CanonicalForm(0, 1)` | left boundary |
| `BondCanonical(k)` | `CanonicalForm(k, k+1)` | bond ``k \leftrightarrow k+1`` |
| `SiteCanonical(k)` | `CanonicalForm(k-1, k+1)` | site ``k`` |

`canonical_error` measures how far a site tensor deviates from left-isometry: ``\|A^\dagger A - I\|_F``. This is the MPS "health metric" — values near machine epsilon confirm the canonical form is exact; values growing over a long TEBD run signal gauge drift.

---

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

---

## Vidal (Γ–Λ) form

The Vidal representation stores the MPS in a manifestly symmetric gauge. Each bond carries an explicit diagonal matrix ``\Lambda_i`` of Schmidt values, and each site stores a ``\Gamma_i`` tensor related to the left-canonical tensor ``A_i`` by ``\Gamma_i = \Lambda_{i-1}^{-1} A_i``. The state is recovered as

```math
|\psi\rangle = \sum_{\sigma_1,\ldots,\sigma_L} \Lambda_0 \,\Gamma_1^{\sigma_1}\, \Lambda_1 \,\Gamma_2^{\sigma_2}\, \Lambda_2 \cdots \Gamma_L^{\sigma_L}\, \Lambda_L
```

This gauge is symmetric under reflection of the chain and makes the Schmidt values at every bond immediately accessible without any contraction. It is the natural form for TEBD (time-evolving block decimation): after a two-site gate is applied and the bond is re-SVD'd, updating ``\Lambda_i`` and the two neighbouring ``\Gamma`` tensors is a local operation that does not require re-gauging the entire chain.

Zero entries in ``\Lambda_i`` (reduced effective bond dimension) are handled by clamping the inversion, preventing division blow-up.

---

```@docs
to_vidal
to_canonical
```

---

## Observables & correlators

For a left-canonical MPS, expectation values and overlaps are computed by a single left-to-right environment contraction. The running environment ``E_i`` (shape ``\chi \times \chi``) is updated site by site:

```math
E_i = \sum_{\sigma,\sigma'} W_{\sigma\sigma'} \, A_i^\dagger[\cdot,\sigma,\cdot] \; E_{i-1} \; A_i[\cdot,\sigma',\cdot]
```

where ``W = O`` at the operator site(s) and ``W = I`` elsewhere. For a left-canonical MPS, ``A_i^\dagger E_{i-1} A_i = I`` at every site to the left of the operator — the environments collapse and only the right half needs explicit contraction.

`two_point` reuses exactly the same loop as `local_expectation` with two operator insertions — no extra contraction steps — making ``\langle O_i O_j \rangle`` the same cost as ``\langle O_i \rangle``.

---

```@docs
overlap
local_expectation
two_point
```
