# MPO & Expectation Values

A **[`Matrix Product Operator`](../references/glossary.md#matrix-product-operator)** ([`MPO`](../references/glossary.md#mpo)) is the operator analogue of an [`MPS`](../references/glossary.md#mps): a
``d^L \times d^L`` operator factored into a product of ``L`` local rank-4 tensors

```math
O = \sum_{\{\sigma\},\{\sigma'\}} W^{\sigma_1}_{\sigma_1'} W^{\sigma_2}_{\sigma_2'} \cdots
    W^{\sigma_L}_{\sigma_L'} \; |\sigma_1 \cdots \sigma_L\rangle \langle \sigma_1' \cdots \sigma_L'|
```

where each ``W^{\sigma_i}_{\sigma_i'}`` is a ``\chi_{i-1} \times \chi_i`` matrix of
``d \times d`` operator blocks. For nearest-neighbour Hamiltonians with ``K`` distinct
bond operator types the optimal [`bond dimension`](../references/glossary.md#bond-dimension) is ``\chi = 2 + K``.

### The finite-state-machine (FSM) construction

The W-matrix at each site encodes a finite automaton with three kinds of states:

| Index | State | Meaning |
|---|---|---|
| `1` | "done" | All terms left of this site have been accumulated |
| `2 … K+1` | channel ``k`` | Bond type ``k`` was opened at some site ``i' < i``; waiting to close |
| `K+2` | "start" | No terms have been accumulated yet |

Transitions:
- **start → done:** an on-site term ``h \cdot O_i`` at the current site
- **start → channel ``k``:** the left operator ``J_k A_k`` of a bond starting here
- **channel ``k`` → done:** the right operator ``B_k`` closing a bond that opened earlier

The left boundary picks only the "start" row; the right boundary picks only the "done"
column. This is why the boundary tensors have bond dimension 1.

### Expectation value

``\langle \psi | O | \psi \rangle`` is computed by sweeping a left environment tensor
``L_{[α, a, β]}`` (shape ``\chi_\text{bra} \times \chi_\text{mpo} \times \chi_\text{ket}``)
from site 1 to ``L``:

```math
L^{(i)}[\alpha', a', \beta'] = \sum_{\alpha, a, \beta, \sigma, \sigma'}
    L^{(i-1)}[\alpha, a, \beta]\,
    \overline{A_i[\alpha, \sigma, \alpha']}\,
    W_i[a, \sigma, \sigma', a']\,
    A_i[\beta, \sigma', \beta']
```

The result ``L^{(L)}[1,1,1]`` is the scalar expectation value. Per-site cost:
``O(\chi^3 d \chi_\text{mpo})``.

---

## Quick Reference

**MPO types:** [`FiniteMPO`](@ref) · [`MPO`](@ref)

**Functions:** [`expect`](@ref) · [`apply_mpo`](@ref)

---

## Types

```@docs
FiniteMPO
MPO
```

## Functions

```@docs
expect(ψ::FiniteMPS, O::FiniteMPO)
apply_mpo
```
