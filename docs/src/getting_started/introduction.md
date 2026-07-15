# Introduction: Legs, Indices, and the Design Idea

When you sketch a tensor network on paper, each line leaving a node has a *direction* and a *role*. In a matrix product state the physical leg points upward, labelling the local Hilbert-space basis. The left bond flows *into* the site tensor; the right bond flows *out*. These directions are not decoration — they determine which index pairs are allowed to contract, and they are the hook on which block-sparse symmetry sectors are later attached.

Opening a Julia file and staring at `rand(2, 4, 4)` breaks that correspondence completely: the array has three slots but no record of which slot is the physical leg or which direction each bond points.

## From leg to index

In Qritical every leg of a tensor is represented by an **index** — a Julia value that bundles three pieces of information:

| What | Why it matters |
|:-----|:---------------|
| A **label** (`:σ`, `:αL`) | Identifies which leg this is; contraction matches by label |
| A **dimension** | Number of basis states the leg can take |
| A **variance** (`Upper` or `Lower`) | Records whether the leg is contravariant (ket / row) or covariant (bra / column) |

*Upper* corresponds to a superscript in Einstein notation — the primal vector space ``V``, whose basis vectors transform like kets ``|\psi\rangle``. *Lower* corresponds to a subscript — the dual space ``V^*``, whose basis covectors transform like bras ``\langle\psi|``. A valid Einstein contraction pairs exactly one `Upper` with one `Lower` index carrying the same label.

## The MPS leg convention

In Qritical the settled convention for a finite MPS site tensor is:

| Leg | Variance | Rationale |
|:----|:---------|:----------|
| Physical ``\sigma`` | `Lower` (covariant) | The local Hilbert-space basis is a row index — the site tensor is a map *from* spin states |
| Left virtual bond ``\alpha_L`` | `Upper` (contravariant) | Flows into the site from the left boundary |
| Right virtual bond ``\alpha_R`` | `Lower` (covariant) | Flows into the next site, i.e. out of this one |

This is a *design choice*, not an accident. It ensures that bra–ket expectation values ``\langle\psi|\hat{O}|\psi\rangle`` automatically pair indices without any manual tracking of which legs are "input" and which are "output". It also future-proofs the code for U(1) charge conservation: an upper index carries charge ``+q`` and its lower partner carries ``-q``, so conservation falls out of contraction automatically.

## What comes next

The [Indices and QTensor](indexed_tensor.md) page introduces the concrete types — `TIx`, `QTensor`, `Partition`, `Bipartition` — that implement this design in code.
