# ADR 0003: Explicit Arrow Orientation in `BondIndex` via `from`/`to`

## Status

Accepted

## Why this exists

Exercises 2, 3, and 4 from Prof. Kennes' "Hands on Tensor Networks" course
(RWTH Aachen, SoSe 2026) work through left normalization, right normalization,
and left/right/mixed canonical forms of an MPS. Doing these exercises by hand, the arrow on each bond in the
diagram is not decoration — it tells you which direction you're sweeping the
QR or SVD decomposition, and it determines which tensor satisfies which
orthogonality condition.

Concretely:

- **Left-normalized tensor** at site `i`: the bond arrow points right
  (`i → i+1`). The tensor satisfies `A†A = I` when you view it as a matrix
  that maps `(σ_i, α_{i-1})` on the left to `α_i` on the right.
- **Right-normalized tensor** at site `i`: the bond arrow points left
  (`i ← i+1`, or equivalently the arrow on the right bond enters site `i`).
  The tensor satisfies `AA† = I` when viewed as a matrix in the other
  direction.
- **Mixed canonical / orthogonality center**: left-normalized tensors to the
  left, right-normalized to the right. The arrows in the diagram all point
  away from the orthogonality center site.

If a bond index doesn't know which direction its arrow points, the code has
no way to check whether a given tensor is in the right canonical form for
a particular step of the sweep algorithm. I'd have to track that information
separately, which defeats the point of having rich index objects in the first
place.

## The original design and its problem

The first version of `BondIndex` had a single integer field called
`bond_site`:

```julia
struct BondIndex <: AbstractIndex
    label::Symbol
    bond_site::Int    # which site does this bond "belong to"?
    dim::Int
    dir::IndexDirection
end
```

This doesn't work for canonical forms because `bond_site = 2` can't tell
you whether the bond arrow runs `2 → 3` or `3 → 2`. Those two bonds are
physically different: on a left-normalized MPS the bond between sites 2 and 3
has an arrow pointing right, on a right-normalized MPS it points left. They
need to be distinguishable.

## The decision: `from` and `to`

```julia
struct BondIndex <: AbstractIndex
    label::Symbol
    from::Int     # ordinal of the site the arrow leaves
    to::Int       # ordinal of the site the arrow arrives at
    dim::Int
    dir::IndexDirection
end
```

The arrow direction is now encoded in the struct itself. The convention
matches the hand-diagram convention from the lectures:

| The tensor sits at | `dir` on that tensor's slot | What it means in the diagram   |
|:-------------------|:---------------------------|:-------------------------------|
| `from` site        | `DownIndex`                | arrow leaves this tensor (outgoing, subscript) |
| `to` site          | `UpIndex`                  | arrow enters this tensor (incoming, superscript) |

For a left-normalized MPS with bond `α` between sites 2 and 3:
- Site 2 holds `BondIndex(:α, 2, 3, D, DownIndex)` — arrow leaves to the right
- Site 3 holds `BondIndex(:α, 2, 3, D, UpIndex)` — arrow arrives from the left

Flipping to right-normalized would mean reversing the arrow:
- Site 2 holds `BondIndex(:α, 3, 2, D, UpIndex)` — arrow now arrives from the right
- Site 3 holds `BondIndex(:α, 3, 2, D, DownIndex)` — arrow now leaves to the left

This makes the canonical form of a tensor readable directly from its index
objects, without any external bookkeeping.

## How `isdual` works with this

Two bond index slots are a valid contraction pair (`isdual`) only if they
agree on `from`, `to`, `label`, and `dim`, and carry opposite directions.
This means you can't accidentally contract a left-bond slot with a right-bond
slot from a different bond just because they have the same dimension — they
have to name the same `from` and `to` sites.

```julia
isdual(i::BondIndex, j::BondIndex) =
    i.dir != j.dir && i.label == j.label &&
    i.from == j.from && i.to == j.to && i.dim == j.dim
```

## What this enables for canonical forms

Once the tensors carry this information, a future `is_left_normalized` or
`is_right_normalized` predicate can inspect the bond indices directly to
verify the expected arrow orientations, and a canonicalization sweep function
can construct the `BondIndex` objects for each site automatically with the
correct `from`/`to` fields based on which direction the sweep is running.

## References

- Von Delft tensor networks course, Exercises 2–4 (left/right normalization,
  canonical forms of MPS):
  https://www2.physik.uni-muenchen.de/lehre/vorlesungen/sose_24/tensor_networks_25
- Schollwöck, *The density-matrix renormalization group in the age of matrix
  product states*, Section 4 (MPS canonical forms and gauge freedom)
- ITensor.jl uses integer site tags on Index objects to distinguish bonds at
  different positions; arrow direction is tracked separately via `dir` on
  ITensor's Index. The `from`/`to` approach encodes both in one place.
