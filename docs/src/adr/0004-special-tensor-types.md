# ADR 0004: Special Tensor Types for Structured Factors

## Status

Accepted (implementation deferred to PLAN 3)

## Why this exists

When you decompose a tensor — SVD, QR, polar decomposition — the factors come
back with mathematical guarantees that the original tensor didn't have. After
an SVD of a site tensor `A`, you get back `U`, `S`, and `V†` where `U` and
`V†` are isometries. The standard MPS canonical form algorithms (Exercises 2–4
in the course) rely on this: turning a chain left- or right-canonical gives you
a sequence of isometries, and the normalisation of the state follows from that
structure.

The question is: should the code know about this?

If everything stays as plain `IndexedTensor`, the structure is invisible to the
type system. You have to remember by convention which tensors are isometric,
and any sanity checks you write operate on raw arrays without context. The
alternative is to let the algorithm that produces `U` *annotate* it as an
`Isometry` at the point of creation, so that downstream code and tests can
reason about it explicitly.

## The design question: what does a type annotation mean here?

The natural instinct (coming from Python) is to make `Isometry` a subclass of
`IndexedTensor`. Julia doesn't allow this — **concrete types cannot be
subtyped**. Only `abstract type` declarations can form hierarchies. So Python
inheritance patterns don't transfer directly.

The three real options in Julia were:

**Option A — Abstract supertype + sibling concretes**
Introduce `AbstractIndexedTensor` and make `IndexedTensor`, `Isometry`,
`Unitary` all subtypes of it. Full dispatch power, but requires restructuring
the existing code.

**Option B — Wrapper structs (chosen)**
`Isometry` holds an `IndexedTensor` plus metadata. Nothing about the existing
code changes. The producing algorithm wraps its output; the type is a claim
made at the call site.

**Option C — Tag type parameter**
Add a `Structure` parameter to `IndexedTensor` (e.g. `:general`, `:isometric`).
One struct, dispatch via the tag. Feels clever but adds a fourth parameter to
an already three-parameter struct and makes generic code noisier.

Option B was chosen because it leaves `IndexedTensor` untouched, makes the
semantics explicit at the wrapping site, and maps naturally to how Julia's own
`LinearAlgebra` handles structured matrices (`Symmetric(A)`, `Diagonal(v)` —
claims made by the caller, not verified at construction).

## The type is a claim, not a guarantee

This is the most important design decision in this ADR.

`Isometry` means: *the algorithm that produced this tensor guarantees it was
isometric when it was created.* It says nothing about what happens afterward.
Floating point drift over many iterations, bugs in upstream code, further
operations on the tensor — any of these can erode isometry without changing
the type. The type is a semantic annotation, not a runtime invariant that the
system polices.

Verifying isometry at construction (`norm(A†A - I) < ε`) was considered and
rejected for two reasons:

1. **Floating point**: the tolerance is arbitrary and context-dependent. A
   tolerance that is fine for double precision may be too tight after 1000
   DMRG sweeps.
2. **Performance**: canonical form algorithms run normalisation checks as
   optional sanity steps, never after every operation. Enforcing at
   construction collapses that distinction.

## The `check_isometry` design

Instead of a Bool, `check_isometry` returns both a pass/fail and the error
magnitude:

```julia
check_isometry(iso::Isometry; atol=1e-12) → (passed::Bool, err::Float64)
```

The magnitude matters:
- `err ≈ 1e-15` — within floating point noise, fine
- `err ≈ 1e-6` — accumulated drift, worth logging
- `err ≈ 1e-2` — something is wrong upstream

In tests you assert `passed`. In debugging you print `err` to see how far the
deviation has grown. A failed check on an `Isometry` object is not a type
error — it is a signal. What you do with it (ignore, warn, re-orthogonalise)
is a decision for the algorithm, not the type system.

## The `Bisection` connection

Isometry is not a property of a tensor in isolation — it is a property with
respect to a *leg grouping*. `A†A = I` only makes sense once you specify which
legs you contract over to form the square matrix whose isometry you are
checking. `Bisection` (already in the codebase) is exactly the right object
for this:

```julia
struct Isometry{Element, Order, D<:AbstractArray{Element,Order}}
    tensor::IndexedTensor{Element,Order,D}
    legs::Bisection    # left legs form the isometric side: A†A = I
end
```

Left isometry (`A†A = I`, left bond contracted) vs right isometry
(`AA† = I`, right bond contracted) are encoded via the `legs` field.
`Unitary` is the square case where both hold:

```julia
struct Unitary{Element, Order, D<:AbstractArray{Element,Order}}
    tensor::IndexedTensor{Element,Order,D}
    # square matrix: both A†A = I and AA† = I
end
```

## What the SVD return type looks like

```julia
function factorize_with_svd(tensor::IndexedTensor, bisection::Bisection)
    # ... perform SVD ...
    return Isometry(U_tensor, bisection),
           S_tensor,
           Isometry(VDag_tensor, dual_bisection)
end
```

The SVD is the one that *knows* the output is isometric. It wraps the result.
The caller gets typed objects without having to make any assertion themselves.

## What is deferred

- Full implementation: `src/special_tensors.jl`, tests, `@tensor` delegation
- Other special types: `Diagonal` (may just use `LinearAlgebra.Diagonal` as
  the backing store `D`), `Projector`, `BasisState`
- Re-orthogonalisation helpers (polar decomposition, QR) for when drift becomes
  too large — these are algorithm-level concerns, not type-level
- Sector-aware isometry once `BondIndex` gains symmetry information (see ADR 0003)

## References

- `src/tensor_core.jl` — `IndexedTensor` and `Bisection` definitions
- `src/schmidt_decomposition.jl` — existing SVD code to be updated
- PLAN 3 — implementation plan for this ADR
- Issue #61 — implementation tracking
- Julia `LinearAlgebra` structured matrix wrappers (prior art):
  https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#Special-matrices
