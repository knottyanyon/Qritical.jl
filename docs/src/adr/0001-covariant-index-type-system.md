# ADR 0001: Covariant Index Type System

## Status

Accepted

## Why this exists

I'm taking "Hands on Tensor Networks" by Prof. Dante Kennes at RWTH Aachen
(SoSe 2026), with the Ulrich Schollwöck review paper (*The density-matrix
renormalization group in the age of matrix product states*) as assigned
reading. To supplement my understanding I've also been working through the
lecture notes from Prof. Jan von Delft's course at LMU Munich
(https://www2.physik.uni-muenchen.de/lehre/vorlesungen/sose_24/tensor_networks_25),
which introduces Penrose graphical notation alongside explicit index algebra.
I've been doing hand derivations in both styles — diagrams and index
expressions — to make sure I actually understand what's going on.

The problem I kept running into was that when I translated those hand
derivations into code, nothing stopped me from accidentally contracting the
wrong pair of indices. If I have `σ_1` and `σ_2` (physical legs at different
sites) with the same dimension, a plain array contraction won't complain.
On paper I'd never make that mistake because the labels are explicit. I
wanted the code to be as strict as the paper.

Covariant index notation (up/down indices, Einstein summation where you always
contract one up with one down) is not standard in quantum computing or quantum
mechanics the way it is in general relativity. Most tensor network papers,
including Schollwöck's review, just use named indices without worrying about
variance. I chose it anyway because:

1. Von Delft's notes use it, and that's what my hand derivations follow.
2. It gives a mechanical rule for what constitutes a valid contraction that
   the computer can check — up must pair with down.
3. When symmetries enter the picture later (which Schollwöck's review doesn't
   cover carefully), covariant notation maps naturally onto the
   domain/codomain language that TensorKit uses. I wanted the foundation
   to be right from the start even if I'm not using it for symmetries yet.

My tutor introduced me to TensorOperations.jl and the `@tensor` macro. The
`@tensor C[s,b] := A[s,a] * B[a,b]` syntax was immediately readable — it
looked exactly like what I was writing in my notebook. That locked in the
approach.

## The design decisions

### `IndexDirection` enum

```julia
@enum IndexDirection UpIndex DownIndex
```

`UpIndex` means the index is a superscript (upper index, contravariant).
`DownIndex` means subscript (lower index, covariant). A valid contraction
always pairs one `UpIndex` leg with one `DownIndex` leg — same rule as in GR.

The main inspiration here is xTensor (part of the xAct package for
Mathematica, by J. M. Martín-García), which encodes this as a sign (+1/−1)
on abstract index objects. I looked at its source to understand how to
implement the pairing check.

### `PhysicalIndex` — the local Hilbert space leg

```julia
struct PhysicalIndex <: AbstractIndex
    label::Symbol
    site::AbstractSite    # knows the lattice position and local dimension
    dir::IndexDirection
end
```

Physical indices are the ones that touch the local Hilbert space at a site —
the `σ_i` in diagrams. The key thing is that `σ_1` and `σ_2` are different
indices even if they have the same dimension. Storing the full `site` object
(which carries `lattice_ordinal`) means the index knows which site it belongs
to. A bonus: `site.space` is already a `TensorKit.ElementarySpace`, which
will matter if I ever want to switch to a TensorKit backend (see ADR 0002).

### `BondIndex` — the entanglement/virtual leg

```julia
struct BondIndex <: AbstractIndex
    label::Symbol
    from::Int     # ordinal of the site the arrow leaves
    to::Int       # ordinal of the site the arrow arrives at
    dim::Int
    dir::IndexDirection
end
```

Bond indices carry the entanglement between two tensors in the network —
the `α` in an MPS diagram. In a network diagram the bond has an arrow, and
that arrow has a direction. The physical meaning is: `from` is the site the
arrow leaves (that tensor's slot is `DownIndex`, outgoing), `to` is the site
the arrow enters (that tensor's slot is `UpIndex`, incoming).

I originally stored just a single `bond_site::Int` but that can't distinguish
a `2→3` bond from a `3→2` bond on the same pair of sites. Switching to
`from`/`to` was suggested by looking at how ITensor.jl tags its Index objects
with site numbers — they use integer tags to distinguish copies of what would
otherwise be identical indices.

### `isdual` — what makes two indices a valid contraction pair

Inspired by xTensor's `PairQ[a, -a]`. Two indices can be contracted if and
only if they are "dual": same label, same location, same dimension, opposite
direction. The implementation:

```julia
isdual(i::BondIndex, j::BondIndex) =
    i.dir != j.dir && i.label == j.label &&
    i.from == j.from && i.to == j.to && i.dim == j.dim

function isdual(i::PhysicalIndex, j::PhysicalIndex)
    i.dir != j.dir && i.label == j.label && i.site === j.site
end

isdual(::AbstractIndex, ::AbstractIndex) = false  # cross-kind is never valid
```

The `===` for sites uses Julia's reference/value equality on immutable structs.
Two separately constructed `SpinSite(half(1), 1)` are `===` in Julia because
the struct is fully immutable — bitwise comparison gives value equality. This
means if I write the same site inline in two different index expressions, it
still works. But in practice I store sites in a lattice array and reuse the
same object, which is the cleaner pattern anyway.

## What this enables

After this, writing a contraction looks like:

```julia
σ = PhysicalIndex(:σ, sites[2], UpIndex)
αL = BondIndex(:α, 1, 2, D, DownIndex)   # bond 1→2, outgoing from site 2
αR = BondIndex(:β, 2, 3, D, UpIndex)     # bond 2→3, incoming to site 2

A = IndexedTensor(rand(2, D, D), (σ, αL, αR))
```

and the `@tensor` macro will catch if I accidentally try to contract `αL`
with a bond index from a different bond, or contract two indices in the same
direction.

## References

- Von Delft tensor networks course (LMU Munich, SoSe 2024/25):
  https://www2.physik.uni-muenchen.de/lehre/vorlesungen/sose_24/tensor_networks_25
- Schollwöck, *The density-matrix renormalization group in the age of matrix
  product states*, Annals of Physics 326 (2011) 96–192.
- xAct/xTensor (J. M. Martín-García): https://www.xact.es/xTensor/index.html
  — `UpIndex`/`DownIndex` sign encoding, `PairQ`, `EIndexQ`, `delta`
- TensorOperations.jl `tensorstructure`/`checkcontractible` interface:
  https://jutho.github.io/TensorOperations.jl/stable/
- ITensor.jl Index and tag system (inspiration for site/bond enumeration):
  https://itensor.github.io/ITensors.jl/stable/
- TensorKit.jl domain/codomain approach (the structural alternative I
  considered and eventually want to bridge to):
  https://jutho.github.io/TensorKit.jl/stable/
