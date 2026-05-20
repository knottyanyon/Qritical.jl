# ADR 0003: Backend Dispatch via Scoped Context

## Status

Accepted (implementation in progress)

## Why this exists

The immediate goal of this package is learning — I want to be able to write
tensor network contractions in a readable way that matches my hand derivations
and check that the index algebra is correct. For that, plain Julia arrays are
fine and keeping things simple matters more than speed.

The longer-term picture is less clear. Implementing symmetries (U(1), SU(2))
is not a goal of the course — Prof. Kennes was upfront that it's a lot of
work to do properly — and I'm not planning to go there right now. But I also
don't want to close the door. A spin chain with U(1) symmetry (conserved
total Sz) has a block-sparse tensor structure that makes contractions
dramatically cheaper at large bond dimensions. The Schollwöck review touches
on this, and it's exactly what the QuantumKitHub ecosystem is built around.

The key observation is that `TensorKit.jl` — introduced indirectly by our
tutor via `TensorOperations.jl`, which wraps it — already has all of this
symmetry-aware, block-sparse machinery built in. It would be a shame to write
code now that can never use it. So the goal is: keep the `:native` (plain
array) path working exactly as it does today, and leave a clearly-labelled
`:tensorkit` door that a future version of this package can walk through
without breaking anything.

The tension is: I don't want to throw away the easy-to-understand `@tensor`
Einstein-notation code I've been writing. I want the same algorithm code to
work in both modes, just with a different backing tensor type underneath.

This is the same problem PennyLane solves for quantum circuits — you write
your circuit once and choose whether to run it on NumPy, PyTorch, or an actual
quantum device. The circuit description doesn't change; the execution backend
does.

## The options I considered

**Global flag** (`set_backend!(:tensorkit)` at the top of a file): simple,
matches the PennyLane mental model, but not safe if I ever want to run two
backends in the same session — say, benchmarking one against the other.

**Scoped context** (`with_backend(:tensorkit) do ... end`): the backend
choice is explicit and local. Two independent code blocks can use different
backends without interfering. This is how JAX handles device placement with
`with jax.default_device(...)`. Julia has a built-in mechanism for this called
`ScopedValues` (standard library since 1.11, available as a package for
earlier versions) — it's also how Julia's own IO system threads contextual
settings through call stacks without explicit parameter passing.

**Compile-time preference** (Preferences.jl, stored in
`LocalPreferences.toml`): zero runtime overhead because Julia specialises
everything at compile time, but you can't switch in a running session. That
rules out any in-session comparison of the two modes.

## The decision: scoped context

```julia
using ScopedValues

const _BACKEND = ScopedValue{Symbol}(:native)

with_backend(f, b::Symbol) = with(_BACKEND => b, f)
current_backend()          = _BACKEND[]
```

Usage looks like:

```julia
# Default: plain dense arrays, readable @tensor notation
A = IndexedTensor(rand(2, 4, 4), (σ, αL, αR))

# Opt in to production backend for a block of code
with_backend(:tensorkit) do
    A = IndexedTensor(rand(2, 4, 4), (σ, αL, αR))  # TensorKit-backed
    @tensor C[s,g] := A[s,a] * B[a,g]              # same syntax, faster execution
end
```

The `:native` default means existing code never breaks. Moving to
production is always an explicit opt-in.

## How `IndexedTensor` plugs into this

`IndexedTensor` is made parametric over its backing store type:

```julia
struct IndexedTensor{T, N, D<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::D
    indices::NTuple{N, AbstractIndex}
end
```

The constructor reads `current_backend()` and picks the right backing type.
In `:native` mode `D = Array{T,N}`. In `:tensorkit` mode `D` becomes
whatever the production backing type is — initially likely
`TensorKit.TensorMap`, but this is left open deliberately (see below).

The bridge between the two worlds is already partly there: `PhysicalIndex`
stores `site::AbstractSite`, and each site already carries a
`TensorKit.ElementarySpace`. So the conversion from index metadata to TensorKit
space is:

```julia
tensorkit_space(i::PhysicalIndex) = i.site.space
tensorkit_space(i::BondIndex)     = TensorKit.ComplexSpace(i.dim)
```

Bond indices use plain `ComplexSpace` for now because `BondIndex` doesn't yet
carry sector information (which quantum numbers live at which bond). Adding
sector information to `BondIndex` is a future change that will unlock the
actual block-sparse speedup for entanglement legs.

## What's still deferred

The concrete production backing type is intentionally not committed to in this
ADR. TensorKit.TensorMap is the obvious candidate, but making `@tensor` work
with a TensorKit-backed `IndexedTensor` requires implementing the full
TensorOperations.jl contraction interface (`tensorcontract!`, `tensortrace!`,
`tensoradd!`) as delegation to TensorKit's internals. That's a real
implementation effort and should get its own ADR when I actually do it.

The sector information on `BondIndex` is also deferred. Adding it is a
breaking change to the struct (it will change how `isdual` works for bonds
with symmetry), and it only makes sense once I have a concrete physical
system I want to simulate with a specific symmetry group.

## References

- JAX default device context:
  https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
- PennyLane device model:
  https://docs.pennylane.ai/en/stable/introduction/circuits.html
- Julia `ScopedValues` (stdlib, 1.11+):
  https://docs.julialang.org/en/v1/base/scopedvalues/
- Julia `IOContext` — prior art for scoped ambient context in Julia:
  https://docs.julialang.org/en/v1/base/io-network/#Base.IOContext
- Flux.jl `gpu()`/`cpu()` transfer (parametric model/array over backend type):
  https://fluxml.ai/Flux.jl/stable/guide/gpu/
- TensorKit.jl block-sparse TensorMap (likely production backing):
  https://jutho.github.io/TensorKit.jl/stable/man/tensors/
- MPSKit.jl and PEPSKit.jl (QuantumKitHub ecosystem) — these go all-in on
  the TensorKit structural approach and never use `@tensor` for inner products;
  they confirm there is no Julia ecosystem precedent for the hybrid approach
  this ADR is trying to build:
  https://github.com/QuantumKitHub/MPSKit.jl
