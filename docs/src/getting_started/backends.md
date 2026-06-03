# Backend Modes (deferred)

The long-term design calls for two backends, selectable via a scoped context:

- **`:native`** — plain Julia `Array`. Default. Easy to inspect, debug, and profile.
- **`:tensorkit`** — [`TensorKit.TensorMap`](https://jutho.github.io/TensorKit.jl/stable/man/tensors/). Block-sparse, symmetry-aware storage. Dramatically faster at large bond dimensions when the Hamiltonian has conserved quantum numbers.

The same algorithm code is intended to work in both modes — only the backing store of `IndexedTensor` changes. The switch would look like:

```julia
with_backend(:tensorkit) do
    # contractions use TensorKit block-sparse tensors here
end
```

**Current status:** `:native` is the only active backend. `with_backend` and `current_backend` are not yet implemented — the `backend.jl` source file and `BondIndex` sector information needed to construct symmetry-aware TensorKit spaces are deferred to a future milestone. The plan is in `CORE_DESIGN_JL.md` (in `.claude/design_plans/`).
