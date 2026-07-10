# Visualisation

Qritical's plotting lives in a **package extension**, `QriticalMakieExt`, which
loads automatically the moment a [Makie](https://docs.makie.org) backend is
imported. Nothing here is available until you do so:

```julia
using Qritical
using CairoMakie   # or GLMakie / WGLMakie — this activates QriticalMakieExt
```

Keeping the drawing code in an extension means `Qritical` itself has no hard
dependency on Makie: users who only run algorithms never pay the plotting
package's load cost.

There are two entry points, mirroring the split in
[quimb](https://quimb.readthedocs.io/en/latest/examples/schematic-demo.html):

- [`draw`](@ref) — the **automatic** path. Hand it a `QTensor` or a `FiniteMPS`
  and it lays the diagram out for you. See
  [GS-1](../getting_started/gs1_tensors_and_indices.md).
- [`schematic`](@ref) — the **manual** path. A small drawing DSL for building a
  bespoke tensor-network diagram node by node, including translucent partition
  regions. See [GS-3](../getting_started/gs3_drawing_tensor_networks.md).

## Automatic diagrams

```@docs
draw
```

## Schematic drawing DSL

Build a diagram by threading a canvas from [`schematic`](@ref) through the
drawing verbs, then display the canvas's `.fig`:

```@docs
schematic
node!
wire!
stub!
region!
note!
```
