module Qritical

using LinearAlgebra
using TensorOperations
import SparseArrays
import SparseArrays: sparse

"""
    AbstractGeometry

Abstract supertype for lattice geometries.

A geometry answers exactly two queries: which sites exist, and which pairs of
sites are connected by bonds.  That is all the Hamiltonian builder needs — it
does not care whether the underlying graph is a chain, a square lattice, or a
torus.  Concrete subtypes only need to implement [`sites`](@ref) and
[`bonds`](@ref).

Current concrete geometry: [`Chain`](@ref) (1D open/periodic chain).
Planned extensions: `Square`, `Torus`, `Lattice{V,E}` (see §2 of the design plan).

See also: [`Chain`](@ref), [`sites`](@ref), [`bonds`](@ref)
"""
abstract type AbstractGeometry end

include("min_model_kit/lattice/geometry.jl")
include("min_model_kit/dof.jl")
include("min_model_kit/lattice/symmetries.jl")
include("tensors/storage_format.jl")
include("operators/operator.jl")
include("tensors/tix.jl")
include("tensors/multix.jl")
include("tensors/partition.jl")
include("tensors/qtensor.jl")
include("tensors/spectrum.jl")
include("tensors/svd.jl")
include("utils/io.jl")
include("states/mps.jl")
include("states/canonicalize.jl")
include("states/vidal.jl")
include("operators/correlators.jl")
include("operators/finite_mpo.jl")
include("algorithms/power_method.jl")
include("algorithms/tebd.jl")
include("algorithms/quench.jl")
include("algorithms/ed.jl")
include("studies/disorder.jl")

# ==== Index layer =============================================================
export IxLoc, Upper, Lower
export AbstractIx, TIx, MulTIx
export dim, label, which_space, flip
export upper, lower, uppers, lowers, uppers_range, lowers_range, bond_label

# ==== QTensor + partitions ====================================================
export QTensor, dagger
export Partition, Bipartition, complement, bipartition, group_legs

# ==== SVD + truncation ========================================================
export AbstractTrunc, NoTrunc, MaxBondDimTrunc, ValCutoffTrunc
export FullSVD, ReducedSVD, do_svd

# ==== Spectrum + orthogonality centre =========================================
export Bond, OrthoCenter, BondCenter, SiteCenter
export AbstractSpectrum, SingValSpectrum, EigValSpectrum, SchmidtSpectrum
export schmidt_rank, spectral_gap, schmidt_values
export entanglement_entropy, entanglement_spectrum

# ==== State utilities + I/O ===================================================
export bipartition_matrix, as_state, load_array

# ==== MPS & canonical forms ==================================================
export AbstractMPSForm, CanonicalForm, VidalForm, ArbitraryForm
export FiniteMPS, to_mps, add_mps
export CanonicalizeConfig, LeftCanonical, RightCanonical, BondCanonical, SiteCanonical
export canonicalize, canonical_error, is_canonical, overlap, local_expectation, two_site_op
export to_vidal, to_canonical

# ==== Geometry ================================================================
export AbstractGeometry, Chain, sites, bonds

# ==== DoF layer ===============================================================
export AbstractDoF
export Spin, SpinHalf, SpinOne
export SpinlessFermion, Electron, MajoranaFermion, HardCoreBoson
export CanonicalRelation, CCR, CAR
export local_dim, canonical_relation, algebra_generators
# ==== Symmetry tags ===========================================================
export NoSymmetry, physical_space

# ==== LatticeOperator / Hamiltonian ==================================================
export uniform_coupling
export OneSiteTerm, TwoSiteTerm, LatticeOperator, Hamiltonian
export XXZ, Heisenberg, Ising
export total_magnetization, staggered_magnetization, op_at_site, two_site_op
export identity_operator
# ==== Storage format tags =====================================================
export StorageFormat, DenseFormat, SparseFormat
export matrix_repr

# ==== MPO + expect ============================================================
export FiniteMPO, MPO
export expect, apply_mpo

# ==== Power Method ============================================================
export PowerMethodResult, power_method

# ==== TEBD + time evolution ===================================================
export TimeAxis, RealTime, ImaginaryTime
export Unitary, HermitianPSD
export Propagator, opclass, gate
export ConstantProtocol, total_time
export bond_hamiltonian
export apply_gate
export TrotterSubstep, SuzukiTrotter, trotter_steps, trotter_step

# ==== Quench + solve interface ================================================
export neel_state
export Quench, TEBD, NoTracker, Tracker
export QuenchResult
export solve

# ==== ExactDiagonalization ====================================================
export GroundState, ExactDiagonalization, EDResult
export StatevectorState, as_statevector, EDTimeResult

# ==== Disorder ================================================================
export Uniform, disorder_realization, parameter_sweep

# ==== Visualisation ==========================================================
# Generic-function stubs; the methods are implemented in `ext/QriticalMakieExt.jl`,
# which loads automatically once a Makie backend (e.g. CairoMakie) is imported.
# The docstrings live here on the stubs so the API reference has a single
# documented source per function.

"""
    draw(A::QTensor;  name="", show_dims=true, show_arrows=true, shape=:general, rotation=0.0, figure_kw=(;))
    draw(mps::FiniteMPS; show_dims=true, show_arrows=true, show_shapes=true, show_legend=true, figure_kw=(;))

Draw an **automatic** tensor-network diagram in the style of quimb. Requires a
Makie backend to be loaded first:

```julia
using Qritical, CairoMakie   # importing CairoMakie activates QriticalMakieExt
draw(A)
```

For a `QTensor`, `A` becomes a single labelled node with one directed leg stub
per index: blue stubs with an inward arrow are `Upper` (incoming / contravariant)
indices, grey stubs with an outward arrow are `Lower` (outgoing / covariant). The
node itself follows the tensor-kind shape convention described on
[`node!`](@ref) via `shape`/`rotation`.

For a `FiniteMPS`, sites are drawn as a horizontal chain coloured by canonical
form (**blue** left-canonical, **red** orthogonality centre, **green**
right-canonical, **grey** arbitrary), with virtual-bond arrows converging on the
centre and physical legs hanging below each site. Left/right-canonical sites are
genuine isometries (``A^\\dagger A = I`` resp. ``BB^\\dagger = I``), so — when
`show_shapes=true` — they are automatically drawn as isometry wedges whose
narrow end points toward the bond they contract onto; the orthogonality centre
(no such guarantee) stays a general circle.

# Keyword arguments
- `name` — label inside the tensor node (`QTensor` only)
- `show_dims::Bool` — annotate bond / leg dimensions (default `true`)
- `show_arrows::Bool` — draw directed arrowheads (default `true`)
- `shape`, `rotation` — node shape/orientation (`QTensor` only); see [`node!`](@ref)
- `show_shapes::Bool` — auto-select isometry wedges by canonical form (`FiniteMPS` only, default `true`)
- `show_legend::Bool` — show the canonical-form colour legend (`FiniteMPS` only, default `true`)
- `figure_kw` — named-tuple forwarded to `Makie.Figure`

See also [`schematic`](@ref) for building diagrams by hand.
"""
function draw end

"""
    schematic(; figure_kw=(;), pad=0.6)

Create an empty **hand-drawing canvas** for bespoke tensor-network diagrams — the
manual counterpart to [`draw`](@ref), in the spirit of quimb's `schematic.Drawing`.
Requires a Makie backend (`using CairoMakie`).

Build a diagram by threading the returned canvas through the drawing verbs
[`node!`](@ref), [`wire!`](@ref), [`stub!`](@ref), [`region!`](@ref) and
[`note!`](@ref), then display its `.fig` field. Everything is drawn in data
coordinates and the view box auto-expands (with margin `pad`) so nothing is
cropped. `figure_kw` is forwarded to `Makie.Figure`.

# Example
```julia
using Qritical, CairoMakie
s = schematic()
node!(s, :A, (0, 0)); node!(s, :B, (1, 0))
wire!(s, :A, :B; label="χ")
region!(s, [:A, :B]; label="block", color=:steelblue)
s.fig
```
"""
function schematic end

"""
    node!(s, name::Symbol, coo; radius=0.22, color, label=name, shape=:general, rotation=0.0, ...)

Place a tensor node at `coo` (a `(x, y)` tuple or `Point2`) on a [`schematic`](@ref)
canvas `s`, with `label` centred inside. The node is remembered under `name` so
later [`wire!`](@ref), [`stub!`](@ref) and [`region!`](@ref) calls can refer
to it by that symbol. Returns `s`.

**Tensor-kind shapes**

`shape` selects the node's geometry; the silhouette encodes the operator's index
structure:

| `shape` | Geometry | Meaning |
|:--------|:---------|:--------|
| `:general`  | circle    | a generic tensor with no special structure |
| `:diagonal` | diamond   | a diagonal matrix (e.g. a singular-value spectrum ``\\Sigma``) |
| `:unitary`  | D-arch    | ``U`` with ``UU^\\dagger = U^\\dagger U = I`` — a flat base with a domed top, centred on the node |
| `:isometry` | trapezoid | ``W`` with ``W^\\dagger W = I`` but ``WW^\\dagger \\neq I`` — the in/out dimensions *differ*, so the wide base carries the larger index and the shape tapers to a narrow top on the *contracted* (smaller) side; stacking two mirrored wedges base-to-base annihilates to the identity |

`rotation` (radians) orients the `:unitary` dome and the `:isometry` narrow top —
`0` points along `+x`, `π` along `-x`, and so on; it has no visible effect on the
symmetric `:general`/`:diagonal` shapes.

# Example
```julia
using Qritical, CairoMakie
s = schematic()
node!(s, :U, (0, 0); shape=:unitary, label="U")                  # half disk, dome up
node!(s, :W, (1, 0); shape=:isometry, rotation=0.0, label="W")   # narrow top → +x
node!(s, :Σ, (2, 0); shape=:diagonal, label="Σ", color=:gray)    # diamond
s.fig
```
"""
function node! end

"""
    wire!(s, a, b; color, linewidth=2.5, label=nothing, arrow=false, shorten=0.0)

Connect `a` and `b` — each a tensor name (`Symbol`) or a coordinate — with a
bond line drawn *behind* the tensor circles on canvas `s`. Optionally annotate it
with `label` (offset perpendicular to the line) and a mid-bond `arrow`. Returns `s`.
"""
function wire! end

"""
    stub!(s, a, dir; length=0.5, label=nothing, arrow=false, into=false, base=false)

Draw an open (dangling) leg — a physical index — from tensor/point `a` in
direction `dir`, given either as an angle in radians or a `(dx, dy)` vector. With
`arrow=true`, `into=true` points the arrowhead toward the tensor (an incoming
`Upper` index); `into=false` points it outward.

When `a` names an `:isometry` or `:unitary` node, `base=true` routes the leg into
that node's flat base with an L-bend — a short stub straight out of the base, then
a turn toward `dir` — so the leg meets the base head-on instead of crossing a
slanted edge (the tensors.net convention for legs on a wedge). Returns `s`.
"""
function stub! end

"""
    region!(s, members; padding=0.4, color, alpha=0.16, label=nothing, ...)

Highlight a group of tensors on canvas `s` with a translucent rounded blob — the
schematic equivalent of a bipartition cut or an entanglement region. `members` is
a collection of tensor names and/or coordinates; the blob is the convex hull of
their positions, inflated by `padding` and rounded at the corners, and is always
drawn *behind* everything else. Overlapping calls with different `color`s show
competing partitions. Returns `s`.
"""
function region! end

"""
    note!(s, coo, text; color=:black, fontsize=12, align=(:center, :center))

Place a free-floating text annotation `text` at `coo` on canvas `s`, expanding
the view box so the note is never cropped. Returns `s`.
"""
function note! end

export draw, schematic, node!, wire!, stub!, region!, note!

end
