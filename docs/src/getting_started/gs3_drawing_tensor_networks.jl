# # GS-3 · Drawing Tensor Networks
#
# The [`draw`](@ref) methods from GS-1 render a `QTensor` or a `FiniteMPS`
# automatically. Sometimes you instead want to *hand-draw* a diagram — a bespoke
# tensor network for a figure, a talk, or a derivation on paper made precise.
# Qritical ships a small schematic-drawing DSL for exactly this, in the spirit of
# [quimb's `schematic` module](https://quimb.readthedocs.io/en/latest/examples/schematic-demo.html).
#
# The whole API is five verbs threaded through a canvas:
#
# | Verb | Draws |
# |------|-------|
# | [`tensor!`](@ref) | a named tensor node (a labelled circle) |
# | [`bond!`](@ref) | a bond line between two tensors |
# | [`leg!`](@ref) | a dangling / open (physical) leg |
# | [`partition!`](@ref) | a translucent blob grouping tensors |
# | [`note!`](@ref) | a free-floating text annotation |
#
# Everything is drawn in **data coordinates**, so shapes stay proportional, and
# the canvas auto-expands its view box as you add elements — nothing gets cropped.
#
# ---

using Qritical
using CairoMakie   # activates QriticalMakieExt and enables the drawing verbs

# ## 1. A canvas, a tensor, a bond
#
# [`schematic`](@ref) returns an empty canvas. Every verb takes it as the first
# argument, mutates it, and returns it again (so calls can be chained). Tensors
# are placed at explicit `(x, y)` coordinates and remembered by a `Symbol` name;
# bonds and partitions then refer to tensors by that name rather than by position.

s = schematic(; figure_kw=(size=(420, 240),))
tensor!(s, :A, (0, 0))
tensor!(s, :B, (1.4, 0))
bond!(s, :A, :B; label="χ")
s.fig

# The canvas is `s.fig` — a plain `Makie.Figure` you can `save(...)`, embed, or
# display. Notice the bond is drawn *behind* the circles: nodes always sit on top.

# ## 2. Directed bonds and open legs
#
# A tensor's *open* legs — physical indices with no partner — are drawn with
# [`leg!`](@ref), pointing in a direction given either as an angle (radians) or a
# `(dx, dy)` vector. Arrowheads follow Qritical's variance convention: an
# `into=true` leg points *toward* the tensor (an incoming `Upper` index).

s = schematic(; figure_kw=(size=(560, 260),))
for (i, name) in enumerate((:A, :B, :C))
    tensor!(s, name, (i - 1, 0); color=:steelblue)
    leg!(s, name, π/2; label="σ$i", arrow=true, into=true)   # physical leg, pointing in
end
bond!(s, :A, :B; label="χ", arrow=true)
bond!(s, :B, :C; label="χ", arrow=true)
s.fig

# This is the canonical picture of a three-site MPS: a horizontal spine of virtual
# bonds with physical legs hanging off each site. The bond arrows show the flow of
# the gauge (here left-to-right, as in a left-canonical state).

# ## 3. Partitions — the whole point
#
# The reason to hand-draw is usually to **highlight a grouping**: a bipartition
# cut, an entanglement region, a block to be coarse-grained. [`partition!`](@ref)
# takes a list of tensor names and wraps them in a translucent rounded blob — the
# convex hull of their positions, inflated and smoothed. The blob is always drawn
# behind everything else, so it reads as a background region.
#
# Here is the bipartition ``\{1,2,3 \mid 4,5,6\}`` of a six-site chain — the cut
# whose Schmidt spectrum GS-2 truncates:

s = schematic(; figure_kw=(size=(720, 300),))
for i in 1:6
    tensor!(s, Symbol(:t, i), (i - 1, 0); color=:slategray)
    leg!(s, Symbol(:t, i), π/2; length=0.45)
end
for i in 1:5
    bond!(s, Symbol(:t, i), Symbol(:t, i + 1))
end
partition!(s, [Symbol(:t, i) for i in 1:3]; color=:steelblue,  label="block L")
partition!(s, [Symbol(:t, i) for i in 4:6]; color=:tomato,     label="block R")
note!(s, (2.5, -1.15), "the cut bond carries the Schmidt spectrum"; fontsize=11, color=:gray)
s.fig

# ## 4. Different partitions of the same network
#
# The same tensors can be cut many ways, and the drawing DSL makes each cut a
# one-liner. Below, a 2×3 grid of tensors (think a small PEPS patch or a two-leg
# ladder) is shown twice: once cut **vertically** into left/right halves, once cut
# **horizontally** into top/bottom rows. Only the `partition!` calls differ.

## helper: lay down the same 2×3 grid on any canvas
function grid_2x3!(s)
    for r in 1:2, c in 1:3
        tensor!(s, Symbol(:g, r, c), (c - 1, r - 1); radius=0.18, color=:slategray)
    end
    for r in 1:2, c in 1:2                       # horizontal bonds
        bond!(s, Symbol(:g, r, c), Symbol(:g, r, c + 1))
    end
    for c in 1:3                                 # vertical (rung) bonds
        bond!(s, Symbol(:g, 1, c), Symbol(:g, 2, c))
    end
    return s
end

sv = schematic(; figure_kw=(size=(460, 320),))
grid_2x3!(sv)
partition!(sv, [Symbol(:g, r, c) for r in 1:2, c in 1:1]; color=:seagreen, label="A")
partition!(sv, [Symbol(:g, r, c) for r in 1:2, c in 2:3]; color=:mediumpurple, label="B")
note!(sv, (1.0, -0.95), "vertical cut"; fontsize=12, color=:gray)
sv.fig

#-

sh = schematic(; figure_kw=(size=(460, 320),))
grid_2x3!(sh)
partition!(sh, [Symbol(:g, 1, c) for c in 1:3]; color=:orange, label="top")
partition!(sh, [Symbol(:g, 2, c) for c in 1:3]; color=:steelblue, label="bottom")
note!(sh, (1.0, -0.95), "horizontal cut"; fontsize=12, color=:gray)
sh.fig

# Because `partition!` just takes whatever tensors you name, overlapping and
# nested regions work too — draw a large faint blob over the whole network and a
# smaller saturated one over a sub-block to show a region inside a region.

# ## 5. Putting it together
#
# A slightly richer figure: a five-site MPS with the orthogonality centre marked,
# a bond label on every link, and a partition highlighting the two-site window an
# algorithm like DMRG or TEBD updates at each step.

s = schematic(; figure_kw=(size=(820, 300),))
colors = [:steelblue, :steelblue, :tomato, :seagreen, :seagreen]
for i in 1:5
    tensor!(s, Symbol(:m, i), (i - 1, 0); color=colors[i])
    leg!(s, Symbol(:m, i), π/2; label="σ$i", arrow=true, into=true, length=0.5)
end
for i in 1:4
    bond!(s, Symbol(:m, i), Symbol(:m, i + 1); label="χ$i", arrow=true)
end
partition!(s, [:m2, :m3]; color=:goldenrod, padding=0.45, label="two-site update")
note!(s, (2.0, -1.2), "blue = left-canonical · red = centre · green = right-canonical";
      fontsize=11, color=:gray)
s.fig

# ---
#
# *The API mirrors quimb's `schematic.Drawing` — `tensor!`/`bond!`/`leg!` for the
# skeleton, `partition!` for the regions — but stays in pure Julia + Makie, so the
# same figures render in the docs, a notebook, or a paper with no extra tooling.*
