module QriticalMakieExt

using Qritical
using Makie

# ── Canonical-form colour palette (Julia-brand colours) ───────────────────────
# github.com/JuliaLang/julia-logo-graphics — the four dots of the Julia logo.
const _LEFT_COLOR = RGBf(0.251, 0.388, 0.847)   # Julia blue   — left-canonical
const _CENTER_COLOR = RGBf(0.796, 0.235, 0.200)   # Julia red    — ortho centre
const _RIGHT_COLOR = RGBf(0.220, 0.596, 0.149)   # Julia green  — right-canonical
const _ARBITRARY_COLOR = RGBf(0.584, 0.345, 0.698)   # Julia purple — unknown form

# ── Leg variance colours ──────────────────────────────────────────────────────
const _UPPER_COLOR = _LEFT_COLOR                  # incoming / contravariant
const _LOWER_COLOR = RGBf(0.55, 0.55, 0.58)       # outgoing / covariant

# Coerce a label to something `Makie.text!` renders: pass `AbstractString`
# through untouched (so a `LaTeXString` keeps its math rendering) and stringify a
# `Symbol` name. Returns `nothing`/`""` unchanged so callers can skip empty labels.
_txt(x) = x isa Symbol ? string(x) : x
# Plain text is drawn bold; a LaTeXString carries its own math weight, so leave it.
_boldish(x) = x isa String ? :bold : :regular

# ── Tensor-kind shapes ─────────────────────────────────────────────────────────
# The silhouette encodes an operator's index structure (cf. tensors.net):
#
#   :diagonal → diamond   — a diagonal matrix (e.g. a singular-value spectrum Σ);
#                           a square whose only surviving entries lie on the
#                           diagonal, drawn as a rotated square.
#   :unitary  → D-arch    — U with UU† = U†U = I. Drawn as a flat base + straight
#                           sides + a semicircular dome, centred on the node: the
#                           flat base faces −y', the dome bulges toward +y'.
#   :isometry → trapezoid — W with W†W = I but WW† ≠ I. The in/out dimensions
#                           differ, so an *asymmetric* wedge fits: the WIDE base
#                           carries the larger (contracted) index and the shape
#                           tapers to a NARROW (still flat — not a point) top on
#                           the smaller side. A triangle would be wrong: it says
#                           the contracted index has dimension one.
#
# `rotation` (radians) orients the shape: 0 points the isometry's narrow top /
# the unitary's dome along +y' (dome up); π flips it, etc. The diamond is
# rotationally (mod π/2) symmetric, so `rotation` has no visible effect on it.
# The polygon shapes are tuned so a circle, diamond, D-arch and trapezoid of the
# same `radius` read as the *same visual size* — each roughly fills the ±r box.
const _ISO_BASE = 0.9f0    # trapezoid base/top |x'| (fraction of r)
const _ARCH = 0.9f0        # D-arch half-extent (fraction of r) — trims the dome

function _shape_points(shape::Symbol, p::Point2f, radius::Real, rotation::Real; nseg::Int=24)
    r = Float32(radius)
    c, sn = Float32(cos(rotation)), Float32(sin(rotation))
    rot(x, y) = Point2f(p[1] + x * c - y * sn, p[2] + x * sn + y * c)
    if shape === :diagonal
        d = r * 1.02f0          # diamond half-diagonal (slightly > r for equal heft)
        return [rot(d, 0), rot(0, d), rot(-d, 0), rot(0, -d)]
    elseif shape === :unitary
        # "D-arch": flat base at y' = -a, straight sides up to the springline
        # y' = 0, then a semicircular dome (radius a) over the top. Centred on p;
        # poly! closes the dome back to the base.
        a = _ARCH * r
        pts = Point2f[rot(a, -a), rot(-a, -a)]           # base: bottom-right → bottom-left
        for t in range(Float32(π), 0.0f0; length=nseg)   # dome: (-a,0) over (0,a) to (a,0)
            push!(pts, rot(a * cos(t), a * sin(t)))
        end
        return pts
    elseif shape === :isometry
        base_x = -_ISO_BASE * r   # wide base — larger index
        top_x = _ISO_BASE * r     # narrow top — smaller (contracted) index
        base_hw = 0.95f0 * r      # base half-width
        top_hw = 0.42f0 * r       # top half-width (nonzero: a trapezoid, not a triangle)
        return [
            rot(base_x, -base_hw),
            rot(top_x, -top_hw),
            rot(top_x, top_hw),
            rot(base_x, base_hw),
        ]
    else
        throw(ArgumentError("no polygon for shape :$shape"))
    end
end

# The wide flat base of an isometry (and the flat base of a unitary D-arch) is the
# face its contracted legs attach to. Returns the mid-point of that base face for
# a node centred at `p` with the given `rotation`, so legs can be routed *into* it
# rather than across a slanted edge. For the trapezoid the base sits at x' = -0.85r.
function _base_anchor(shape::Symbol, p::Point2f, radius::Real, rotation::Real)
    r = Float32(radius)
    c, sn = Float32(cos(rotation)), Float32(sin(rotation))
    bx = shape === :isometry ? -_ISO_BASE * r : 0.0f0
    by = shape === :isometry ? 0.0f0 : -_ARCH * r   # unitary base is the bottom edge
    return Point2f(p[1] + bx * c - by * sn, p[2] + bx * sn + by * c)
end

# Outward unit normal of that base face (points away from the node body), so a
# leg can approach the base head-on.
function _base_normal(shape::Symbol, rotation::Real)
    c, sn = Float32(cos(rotation)), Float32(sin(rotation))
    nx, ny = shape === :isometry ? (-1.0f0, 0.0f0) : (0.0f0, -1.0f0)
    return Point2f(nx * c - ny * sn, nx * sn + ny * c)
end

# L-bend path for a leg entering a node's flat base head-on (image #2): attach a
# little way along the base *toward* the leg direction `d` (so a downward leg meets
# the lower base and an upward leg the upper — keeping it clear of the horizontal
# bond that enters the base centre), step out along the base normal to an elbow,
# then run to the tip along `d`. Returns `(attach, elbow, tip)`.
function _base_leg_path(
    shape::Symbol, p::Point2f, radius::Real, rotation::Real, d::Point2f, length::Real
)
    r = Float32(radius)
    B = _base_anchor(shape, p, r, rotation)
    N = _base_normal(shape, rotation)
    T = Point2f(-N[2], N[1])                       # tangent along the base face
    along = d[1] * T[1] + d[2] * T[2]
    sg = along == 0 ? 0.0f0 : Float32(sign(along))
    attach = Point2f(B[1] + 0.5f0 * r * sg * T[1], B[2] + 0.5f0 * r * sg * T[2])
    stub = 0.3f0 * Float32(length)
    elbow = Point2f(attach[1] + stub * N[1], attach[2] + stub * N[2])
    tip = Point2f(elbow[1] + Float32(length) * d[1], elbow[2] + Float32(length) * d[2])
    return attach, elbow, tip
end

# Draws a single tensor node — the shared primitive behind `node!` and the
# `draw` node(s) — dispatching on `shape`:
#   :general  → circle            (generic tensor, no special structure)
#   :diagonal → diamond           (diagonal / singular-value matrix Σ)
#   :unitary  → D-arch            (U with UU† = U†U = I — flat base + domed top)
#   :isometry → trapezoidal wedge (W with W†W = I; narrow top points toward the
#                                   contracted/smaller side, per `rotation`)
function _draw_node!(
    ax,
    p::Point2f,
    radius::Real;
    shape=:general,
    rotation=0.0,
    color=_ARBITRARY_COLOR,
    strokecolor=:black,
    strokewidth=1.6,
)
    if shape === :general
        poly!(
            ax,
            Circle(p, Float32(radius));
            color=color,
            strokecolor=strokecolor,
            strokewidth=strokewidth,
        )
    elseif shape === :diagonal || shape === :unitary || shape === :isometry
        poly!(
            ax,
            _shape_points(shape, p, radius, rotation);
            color=color,
            strokecolor=strokecolor,
            strokewidth=strokewidth,
        )
    else
        throw(
            ArgumentError(
                "unknown tensor shape :$shape — expected :general, :diagonal, :unitary or :isometry",
            ),
        )
    end
    return nothing
end

# Leg angles (radians) per tensor rank — gives natural layouts for the most
# common tensor shapes without any spring-force computation.
const _LEG_ANGLES = Dict{Int,Vector{Float64}}(
    1 => [0.0],
    2 => [π, 0.0],
    3 => [π, 3π/2, 0.0],
    4 => [π, 5π/4, 7π/4, 0.0],
    5 => [π, 5π/4, 3π/2, 7π/4, 0.0],
)

function _angles_for(N::Int)
    haskey(_LEG_ANGLES, N) && return _LEG_ANGLES[N]
    return [2π * k / N for k in 0:(N - 1)]   # evenly spaced fallback
end

# Draw a directed arrowhead at fraction `t` along the segment (0,0)→(cx,cy).
# Upper (incoming) → head points toward tensor (θ + π).
# Lower (outgoing) → head points away from tensor (θ).
function _arrowhead!(ax, cx, cy, θ, col, is_upper)
    t = 0.52
    rotation = is_upper ? θ + π : θ
    scatter!(
        ax,
        [t * cx],
        [t * cy];
        marker=:rtriangle,
        markersize=13,
        rotation=rotation,
        color=(col, 0.9),
        strokewidth=0,
    )
end

# ── QTensor diagram ───────────────────────────────────────────────────────────
# Public docstring lives on the `draw` stub in `src/Qritical.jl`.
function Qritical.draw(
    A::QTensor;
    name="",
    show_dims=true,
    show_arrows=true,
    shape=:general,
    rotation=0.0,
    figure_kw=(;),
)
    N = ndims(A)
    angles = _angles_for(N)
    r_stub = 0.65   # stub length in data coords
    r_pad = 0.45   # extra space beyond stub tip for text labels

    fig = Figure(; figure_kw...)
    ax = Axis(
        fig[1, 1];
        aspect=DataAspect(),
        limits=(-r_stub - r_pad, r_stub + r_pad, -r_stub - r_pad, r_stub + r_pad),
    )
    hidedecorations!(ax)
    hidespines!(ax)

    for (ix, θ) in zip(A.indices, angles)
        cx, cy = r_stub * cos(θ), r_stub * sin(θ)
        lx, ly = (r_stub + 0.25) * cos(θ), (r_stub + 0.25) * sin(θ)
        is_upper = ix isa TIx{Upper}
        col = is_upper ? _UPPER_COLOR : _LOWER_COLOR

        lines!(ax, [0.0, cx], [0.0, cy]; color=(col, 0.85), linewidth=3, linecap=:round)
        show_arrows && _arrowhead!(ax, cx, cy, θ, col, is_upper)

        lbl = show_dims ? "$(label(ix)) ($(dim(ix)))" : string(label(ix))
        text!(ax, lx, ly; text=lbl, fontsize=14, align=(:center, :center), color=col)
    end

    center = Point2f(0, 0)
    r_node = 0.30
    _draw_node!(ax, center, r_node; shape=shape, rotation=rotation, color=_ARBITRARY_COLOR)
    text!(
        ax,
        center;
        text=_txt(name),
        fontsize=17,
        align=(:center, :center),
        color=:white,
        font=_boldish(_txt(name)),
    )

    return fig
end

# ── FiniteMPS diagram ─────────────────────────────────────────────────────────
function _site_color(i::Int, form::AbstractMPSForm)
    form isa CanonicalForm || return _ARBITRARY_COLOR
    i <= form.llim && return _LEFT_COLOR
    i >= form.rlim && return _RIGHT_COLOR
    return _CENTER_COLOR
end

# Bond arrow direction: bonds to the left of the ortho centre point →,
# bonds to the right point ←, so all arrows converge on the centre.
function _bond_arrow_right(i::Int, form::AbstractMPSForm)
    form isa CanonicalForm || return true   # default: left-to-right
    center = (form.llim + form.rlim) / 2
    return i < center
end

# Left/right-canonical sites are genuine isometries (A†A = I resp. BB† = I),
# so they get the tensors.net wedge; the orthogonality centre and any site of
# unknown/arbitrary form carry no such guarantee and stay a general circle.
function _site_shape(i::Int, form::AbstractMPSForm)
    form isa CanonicalForm || return :general
    (i <= form.llim || i >= form.rlim) && return :isometry
    return :general
end

# The wedge's narrow top points toward the side the isometry contracts onto:
# left-canonical sites contract (vL,σ)→vR, so the narrow end points right (toward
# the next site); right-canonical sites contract (σ,vR)→vL, so it points left.
function _site_rotation(i::Int, form::AbstractMPSForm)
    form isa CanonicalForm && i >= form.rlim && return Float64(π)
    return 0.0
end

# Public docstring lives on the `draw` stub in `src/Qritical.jl`.
function Qritical.draw(
    mps::FiniteMPS;
    show_dims=true,
    show_arrows=true,
    show_shapes=true,
    show_legend=true,
    figure_kw=(;),
)
    L = length(mps.tensors)
    form = mps.form
    xs = Float64.(1:L)

    colors = [_site_color(i, form) for i in 1:L]

    # y bounds: +0.35 above (bond dim labels) to -0.9 below (phys stub + dim label)
    y_lo = show_dims ? -0.9 : -0.65
    y_hi = show_dims ? 0.35 : 0.2
    x_lo = xs[1] - 0.6
    x_hi = xs[L] + 0.6

    fig = Figure(; figure_kw...)
    ax = Axis(fig[1, 1]; aspect=DataAspect(), limits=(x_lo, x_hi, y_lo, y_hi))
    hidedecorations!(ax)
    hidespines!(ax)

    # Virtual bond lines + arrows — the network's "spine", drawn heaviest
    for i in 1:(L - 1)
        x1, x2 = xs[i], xs[i + 1]
        lines!(ax, [x1, x2], [0.0, 0.0]; color=(:gray, 0.7), linewidth=3, linecap=:round)

        # Arrow at midpoint: direction encodes canonical form
        if show_arrows
            goes_right = _bond_arrow_right(i, form)
            mx = (x1 + x2) / 2
            rot = goes_right ? 0.0 : π
            scatter!(
                ax,
                [mx],
                [0.0];
                marker=:rtriangle,
                markersize=12,
                rotation=rot,
                color=(:gray, 0.8),
                strokewidth=0,
            )
        end

        if show_dims
            χ = size(mps.tensors[i].data, 3)
            text!(
                ax,
                mx,
                0.14;
                text=string(χ),
                fontsize=13,
                align=(:center, :bottom),
                color=:gray,
            )
        end
    end

    # Physical leg stubs — thinner than bonds so the spine reads first. σ is always
    # Upper so the arrow points INTO the tensor (upward). On isometry sites the leg
    # bends into the flat base (the contracted (vL,σ) / (σ,vR) side) instead of
    # spearing the slanted wedge edge; on a general site it drops straight down.
    r_node = 0.24
    for i in 1:L
        p = Point2f(xs[i], 0.0)
        shp = show_shapes ? _site_shape(i, form) : :general
        rot_i = _site_rotation(i, form)
        if shp === :isometry
            attach, elbow, tip = _base_leg_path(:isometry, p, r_node, rot_i,
                                                Point2f(0.0, -1.0), 0.5)
            lines!(ax, [attach, elbow, tip]; color=(:gray, 0.7), linewidth=2.5,
                   linecap=:round)
            arr = _P((elbow + tip) / 2)
            lx = tip[1]
        else
            tip = _P(p + Point2f(0.0, -0.5))
            lines!(ax, [p, tip]; color=(:gray, 0.7), linewidth=2.5, linecap=:round)
            arr = Point2f(xs[i], -0.26)
            lx = xs[i]
        end
        if show_arrows
            scatter!(ax, [arr]; marker=:rtriangle, markersize=13, rotation=π/2,
                     color=(:gray, 0.8), strokewidth=0)
        end
        if show_dims
            d = size(mps.tensors[i].data, 2)
            text!(ax, lx, -0.65; text=string(d), fontsize=13,
                  align=(:center, :top), color=:gray)
        end
    end

    # Site nodes + labels. Left/right-canonical sites are drawn as isometry wedges
    # (per tensors.net convention); the ortho centre / arbitrary-form sites stay a
    # general circle.
    for i in 1:L
        p = Point2f(xs[i], 0.0)
        shape = show_shapes ? _site_shape(i, form) : :general
        rot = _site_rotation(i, form)
        _draw_node!(ax, p, r_node; shape=shape, rotation=rot, color=colors[i])
        text!(
            ax,
            p;
            text=string(i),
            fontsize=15,
            align=(:center, :center),
            color=:white,
            font=:bold,
        )
    end

    # Legend
    if show_legend && form isa CanonicalForm
        Legend(
            fig[1, 2],
            [
                MarkerElement(; color=_LEFT_COLOR, marker=:circle, markersize=14),
                MarkerElement(; color=_CENTER_COLOR, marker=:circle, markersize=14),
                MarkerElement(; color=_RIGHT_COLOR, marker=:circle, markersize=14),
            ],
            if show_shapes
                [
                    "left-canonical (isometry →)",
                    "ortho centre",
                    "right-canonical (isometry ←)",
                ]
            else
                ["left-canonical", "ortho centre", "right-canonical"]
            end;
            framevisible=false,
            labelsize=13,
            rowgap=4,
        )
    end

    return fig
end

# ============================================================================
# Schematic drawing DSL — a quimb-style hand-drawing canvas
#
# Mirrors `quimb.schematic.Drawing`: a thin builder that lets you place tensor
# nodes, connect them with bonds, add dangling legs, and — most usefully —
# highlight groups of tensors with translucent **partition** blobs.  Everything
# lives in data coordinates so shapes stay proportional under `DataAspect`.
# ============================================================================

# Hand-drawing canvas returned by `schematic` (see the `schematic` stub in
# `src/Qritical.jl` for the public docstring).  Holds the Makie `Figure`/`Axis`,
# remembers each named tensor's position, and tracks a bounding box so the view
# auto-expands as elements are added.
mutable struct Schematic
    fig::Makie.Figure
    ax::Makie.Axis
    pos::Dict{Symbol,Point2f}
    meta::Dict{Symbol,Tuple{Symbol,Float64,Float64}}   # name → (shape, rotation, radius)
    bbox::NTuple{4,Float64}   # xmin, xmax, ymin, ymax
    pad::Float64
end

# ── coordinate helpers ────────────────────────────────────────────────────────
_P(x) = Point2f(x[1], x[2])   # coerce any point-like (tuple, vector, Point) to Point2f
_resolve(s::Schematic, x::Symbol) = s.pos[x]
_resolve(s::Schematic, x) = _P(x)

_vnorm(p) = sqrt(p[1]^2 + p[2]^2)
function _unit(p)
    n = _vnorm(p)
    n == 0 ? Point2f(0, 0) : Point2f(p[1] / n, p[2] / n)
end

# Grow the tracked bounding box to include point `p` (± radius `r`) and refresh
# the axis limits so nothing — including text near the edges — gets cropped.
function _grow!(s::Schematic, p, r::Real=0.0)
    xmin, xmax, ymin, ymax = s.bbox
    s.bbox = (
        min(xmin, p[1] - r), max(xmax, p[1] + r), min(ymin, p[2] - r), max(ymax, p[2] + r)
    )
    xmin, xmax, ymin, ymax = s.bbox
    limits!(s.ax, xmin - s.pad, xmax + s.pad, ymin - s.pad, ymax + s.pad)
    return s
end

# ── convex hull (Andrew's monotone chain), returns CCW vertices ───────────────
function _convex_hull(pts::Vector{Point2f})
    n = length(pts)
    n <= 2 && return copy(pts)
    P = sort(pts; by=p -> (p[1], p[2]))
    cross(o, a, b) = (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])
    lower = Point2f[]
    for p in P
        while length(lower) >= 2 && cross(lower[end - 1], lower[end], p) <= 0
            pop!(lower)
        end
        push!(lower, p)
    end
    upper = Point2f[]
    for p in reverse(P)
        while length(upper) >= 2 && cross(upper[end - 1], upper[end], p) <= 0
            pop!(upper)
        end
        push!(upper, p)
    end
    return vcat(lower[1:(end - 1)], upper[1:(end - 1)])
end

# Rounded convex hull = Minkowski sum of the hull with a disk of radius `r`.
# Straight offset edges joined by circular arcs at each vertex → the smooth
# translucent blob quimb draws with `patch_around`.  Degenerate cases: a single
# point becomes a circle, two points a stadium.
function _rounded_hull(hull::Vector{Point2f}, r::Float64; nseg::Int=12)
    m = length(hull)
    m == 0 && return Point2f[]
    if m == 1
        c = hull[1]
        return [
            Point2f(c[1] + r * cos(t), c[2] + r * sin(t)) for
            t in range(0, 2π; length=4nseg)
        ]
    end
    boundary = Point2f[]
    for i in 1:m
        v = hull[i]
        vprev = hull[mod1(i - 1, m)]
        vnext = hull[mod1(i + 1, m)]
        nin = let d = _unit(v - vprev);
            Point2f(d[2], -d[1])
        end   # outward normals
        nout = let d = _unit(vnext - v);
            Point2f(d[2], -d[1])
        end
        a0 = atan(nin[2], nin[1])
        a1 = atan(nout[2], nout[1])
        while a1 <= a0
            a1 += 2π
        end
        for t in range(a0, a1; length=nseg)
            push!(boundary, Point2f(v[1] + r * cos(t), v[2] + r * sin(t)))
        end
    end
    return boundary
end

# ── canvas constructor ────────────────────────────────────────────────────────
# Public docstring lives on the `schematic` stub in `src/Qritical.jl`.
function Qritical.schematic(; figure_kw=(;), pad=0.6)
    fig = Figure(; figure_kw...)
    ax = Axis(fig[1, 1]; aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    return Schematic(
        fig,
        ax,
        Dict{Symbol,Point2f}(),
        Dict{Symbol,Tuple{Symbol,Float64,Float64}}(),
        (Inf, -Inf, Inf, -Inf),
        Float64(pad),
    )
end

# ── tensor node ───────────────────────────────────────────────────────────────
# Public docstring lives on the `node!` stub in `src/Qritical.jl`.
function Qritical.node!(
    s::Schematic,
    name::Symbol,
    coo;
    radius=0.24,
    color=_ARBITRARY_COLOR,
    label=string(name),
    labelcolor=:white,
    strokecolor=:black,
    strokewidth=1.6,
    fontsize=17,
    shape=:general,
    rotation=0.0,
)
    p = _P(coo)
    s.pos[name] = p
    s.meta[name] = (shape, Float64(rotation), Float64(radius))
    _draw_node!(
        s.ax,
        p,
        radius;
        shape=shape,
        rotation=rotation,
        color=color,
        strokecolor=strokecolor,
        strokewidth=strokewidth,
    )
    if label !== nothing && label != ""
        text!(
            s.ax,
            p;
            text=_txt(label),
            fontsize=fontsize,
            color=labelcolor,
            align=(:center, :center),
            font=_boldish(_txt(label)),
        )
    end
    _grow!(s, p, radius)
    return s
end

# ── bond (line between two tensors / points) ──────────────────────────────────
# Public docstring lives on the `wire!` stub in `src/Qritical.jl`.
function Qritical.wire!(
    s::Schematic,
    a,
    b;
    color=(:gray, 0.85),
    linewidth=3.0,
    label=nothing,
    arrow=false,
    shorten=0.0,
    labeloffset=0.18,
    fontsize=14,
)
    pa, pb = _resolve(s, a), _resolve(s, b)
    d = _unit(pb - pa)
    pa2 = _P(pa + shorten * d)
    pb2 = _P(pb - shorten * d)
    ln = lines!(s.ax, [pa2, pb2]; color=color, linewidth=linewidth, linecap=:round)
    translate!(ln, 0, 0, -1)
    mid = _P((pa2 + pb2) / 2)
    if arrow
        sc = scatter!(
            s.ax,
            [mid];
            marker=:rtriangle,
            markersize=13,
            rotation=atan(d[2], d[1]),
            color=color,
        )
        translate!(sc, 0, 0, -1)
    end
    if label !== nothing
        n = Point2f(-d[2], d[1])   # perpendicular
        text!(
            s.ax,
            _P(mid + labeloffset * n);
            text=_txt(label),
            fontsize=fontsize,
            align=(:center, :center),
            color=:gray,
        )
    end
    _grow!(s, pa)
    _grow!(s, pb)
    return s
end

# ── dangling / open leg ───────────────────────────────────────────────────────
# Public docstring lives on the `stub!` stub in `src/Qritical.jl`.
function Qritical.stub!(
    s::Schematic,
    a,
    dir;
    length=0.5,
    color=(:gray, 0.85),
    linewidth=2.5,
    label=nothing,
    arrow=false,
    into=false,
    base=false,
    fontsize=14,
)
    p = _resolve(s, a)
    d = dir isa Real ? Point2f(cos(dir), sin(dir)) : _unit(_P(dir))

    # A leg into the flat base of an isometry/unitary must approach the base
    # head-on, so if `dir` does not already point along the base normal we bend it
    # into an L: a short stub straight out of the base, then a turn toward `dir`
    # (image #2 in the tensors.net convention — never spear the slanted edge).
    meta = a isa Symbol ? get(s.meta, a, nothing) : nothing
    if base && meta !== nothing && meta[1] in (:isometry, :unitary)
        shape, rotd, rad = meta
        attach, elbow, tip = _base_leg_path(shape, p, rad, rotd, d, length)
        ln = lines!(s.ax, [attach, elbow, tip]; color=color, linewidth=linewidth,
                    linecap=:round)
        translate!(ln, 0, 0, -1)
        if arrow
            mid = _P((elbow + tip) / 2)            # arrowhead on the long (approach) leg
            θ = into ? atan(-d[2], -d[1]) : atan(d[2], d[1])
            sc = scatter!(s.ax, [mid]; marker=:rtriangle, markersize=13,
                          rotation=θ, color=color)
            translate!(sc, 0, 0, -1)
        end
        if label !== nothing
            text!(s.ax, _P(tip + 0.2 * d); text=_txt(label), fontsize=fontsize,
                  align=(:center, :center), color=:gray)
        end
        _grow!(s, tip)
        _grow!(s, elbow)
        return s
    end

    tip = _P(p + length * d)
    ln = lines!(s.ax, [p, tip]; color=color, linewidth=linewidth, linecap=:round)
    translate!(ln, 0, 0, -1)
    if arrow
        mid = _P((p + tip) / 2)
        θ = into ? atan(-d[2], -d[1]) : atan(d[2], d[1])
        sc = scatter!(
            s.ax, [mid]; marker=:rtriangle, markersize=13, rotation=θ, color=color
        )
        translate!(sc, 0, 0, -1)
    end
    if label !== nothing
        text!(
            s.ax,
            _P(tip + 0.2 * d);
            text=_txt(label),
            fontsize=fontsize,
            align=(:center, :center),
            color=:gray,
        )
    end
    _grow!(s, tip)
    return s
end

# ── partition blob ────────────────────────────────────────────────────────────
# Public docstring lives on the `region!` stub in `src/Qritical.jl`.
function Qritical.region!(
    s::Schematic,
    members;
    padding=0.4,
    color=_LEFT_COLOR,
    alpha=0.16,
    strokealpha=0.55,
    strokewidth=1.5,
    linestyle=nothing,
    label=nothing,
    fontsize=15,
)
    pts = Point2f[]
    for m in members                       # loop (not comprehension) so a Matrix
        push!(pts, _resolve(s, m))         # of names still flattens to a Vector
    end
    hull = _convex_hull(pts)
    boundary = _rounded_hull(hull, Float64(padding))
    isempty(boundary) && return s
    col = Makie.to_color(color)
    pl = poly!(
        s.ax,
        boundary;
        color=(col, alpha),
        strokecolor=(col, strokealpha),
        strokewidth=strokewidth,
    )
    linestyle !== nothing && (pl.linestyle = linestyle)
    translate!(pl, 0, 0, -10)   # always behind tensors and bonds
    for p in boundary
        _grow!(s, p)
    end
    if label !== nothing
        xs = [p[1] for p in boundary]
        ys = [p[2] for p in boundary]
        gap = max(0.18, 0.35 * padding)   # scale with blob inflation so labels
        top = Point2f(sum(xs) / Base.length(xs), maximum(ys) + gap)   # clear neighbours
        text!(
            s.ax,
            top;
            text=_txt(label),
            fontsize=fontsize,
            color=col,
            align=(:center, :bottom),
            font=_boldish(_txt(label)),
        )
        _grow!(s, top, 0.1)
    end
    return s
end

# ── free-floating annotation ──────────────────────────────────────────────────
# Public docstring lives on the `note!` stub in `src/Qritical.jl`.
function Qritical.note!(
    s::Schematic, coo, text; color=:black, fontsize=14, align=(:center, :center)
)
    p = _P(coo)
    text!(s.ax, p; text=_txt(text), fontsize=fontsize, color=color, align=align)
    _grow!(s, p)
    return s
end

end # module QriticalMakieExt
