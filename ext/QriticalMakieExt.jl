module QriticalMakieExt

using Qritical
using Makie

# ── Canonical-form colour palette (mirrors quimb's convention) ────────────────
const _LEFT_COLOR      = RGBf(0.25, 0.39, 0.85)   # Julia blue  — left-canonical
const _CENTER_COLOR    = RGBf(0.80, 0.24, 0.20)   # Julia red   — ortho centre
const _RIGHT_COLOR     = RGBf(0.36, 0.55, 0.24)   # Julia green — right-canonical
const _ARBITRARY_COLOR = RGBf(0.55, 0.55, 0.55)   # grey        — unknown form

# ── Leg variance colours ──────────────────────────────────────────────────────
const _UPPER_COLOR = RGBf(0.25, 0.39, 0.85)   # blue — incoming / contravariant
const _LOWER_COLOR = RGBf(0.55, 0.55, 0.55)   # grey — outgoing / covariant

# Leg angles (radians) per tensor rank — gives natural layouts for the most
# common tensor shapes without any spring-force computation.
const _LEG_ANGLES = Dict{Int,Vector{Float64}}(
    1 => [0.0],
    2 => [π,    0.0],
    3 => [π,    3π/2,  0.0],
    4 => [π,    5π/4,  7π/4,  0.0],
    5 => [π,    5π/4,  3π/2,  7π/4,  0.0],
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
    scatter!(ax, [t * cx], [t * cy];
             marker=:rtriangle, markersize=13, rotation=rotation,
             color=(col, 0.9), strokewidth=0)
end

# ── QTensor diagram ───────────────────────────────────────────────────────────
"""
    draw(A::QTensor; name="", show_dims=true, figure_kw=(;))

Draw `A` as a single tensor node: a labelled circle with one directed leg stub
per index.

- **Blue stub, arrow pointing in** → `Upper` index (incoming / contravariant)
- **Grey stub, arrow pointing out** → `Lower` index (outgoing / covariant)

The leg label and dimension are printed at the tip of each stub.

# Keyword arguments
- `name` — label drawn inside the tensor circle
- `show_dims::Bool` — annotate each leg with its dimension (default `true`)
- `show_arrows::Bool` — draw directed arrowheads on stubs (default `true`)
- `figure_kw` — named-tuple forwarded to `Makie.Figure`
"""
function Qritical.draw(A::QTensor;
    name        = "",
    show_dims   = true,
    show_arrows = true,
    figure_kw   = (;),
)
    N      = ndims(A)
    angles = _angles_for(N)
    r_stub = 0.65   # stub length in data coords
    r_pad  = 0.45   # extra space beyond stub tip for text labels

    fig = Figure(; figure_kw...)
    ax  = Axis(fig[1, 1]; aspect=DataAspect(),
               limits=(-r_stub - r_pad, r_stub + r_pad,
                       -r_stub - r_pad, r_stub + r_pad))
    hidedecorations!(ax)
    hidespines!(ax)

    for (ix, θ) in zip(A.indices, angles)
        cx, cy   = r_stub * cos(θ), r_stub * sin(θ)
        lx, ly   = (r_stub + 0.25) * cos(θ), (r_stub + 0.25) * sin(θ)
        is_upper = ix isa TIx{Upper}
        col      = is_upper ? _UPPER_COLOR : _LOWER_COLOR

        lines!(ax, [0.0, cx], [0.0, cy]; color=(col, 0.85), linewidth=3)
        show_arrows && _arrowhead!(ax, cx, cy, θ, col, is_upper)

        lbl = show_dims ? "$(label(ix)) ($(dim(ix)))" : string(label(ix))
        text!(ax, lx, ly; text=lbl, fontsize=11, align=(:center, :center), color=col)
    end

    scatter!(ax, [0.0], [0.0];
             color=_ARBITRARY_COLOR, markersize=46,
             strokecolor=:white, strokewidth=2)
    text!(ax, 0.0, 0.0;
          text=string(name), fontsize=13, align=(:center, :center),
          color=:white, font=:bold)

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

"""
    draw(mps::FiniteMPS; show_dims=true, show_legend=true, figure_kw=(;))

Draw a horizontal tensor-network diagram of `mps`:
- Sites as labelled circles, coloured by canonical form
- Virtual bond lines with directed arrows converging on the ortho centre
- Physical leg stubs (arrows pointing into the tensor, as σ is always Upper)
- Bond and physical dimensions annotated when `show_dims=true`

# Keyword arguments
- `show_dims::Bool` — annotate bond/physical dimensions (default `true`)
- `show_arrows::Bool` — draw directed arrowheads on bonds and physical legs (default `true`)
- `show_legend::Bool` — show canonical-form colour legend (default `true`)
- `figure_kw` — named-tuple forwarded to `Makie.Figure`
"""
function Qritical.draw(mps::FiniteMPS;
    show_dims   = true,
    show_arrows = true,
    show_legend = true,
    figure_kw   = (;),
)
    L    = length(mps.tensors)
    form = mps.form
    xs   = Float64.(1:L)

    colors = [_site_color(i, form) for i in 1:L]

    # y bounds: +0.35 above (bond dim labels) to -0.9 below (phys stub + dim label)
    y_lo = show_dims ? -0.9 : -0.65
    y_hi = show_dims ?  0.35 :  0.2
    x_lo = xs[1] - 0.6
    x_hi = xs[L] + 0.6

    fig = Figure(; figure_kw...)
    ax  = Axis(fig[1, 1]; aspect=DataAspect(),
               limits=(x_lo, x_hi, y_lo, y_hi))
    hidedecorations!(ax)
    hidespines!(ax)

    # Virtual bond lines + arrows
    for i in 1:(L - 1)
        x1, x2 = xs[i], xs[i + 1]
        lines!(ax, [x1, x2], [0.0, 0.0]; color=(:gray, 0.7), linewidth=2.5)

        # Arrow at midpoint: direction encodes canonical form
        if show_arrows
            goes_right = _bond_arrow_right(i, form)
            mx  = (x1 + x2) / 2
            rot = goes_right ? 0.0 : π
            scatter!(ax, [mx], [0.0];
                     marker=:rtriangle, markersize=12, rotation=rot,
                     color=(:gray, 0.8), strokewidth=0)
        end

        if show_dims
            χ = size(mps.tensors[i].data, 3)
            text!(ax, mx, 0.14;
                  text=string(χ), fontsize=11, align=(:center, :bottom), color=:gray)
        end
    end

    # Physical leg stubs — σ is always Upper so arrow points INTO tensor (upward)
    for i in 1:L
        lines!(ax, [xs[i], xs[i]], [0.0, -0.5]; color=(:gray, 0.7), linewidth=2.5)
        if show_arrows
            scatter!(ax, [xs[i]], [-0.26];       # midpoint, arrow points up (σ is Upper)
                     marker=:rtriangle, markersize=12, rotation=π/2,
                     color=(:gray, 0.8), strokewidth=0)
        end
        if show_dims
            d = size(mps.tensors[i].data, 2)
            text!(ax, xs[i], -0.65;
                  text=string(d), fontsize=11, align=(:center, :top), color=:gray)
        end
    end

    # Site circles + labels
    scatter!(ax, xs, zeros(L);
             color=colors, markersize=36, strokecolor=:white, strokewidth=2)
    for i in 1:L
        text!(ax, xs[i], 0.0;
              text=string(i), fontsize=12, align=(:center, :center),
              color=:white, font=:bold)
    end

    # Legend
    if show_legend && form isa CanonicalForm
        Legend(fig[1, 2],
            [
                MarkerElement(; color=_LEFT_COLOR,   marker=:circle, markersize=14),
                MarkerElement(; color=_CENTER_COLOR, marker=:circle, markersize=14),
                MarkerElement(; color=_RIGHT_COLOR,  marker=:circle, markersize=14),
            ],
            ["left-canonical", "ortho centre", "right-canonical"];
            framevisible=false, labelsize=11, rowgap=4)
    end

    return fig
end

end # module QriticalMakieExt
