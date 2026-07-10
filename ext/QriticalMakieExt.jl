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
const _LEG_ANGLES = Dict{Int, Vector{Float64}}(
    1 => [0.0],
    2 => [π,    0.0],
    3 => [π,    3π/2,  0.0],
    4 => [π,    5π/4,  7π/4,  0.0],
    5 => [π,    5π/4,  3π/2,  7π/4,  0.0],
)

function _angles_for(N::Int)
    haskey(_LEG_ANGLES, N) && return _LEG_ANGLES[N]
    return [2π * k / N for k in 0:(N-1)]   # evenly spaced fallback
end

# ── QTensor diagram ───────────────────────────────────────────────────────────
"""
    draw(A::QTensor; name="", show_dims=true, figure_kw=(;))

Draw `A` as a single tensor node: a labelled circle with one leg stub per
index. Stub colour encodes variance — blue for `Upper` (incoming), grey for
`Lower` (outgoing).  The leg label and dimension are printed at the tip.

Suitable for illustrating a single site tensor or a merged contraction result.
"""
function Qritical.draw(A::QTensor;
    name      = "",
    show_dims = true,
    figure_kw = (;),
)
    N      = ndims(A)
    angles = _angles_for(N)
    r_stub = 0.65   # leg stub length
    r_circ = 0.28   # circle radius (for label placement)

    fig = Figure(; size=(320, 320), figure_kw...)
    ax  = Axis(fig[1, 1]; aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)

    for (ix, θ) in zip(A.indices, angles)
        cx, cy = r_stub * cos(θ), r_stub * sin(θ)
        lx, ly = (r_stub + 0.18) * cos(θ), (r_stub + 0.18) * sin(θ)

        col = ix isa TIx{Upper} ? _UPPER_COLOR : _LOWER_COLOR
        lines!(ax, [0.0, cx], [0.0, cy]; color=(col, 0.85), linewidth=3)

        lbl = show_dims ? "$(label(ix)) ($(dim(ix)))" : string(label(ix))
        text!(ax, lx, ly; text=lbl, fontsize=11, align=(:center, :center), color=col)
    end

    scatter!(ax, [0.0], [0.0];
             color=_ARBITRARY_COLOR, markersize=46, strokecolor=:white, strokewidth=2)
    text!(ax, 0.0, 0.0;
          text=string(name), fontsize=13, align=(:center, :center),
          color=:white, font=:bold)

    return fig
end

function _site_color(i::Int, form::AbstractMPSForm)
    form isa CanonicalForm || return _ARBITRARY_COLOR
    i <= form.llim && return _LEFT_COLOR
    i >= form.rlim && return _RIGHT_COLOR
    return _CENTER_COLOR
end

# ── FiniteMPS diagram ─────────────────────────────────────────────────────────
function Qritical.draw(mps::FiniteMPS;
    show_dims   = true,
    show_legend = true,
    figure_kw   = (;),
)
    L    = length(mps.tensors)
    form = mps.form
    xs   = Float64.(1:L)

    colors = [_site_color(i, form) for i in 1:L]

    width = max(500, L * 90 + 100)
    fig = Figure(; size=(show_legend ? width + 140 : width, 220), figure_kw...)
    ax  = Axis(fig[1, 1]; aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)

    # Virtual bond lines
    for i in 1:(L - 1)
        lines!(ax, [xs[i], xs[i+1]], [0.0, 0.0]; color=(:gray, 0.7), linewidth=2.5)
        if show_dims
            χ = size(mps.tensors[i].data, 3)
            text!(ax, (xs[i] + xs[i+1]) / 2, 0.13;
                  text=string(χ), fontsize=11, align=(:center, :bottom), color=:gray)
        end
    end

    # Physical leg stubs
    for i in 1:L
        lines!(ax, [xs[i], xs[i]], [0.0, -0.5]; color=(:gray, 0.7), linewidth=2.5)
        if show_dims
            d = size(mps.tensors[i].data, 2)
            text!(ax, xs[i], -0.62;
                  text=string(d), fontsize=11, align=(:center, :top), color=:gray)
        end
    end

    # Site circles
    scatter!(ax, xs, zeros(L);
             color=colors, markersize=36, strokecolor=:white, strokewidth=2)

    # Site index labels inside circles
    for i in 1:L
        text!(ax, xs[i], 0.0;
              text=string(i), fontsize=12, align=(:center, :center),
              color=:white, font=:bold)
    end

    # Legend (only when form information is available)
    if show_legend && form isa CanonicalForm
        Legend(fig[1, 2],
            [
                MarkerElement(color=_LEFT_COLOR,   marker=:circle, markersize=14),
                MarkerElement(color=_CENTER_COLOR, marker=:circle, markersize=14),
                MarkerElement(color=_RIGHT_COLOR,  marker=:circle, markersize=14),
            ],
            ["left-canonical", "ortho centre", "right-canonical"];
            framevisible=false, labelsize=11, rowgap=4)
    end

    return fig
end

end # module QriticalMakieExt
