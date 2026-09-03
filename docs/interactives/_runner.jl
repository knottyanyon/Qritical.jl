# Build all interactive HTML widgets and write them to docs/src/assets/interactives/.
#
# Called once from make.jl before makedocs.  Each widget file is self-contained —
# it defines a build_* function and nothing else — so adding a new interactive for
# a future exercise only requires:
#   1. Create docs/interactives/my_widget.jl
#   2. Add two lines here (include + write)

using SHA

const _INTERACTIVES_OUT = joinpath(@__DIR__, "..", "src", "assets", "interactives")
mkpath(_INTERACTIVES_OUT)

const _DATA_ROOT = joinpath(@__DIR__, "..", "src", "tutorials", "data")

# SHA-256 stamp helpers — same pattern used by the Literate build in make.jl.
# Each builder script gets a sidecar .sha256 file next to the output HTML.
# If the script hasn't changed since the last build, the (expensive) computation
# is skipped.
function _interactive_changed(src::String, out::String)::Bool
    stamp = out * ".sha256"
    isfile(stamp) || return true
    isfile(out) || return true
    return read(stamp, String) == bytes2hex(sha256(read(src))) ? false : true
end
function _write_interactive_stamp(src::String, out::String)
    return write(out * ".sha256", bytes2hex(sha256(read(src))))
end

# ── SVD image compression (Tutorial 01) ──────────────────────────────────────
include(joinpath(@__DIR__, "svd_compression.jl"))

let src = joinpath(@__DIR__, "svd_compression.jl"),
    out = joinpath(_INTERACTIVES_OUT, "svd_compression.html")

    if _interactive_changed(src, out)
        html = build_svd_compression(joinpath(_DATA_ROOT, "Bahkauv.png"))
        write(out, html)
        _write_interactive_stamp(src, out)
        @info "Interactive built" file = relpath(out, joinpath(@__DIR__, ".."))
    else
        @info "Interactive up-to-date (skipped)" file = relpath(
            out, joinpath(@__DIR__, "..")
        )
    end
end

# ── TEBD Néel-quench explorer (Tutorial 08) ───────────────────────────────────
include(joinpath(@__DIR__, "tebd_explorer.jl"))

let src = joinpath(@__DIR__, "tebd_explorer.jl"),
    out = joinpath(_INTERACTIVES_OUT, "tebd_explorer.html")

    if _interactive_changed(src, out)
        html = build_tebd_explorer()
        write(out, html)
        _write_interactive_stamp(src, out)
        @info "Interactive built" file = relpath(out, joinpath(@__DIR__, ".."))
    else
        @info "Interactive up-to-date (skipped)" file = relpath(
            out, joinpath(@__DIR__, "..")
        )
    end
end
