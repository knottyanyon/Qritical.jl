using Qritical
using Documenter
using DocumenterCitations
using DocumenterMermaid
using Literate
using JSON, Markdown
using SHA
using Base64

# Load custom documentation hooks
include(joinpath(@__DIR__, "symbol_docstring_injector.jl"))
include(joinpath(@__DIR__, "glossary_linker.jl"))

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:authoryear)

DocMeta.setdocmeta!(Qritical, :DocTestSetup, :(using Qritical); recursive=true)

# ---------------------------------------------------------------------------
# Build modes
#
#   Normal (CI):  julia --project=docs docs/make.jl
#   Full local:   julia --project=docs -e 'using LiveServer; servedocs()'
#   Fast local:   DOCS_FAST=1 julia --project=docs -e 'using LiveServer; servedocs()'
#
# DOCS_FAST=1 skips exercises entirely and only builds the Getting Started
# pages + API reference — makedocs finishes in under a minute instead of
# 5-10 minutes.  Use it when iterating on prose or API docs.
# ---------------------------------------------------------------------------
const DOCS_FAST = get(ENV, "DOCS_FAST", "0") == "1"
DOCS_FAST && @info "DOCS_FAST=1 — exercises skipped, building core pages only"

# SHA-256 stamp-file based stale check.  More reliable than mtime comparison
# across git checkouts (git sets all file timestamps to checkout time, making
# mtime-based skipping unpredictable).  A `.sha256` sidecar next to each .jl
# source records the last hash seen; Literate only re-runs when the hash differs.
function _source_changed(src::String)::Bool
    stamp = src * ".sha256"
    isfile(stamp) || return true
    read(stamp, String) == bytes2hex(sha256(read(src))) ? false : true
end
_write_stamp(src::String) = write(src * ".sha256", bytes2hex(sha256(read(src))))

# Extract base64-encoded images from Literate-generated markdown and replace them
# with on-disk file references.
#
# Why: Documenter's search indexer does not strip the content of @raw html blocks,
# so inline `data:image/png;base64,...` blobs land verbatim in search_index.js —
# each figure contributes ~100–300 KB, inflating the index to 2+ MB and making
# makedocs slow.  Saving figures as separate files shrinks the index to <200 KB
# and cuts makedocs time by 60–80 %.
#
# The pattern Literate emits for DocumenterFlavor:
#   ```@raw html
#   <img ... src="data:image/TYPE;base64, DATA">
#   ```
function _debase64_images!(md_path::String)
    content = read(md_path, String)
    dir     = dirname(md_path)
    base    = splitext(basename(md_path))[1]
    pattern = r"```@raw html\n<img[^>]*src=\"data:image/([^;]+);base64,[ \t]*([^\"]+)\"[^>]*>\n```"
    fig_n   = Ref(0)
    modified = replace(content, pattern => function(m)
        cap = match(r"data:image/([^;]+);base64,[ \t]*([^\"]+)\"", m)
        cap === nothing && return m
        mime_type = cap[1]
        b64_clean = replace(cap[2], r"\s" => "")
        ext       = mime_type == "svg+xml" ? "svg" : mime_type
        fig_n[]  += 1
        filename  = "$(base)-fig-$(fig_n[]).$(ext)"
        write(joinpath(dir, filename), base64decode(b64_clean))
        "\n![]($filename)\n"
    end)
    modified != content && write(md_path, modified)
    return modified != content
end

# ---------------------------------------------------------------------------
# Literate: exercises
# ---------------------------------------------------------------------------
const EXECUTABLE_EXERCISES = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
const exercises_dir = joinpath(@__DIR__, "src", "exercises")

if !DOCS_FAST
    for (root, dirs, files) in walkdir(exercises_dir)
        for file in files
            startswith(file, "ex_") && endswith(file, ".jl") || continue
            input_path  = joinpath(root, file)
            output_path = joinpath(root, replace(file, ".jl" => ".md"))

            _source_changed(input_path) || continue

            exercise_id = basename(root)
            executable  = exercise_id in EXECUTABLE_EXERCISES

            Literate.markdown(input_path, root;
                             documenter=true,
                             flavor=Literate.DocumenterFlavor(),
                             execute=executable)

            if !executable && isfile(output_path)
                content  = read(output_path, String)
                modified = replace(content, r"````@example ex_\d+\n" => "````julia\n")
                write(output_path, modified)
            end

            isfile(output_path) && _debase64_images!(output_path)
            _write_stamp(input_path)
        end
    end
end

# ---------------------------------------------------------------------------
# Literate: Getting Started tutorials
# ---------------------------------------------------------------------------
const GS_DIR = joinpath(@__DIR__, "src", "getting_started")

if !DOCS_FAST
    for gs_file in ("gs1_tensors_and_indices.jl", "gs2_svd_and_truncation.jl",
                    "gs3_drawing_tensor_networks.jl")
        input_path = joinpath(GS_DIR, gs_file)
        _source_changed(input_path) || continue
        Literate.markdown(input_path, GS_DIR;
                         documenter=true,
                         flavor=Literate.DocumenterFlavor(),
                         execute=true)
        md_out = joinpath(GS_DIR, replace(gs_file, ".jl" => ".md"))
        isfile(md_out) && _debase64_images!(md_out)
        _write_stamp(input_path)
    end
end

# Scan exercise notebooks and inject "Used in exercises" links into API docstrings.
# Skipped in DOCS_FAST mode — API pages still build, just without the exercise cross-links.
if !DOCS_FAST
    exercise_usage = find_notebook_symbol_usage(exercises_dir)
    exercise_pages = create_exercise_page_mapping(exercises_dir)
    @info "Found $(length(exercise_usage)) symbols used in $(length(exercise_pages)) exercises"
    inject_usage_into_module_docstrings!(Qritical, exercise_usage, exercise_pages)
end

# Preprocess markdown files to convert glossary references {glossary:term} → markdown links
# Only writes files whose content actually changes, so once all {glossary:term}
# references have been converted, reruns under `servedocs()` are no-ops and don't
# re-trigger the file watcher.
function preprocess_glossary_links(src_dir)
    for (root, dirs, files) in walkdir(src_dir)
        for file in files
            if endswith(file, ".md")
                filepath = joinpath(root, file)
                content = read(filepath, String)
                modified = glossary_link_preprocessor(content)
                if modified != content
                    write(filepath, modified)
                    @debug "Processed glossary links in $filepath"
                end
            end
        end
    end
end

src_dir = normpath(joinpath(@__DIR__, "src"))
preprocess_glossary_links(src_dir)

makedocs(;
    modules=[Qritical],
    authors="Bavithra Govintharajah",
    remotes=nothing,
    sitename="Qritical.jl",
    version=string(pkgversion(Qritical)),
    format=Documenter.HTML(;
        canonical="https://knottyanyon.github.io/Qritical.jl",
        edit_link="main",
        mathengine=MathJax3(
            Dict(
                :loader => Dict("load" => ["[tex]/physics"]),
                :tex => Dict(
                    "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                    "tags" => "ams",
                    "packages" => ["base", "ams", "autoload", "physics"],
                ),
            ),
        ),
        assets=[asset("assets/favicon.svg", class=:ico, islocal=true), "assets/custom.css", "assets/output-fold.js"],
        size_threshold=30 * 2^20,   # 30 MiB — exercise pages embed CairoMakie figures
        size_threshold_warn=5 * 2^20,
    ),
    build="build",
    workdir=normpath(joinpath(@__DIR__, "src")),
    clean=true,
    doctest=false,
    warnonly=true,
    plugins=[bib],
    pages=filter(!isnothing, [
        "Home" => "index.md",
        "Getting Started" => [
            "Overview" => "getting_started/index.md",
            "Installation" => "getting_started/installation.md",
            "Introduction: Legs & Indices" => "getting_started/introduction.md",
            "Indices and QTensor" => "getting_started/indexed_tensor.md",
            "GS-1: Tensors & Indices" => "getting_started/gs1_tensors_and_indices.md",
            "GS-2: SVD & Truncation" => "getting_started/gs2_svd_and_truncation.md",
            "GS-3: Drawing Tensor Networks" => "getting_started/gs3_drawing_tensor_networks.md",
        ],
        DOCS_FAST ? nothing : ("Exercises" => [
            "Exercise 01" => "exercises/01/ex_01.md",
            "Exercise 02" => "exercises/02/ex_02.md",
            "Exercise 03" => "exercises/03/ex_03.md",
            "Exercise 04" => "exercises/04/ex_04.md",
            "Exercise 05" => "exercises/05/ex_05.md",
            "Exercise 06" => "exercises/06/ex_06.md",
            "Exercise 07" => "exercises/07/ex_07.md",
            "Exercise 08" => "exercises/08/ex_08.md",
            "Exercise 09" => "exercises/09/ex_09.md",
            "Exercise 10" => "exercises/10/ex_10.md",
            "Exercise 11" => "exercises/11/ex_11.md",
        ]),
        "API Reference" => [
            "Index Layer" => "api/index_layer.md",
            "QTensor" => "api/qtensor.md",
            "SVD & Truncation" => "api/svd.md",
            "Spectra & Entanglement" => "api/spectrum.md",
            "State Utilities & I/O" => "api/io.md",
            "MPS & Canonical Forms" => "api/mps.md",
            "Geometry" => "api/geometry.md",
            "Degrees of Freedom" => "api/dof.md",
            "Operators & Hamiltonians" => "api/operator.md",
            "MPO & Expectation Values" => "api/mpo.md",
            "Power Method" => "api/power_method.md",
            "TEBD" => "api/tebd.md",
            "Quench & TEBD Solve" => "api/quench.md",
            "Storage Formats" => "api/storage_format.md",
            "ExactDiagonalization" => "api/ed.md",
            "ED Time Propagation" => "api/ed_time.md",
            "Disorder" => "api/disorder.md",
            "Visualisation" => "api/visualization.md",
        ],
        "References" => [
            "Glossary" => "references/glossary.md",
        ],
        # "Bibliography" => "references.md",
    ]),
)

if get(ENV, "CI", "false") == "true"
    deploydocs(; repo="github.com/knottyanyon/Qritical.jl", devbranch="main")
end
