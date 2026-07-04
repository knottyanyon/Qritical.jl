using Qritical
using Documenter
using DocumenterCitations
using Literate
using JSON, Markdown

# Load custom documentation hooks
include(joinpath(@__DIR__, "symbol_docstring_injector.jl"))
include(joinpath(@__DIR__, "glossary_linker.jl"))

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:authoryear)

DocMeta.setdocmeta!(Qritical, :DocTestSetup, :(using Qritical); recursive=true)

# Process exercise notebooks with Literate.jl (convert .jl → .md for docs)
# Only regenerate when the .jl source is newer than the existing .md output.
# This matters under `servedocs()`: Literate output lands inside docs/src/, which
# LiveServer watches. Unconditionally rewriting the .md on every build makes the
# watcher see a "change" and re-trigger make.jl, which rewrites it again — an
# infinite rebuild loop. Skipping up-to-date files breaks that cycle.
#
# Most exercises reference paths/state that don't resolve cleanly in the docs
# build environment, so they default to execute=false (source shown, not run).
# Exercises listed in EXECUTABLE_EXERCISES have been verified to run standalone
# (their data dependencies exist under docs/src/exercises/data/) and are left
# executable so their `println`/plot output renders in the built docs.
const EXECUTABLE_EXERCISES = ["01"]

exercises_dir = joinpath(@__DIR__, "src", "exercises")
for (root, dirs, files) in walkdir(exercises_dir)
    for file in files
        if startswith(file, "ex_") && endswith(file, ".jl")
            input_path = joinpath(root, file)
            md_filename = replace(file, ".jl" => ".md")
            output_path = joinpath(root, md_filename)

            if isfile(output_path) && mtime(output_path) >= mtime(input_path)
                continue
            end

            exercise_id = basename(root)
            executable = exercise_id in EXECUTABLE_EXERCISES

            # Process with Literate - specify output directory explicitly
            # This generates .md directly in the exercise directory
            Literate.markdown(input_path, root;
                             documenter=true,
                             flavor=Literate.DocumenterFlavor(),
                             execute=executable)

            # Post-process: for non-executable exercises, convert @example blocks
            # to plain code blocks so Documenter never tries to run them.
            if !executable && isfile(output_path)
                content = read(output_path, String)
                modified = replace(content, r"````@example ex_\d+\n" => "````julia\n")
                write(output_path, modified)
            end
        end
    end
end

# Scan exercise notebooks and prepare symbol usage mapping
exercise_usage = find_notebook_symbol_usage(exercises_dir)
exercise_pages = create_exercise_page_mapping(exercises_dir)
@info "Found $(length(exercise_usage)) symbols used in $(length(exercise_pages)) exercises"

# Inject "Used in exercises" with links into docstrings before building docs
inject_usage_into_module_docstrings!(Qritical, exercise_usage, exercise_pages)

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
        assets=["assets/custom.css"],
        size_threshold=30 * 2^20,   # 30 MiB — exercise pages embed CairoMakie figures
        size_threshold_warn=5 * 2^20,
    ),
    build="build",
    workdir=normpath(joinpath(@__DIR__, "src")),
    clean=true,
    doctest=false,
    warnonly=true,
    plugins=[bib],
    pages=[
        "Home" => "index.md",
        "Exercises" => [
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
        ],
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
        ],
        "References" => [
            "Glossary" => "references/glossary.md",
        ],
        # "Bibliography" => "references.md",
    ],
)

if get(ENV, "CI", "false") == "true"
    deploydocs(; repo="github.com/knottyanyon/Qritical.jl", devbranch="main")
end
