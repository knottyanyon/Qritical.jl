using Qritical
using Documenter
using DocumenterCitations
using Literate
using JSON, Markdown

# Load the symbol usage injection system
include(joinpath(@__DIR__, "symbol_docstring_injector.jl"))

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:authoryear)

DocMeta.setdocmeta!(Qritical, :DocTestSetup, :(using Qritical); recursive=true)

# Process exercise notebooks with Literate.jl (convert .jl → .md for docs)
exercises_dir = joinpath(@__DIR__, "src", "exercises")
for (root, dirs, files) in walkdir(exercises_dir)
    for file in files
        if startswith(file, "ex_") && endswith(file, ".jl")
            input_path = joinpath(root, file)
            md_filename = replace(file, ".jl" => ".md")
            output_path = joinpath(root, md_filename)

            # Process with Literate (output goes to docs/ by default)
            Literate.markdown(input_path;
                             documenter=true,
                             flavor=Literate.DocumenterFlavor())

            # Move the generated .md file to the correct location
            temp_md = joinpath(@__DIR__, md_filename)
            if isfile(temp_md)
                mv(temp_md, output_path; force=true)
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
        # "Bibliography" => "references.md",
    ],
)

if get(ENV, "CI", "false") == "true"
    deploydocs(; repo="github.com/knottyanyon/Qritical.jl", devbranch="main")
end
