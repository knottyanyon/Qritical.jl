using Qritical
using Documenter
using DocumenterCitations

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:authoryear)

DocMeta.setdocmeta!(Qritical, :DocTestSetup, :(using Qritical); recursive=true)

makedocs(;
    modules=[Qritical],
    authors="Bavithra Govintharajah",
    remotes=nothing,
    sitename="Qritical.jl",
    format=Documenter.HTML(;
        canonical="https://knottyanyon.github.io/Qritical.jl",
        edit_link="main",
        mathengine=MathJax3(Dict(
            :loader => Dict("load" => ["[tex]/physics"]),
            :tex => Dict(
                "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                "tags" => "ams",
                "packages" => ["base", "ams", "autoload", "physics"],
            ),
        )),
        assets=["assets/custom.css"]),
    build="build",
    workdir=normpath(joinpath(@__DIR__, "src")),
    clean=true,
    warnonly=true,
    plugins=[bib],
    pages=[
        "Home" => "index.md",
        "Getting Started" => [
            "getting_started/index.md",
            "Installation" => "getting_started/installation.md",
            "Indices & IndexedTensor" => "getting_started/indexed_tensor.md",
            "SVD & Truncation" => "getting_started/svd_truncation.md",
        ],
        "Notation" => "notation.md",
        "Hands-on-TN" => [
            "Exercises" => [
                "Exercise 01" => [
                    "1.1 Julia install party" => "exercises/01/task_1.md",
                    "1.2 SVD a matrix"        => "exercises/01/task_2.md",
                    "1.3 SVD a state"         => "exercises/01/task_3.md",
                    "1.4 SVD an image"        => "exercises/01/task_4.md",
                    "1.5 Contractions"        => "exercises/01/task_5.md",
                ],
                "Exercise 02" => [
                    "2.1 Left canonical form"  => "exercises/02/task_1.md",
                    "2.2 Right canonical form" => "exercises/02/task_2.md",
                    "2.3 Mixed canonical form" => "exercises/02/task_3.md",
                ],
                "Exercise 03" => [
                    "3.1 From Left to Right"       => "exercises/03/task_1.md",
                    "3.2 From Right to Left"       => "exercises/03/task_2.md",
                    "3.3 Checking the Normalization" => "exercises/03/task_3.md",
                ],
                "Exercise 04" => [
                    "4.2 MPS Overlap"    => "exercises/04/task_1.md",
                    "4.3 Observables"    => "exercises/04/task_2.md",
                    "4.4 Adding MPS"     => "exercises/04/task_3.md",
                ],
                "Exercise 05" => [
                    "5.1 Vidal Notation"  => "exercises/05/task_1.md",
                    "5.2 Observables I"   => "exercises/05/task_2.md",
                    "5.3 Observables II"  => "exercises/05/task_3.md",
                    "5.4 Observables III" => "exercises/05/task_4.md",
                ],
            ],
        ],
        "Bibliography" => "references.md",
    ],
)

if get(ENV, "CI", "false") == "true"
    deploydocs(; repo="github.com/knottyanyon/Qritical.jl", devbranch="main")
end
