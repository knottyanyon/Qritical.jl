using Qritical
using Documenter
using Literate
using DocumenterCitations

## Process Literate files
TUTORIALS = joinpath(@__DIR__, "src", "tutorials");
PLAYGROUND_ROOT = joinpath(@__DIR__, "src", "code_playground");
# PLAYGROUND_OUTROOT = joinpath(OUTPUT_ROOT, "code_playground");

EXERCISES_ROOT = joinpath(@__DIR__, "src", "exercises");
# EXERCISES_OUTROOT = joinpath(OUTPUT_ROOT, "exercises");

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:authoryear)
##
# TODO: create an object similar to y python dictionry? where I can collect the names of the generated files that can be later included in the pages argument of the makedocs function.
# Check the exercises directories
for (dpath, dirs, files) in walkdir(EXERCISES_ROOT)
    if !isempty(files)
        dirname = first(splitext(last(splitdir(dpath)))) # get the directory name only
        # out_path = joinpath(EXERCISES_OUTROOT, dirname)
        # process the files in the directory

        for f in filter(f -> endswith(f, ".jl"), files)
            in_path = joinpath(dpath, f) # input path of the file to be processed
            println("Processing file: $dpath")
            Literate.markdown(
                in_path,
                dpath;
                flavor=Literate.DocumenterFlavor(),
                documenter=true,
                execute=true,
            )
        end
    end
end
##
# TODO: create an object similar to y python dictionry? where I can collect the names of the generated files that can be later included in the pages argument of the makedocs function.
# Check the code playground directory

# for f in filter(f -> endswith(f, ".jl"), readdir(PLAYGROUND_ROOT))
#     in_path = joinpath(PLAYGROUND_ROOT, f) # input path of the file to be processed
#     @show in_path
#     Literate.markdown(
#         in_path, PLAYGROUND_OUTROOT; flavor=Literate.DocumenterFlavor(), documenter=true
#     )
# end

##
# Set up doctest 
DocMeta.setdocmeta!(Qritical, :DocTestSetup, :(using Qritical; using HalfIntegers); recursive=true)

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
    plugins=[bib], pages=[
        "Home" => "index.md",
        "Getting Started" => [
            "getting_started/index.md",
            "Installation" => "getting_started/installation.md",
            "Backend Modes" => "getting_started/backends.md",
        ],
        # "Hands-on Tensor Networks" => [
        #     "Week 01" => "htn-sose26/notes/week_01.md",
        #     "Week 02" => "htn-sose26/notes/week_02.md",
        # ],
        "Notation" => "notation.md", "Hands-on-TN" => ["Exercises" => [
            "Exercise 01" => [
                "1.1 Julia install party" => "exercises/01/task_1.md",
                "1.2 SVD a matrix" => "exercises/01/task_2.md",
                "1.3 SVD a state" => "exercises/01/task_3.md",
                "1.4 SVD an image" => "exercises/01/task_4.md",
                "1.5 Contractions" => "exercises/01/task_5.md",
                # "1.4 Contractions" => "generated/htn-sose26/exercises/01/4_Contractions.md",
            ],
            "Exercise 02" => [
                "2.1 Left canonical form" => "exercises/02/task_1.md",
                "2.2 Right canonical form" => "exercises/02/task_2.md",
                "2.3 Mixed canonical form" => "exercises/02/task_3.md",
            ],
            "Exercise 03" => [
                "3.1 From Left to Right" => "exercises/03/task_1.md",
                "3.2 From Right to Left" => "exercises/03/task_2.md",
                "3.3 Checking the Normalization" => "exercises/03/task_3.md",
            ],
            "Exercise 04" => [
                "4.2 MPS Overlap" => "exercises/04/task_1.md",
                "4.3 Observables" => "exercises/04/task_2.md",
                "4.4 Adding MPS" => "exercises/04/task_3.md",
            ],
            "Exercise 05" => [
                "5.1 Vidal Notation" => "exercises/05/task_1.md",
                "5.2 Observables I" => "exercises/05/task_2.md",
                "5.3 Observables II" => "exercises/05/task_3.md",
                "5.4 Observables III" => "exercises/05/task_4.md",
            ],],],

        # "Julia Playground" => [
        #     "01 Understanding Contractions" => "generated/code_playground/01_Tensor Contractions.md",
        # ],
        "ADR" => [
            "adr/index.md",
            "ADR 0001 — Covariant index type system" => "adr/0001-covariant-index-type-system.md",
            "ADR 0002 — Bond index arrow orientation" => "adr/0002-bond-index-arrow-orientation.md",
            "ADR 0003 — Backend dispatch via scoped context" => "adr/0003-backend-dispatch-scoped-context.md",
            "ADR 0004 — Special tensor types" => "adr/0004-special-tensor-types.md",
        ],
        "Bibliography" => "references.md",
    ],
)

if get(ENV, "CI", "false") == "true"
    deploydocs(; repo="github.com/knottyanyon/Qritical.jl", devbranch="main")
end
