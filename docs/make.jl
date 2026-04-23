using Qritical
using Documenter
using Literate

## Process Literate files
TUTORIALS = joinpath(@__DIR__, "src", "tutorials");
literate_dir = joinpath(@__DIR__, "src", "literate");
OUTPUT_ROOT = joinpath(@__DIR__, "src", "generated");

PLAYGROUND_ROOT = joinpath(@__DIR__, "src", "code_playground");
PLAYGROUND_OUTROOT = joinpath(OUTPUT_ROOT, "code_playground");

HTN_ROOT = joinpath(@__DIR__, "src", "htn-sose26");
HTN_OUTROOT = joinpath(OUTPUT_ROOT, "htn-sose26");

EXERCISES_ROOT = joinpath(HTN_ROOT, "exercises");
EXERCISES_OUTROOT = joinpath(HTN_OUTROOT, "exercises");

##
# TODO: create an object similar to y python dictionry? where I can collect the names of the generated files that can be later included in the pages argument of the makedocs function.
# Check the exercises directories
for (dpath, dirs, files) in walkdir(EXERCISES_ROOT)
    if !isempty(files)
        dirname = first(splitext(last(splitdir(dpath)))) # get the directory name only
        out_path = joinpath(EXERCISES_OUTROOT, dirname)

        # process the files in the directory

        for f in filter(f -> endswith(f, ".jl"), files)
            in_path = joinpath(dpath, f) # input path of the file to be processed
            Literate.markdown(
                in_path, out_path; flavor=Literate.DocumenterFlavor(), documenter=true
            )
        end
    end
end
##
# TODO: create an object similar to y python dictionry? where I can collect the names of the generated files that can be later included in the pages argument of the makedocs function.
# Check the code playground directory

for f in filter(f -> endswith(f, ".jl"), readdir(PLAYGROUND_ROOT))
    in_path = joinpath(PLAYGROUND_ROOT, f) # input path of the file to be processed
    @show in_path
    Literate.markdown(
        in_path, PLAYGROUND_OUTROOT; flavor=Literate.DocumenterFlavor(), documenter=true
    )
end

##
# Set up doctest 
DocMeta.setdocmeta!(Qritical, :DocTestSetup, :(using Qritical); recursive=true)

makedocs(;
    modules=[Qritical],
    authors="Bavithra Govintharajah",
    sitename="Qritical.jl",
    format=Documenter.HTML(;
        canonical="https://knottyanyon.github.io/Qritical.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Hands-on Tensor Networks" => [
            "Week 01" => "htn-sose26/notes/week_01.md",
            "Week 02" => "htn-sose26/notes/week_02.md",
        ],
        "Exercises" => [
            "01 Singular Value Decomposition" => "generated/htn-sose26/exercises/01_SVD/01_SVD.md",
            "02 Canonical Decomposition" => "generated/htn-sose26/exercises/02_Canonical_decomposition/02_Canonical_decomposition.md",
        ],
        "Julia Playground" => [
            "01 Understanding Contractions" => "generated/code_playground/01_Tensor Contractions.md",
        ],
    ],
)

deploydocs(; repo="github.com/knottyanyon/Qritical.jl", devbranch="main")
