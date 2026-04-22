using Qritical
using Documenter
using Literate

## Process Literate files
EXAMPLE = joinpath(@__DIR__, "src", "literate", "01_Tensor Contractions.jl");
TUTORIALS = joinpath(@__DIR__, "src", "tutorials");
literate_dir = joinpath(@__DIR__, "src", "literate");
OUTPUT_DIR = joinpath(@__DIR__, "src", "generated");
@__DIR__

##
# Only process if directory exists
if isdir(literate_dir)
    for file in filter(f -> endswith(f, ".jl"), readdir(literate_dir))
        input_path = joinpath(literate_dir, file)
        Literate.markdown(
            input_path, OUTPUT_DIR; flavor=Literate.DocumenterFlavor(), documenter=true
        )
    end
end

# Exercise 01
Literate.markdown(
    joinpath(TUTORIALS, "01_SVD", "01_SVD.jl"),
    OUTPUT_DIR;
    flavor=Literate.DocumenterFlavor(),
    documenter=true,
)
# Exercise 02
Literate.markdown(
    joinpath(TUTORIALS, "02_Canonical_Decomposition", "02_Canonical_Decomposition.jl"),
    OUTPUT_DIR;
    flavor=Literate.DocumenterFlavor(),
    documenter=true,
)

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
        "Exercises" => [
            "01 SVD" => "generated/01_SVD.md",
            "02 Canonical Decomposition" => "generated/02_Canonical_Decomposition.md",
        ],
        "Julia Playground" =>
            ["01 Tensor Contractions" => "generated/01_Tensor Contractions.md"],
    ],
)

deploydocs(; repo="github.com/knottyanyon/Qritical.jl", devbranch="main")
