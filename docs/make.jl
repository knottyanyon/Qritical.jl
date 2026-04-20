using Qritical
using Documenter
using Literate

# Process Literate files
literate_dir = joinpath(@__DIR__, "src", "literate")
output_dir = joinpath(@__DIR__, "src", "generated")

# Only process if directory exists
if isdir(literate_dir)
    for file in readdir(literate_dir; match=r"\.jl$")
        input_path = joinpath(literate_dir, file)
        Literate.markdown(input_path; flavor=Documenter(), documenter=true)
    end
end

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
    pages=["Home" => "index.md", "Tutorials" => ["Example" => "generated/example.md"]],
)

deploydocs(; repo="github.com/knottyanyon/Qritical.jl", devbranch="main")
