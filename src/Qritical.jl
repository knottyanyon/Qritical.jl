module Qritical

using GeometryBasics
using LinearAlgebra
using Makie
using LaTeXStrings

export hinton, hinton!, draw_complex_hinton, draw_svd_hinton
export estimate_multiplication_cost
# Include the separate files
include("hinton_recipe.jl")
include("contraction_benchmaking.jl")
# include("contraction_utils.jl")
end # module