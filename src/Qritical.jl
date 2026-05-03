# src/Qritical.jl
module Qritical

using GeometryBasics
using LinearAlgebra
using Makie
using LaTeXStrings
using Logging

# export estimate_multiplication_cost
# Include the separate files
include("QriticalUtils/QriticalUtils.jl")

using .QriticalUtils

# Re-export submodules
export QriticalUtils

include("hinton_recipe.jl")
include("contraction_benchmaking.jl")

include("schmidt_rank.jl")

export factorize_with_svd,
    validate_bipartition_indices,
    reshape_tensor_for_bipartition,
    reshape_tensor_for_bipartition!

export hinton, hinton!, draw_complex_hinton, draw_svd_hinton
export setup_size_N_rand_input, contract_N_ijk

# include("contraction_utils.jl")
end # module