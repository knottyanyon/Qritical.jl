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

include("site_types.jl")
include("tensor_index.jl")
include("tensor_core.jl")
include("schmidt_decomposition.jl")

export Bisection
export AbstractSite
export SpinSite, SpinlessFermionicSite, SpinlessHardCoreBosonicSite, SpinlessBosonicSite
export local_hilbert_dim
export IndexDirection, UpIndex, DownIndex
export flip
export AbstractIndex, PhysicalIndex, BondIndex
export is_physical, is_bond
export dual, isdual, as_up, as_down
export IndexedTensor
export kronecker_delta

export factorize_with_svd,
    validate_bipartition_indices,
    reshape_tensor_for_bipartition,
    reshape_tensor_for_bipartition!,
    get_schmidt_coefficients,
    get_entanglement_entropy

export hinton, hinton!, draw_complex_hinton, draw_svd_hinton
export setup_size_N_rand_input, contract_N_ijk

# include("contraction_utils.jl")
end # module