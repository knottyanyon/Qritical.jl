using Qritical
using Test

# Aqua quality checks are deferred to v1.0 (stale_deps will fire until all
# planned dependencies are actually used).

@testset "Qritical.jl" begin
    include("test_tensor_index.jl")
    include("test_tensor_core.jl")
    include("test_site_types.jl")
    include("test_backend.jl")
    include("test_tensor_svd.jl")
    include("test_finite_mps.jl")
    include("test_finite_mpo.jl")
    include("test_tebd.jl")
end
