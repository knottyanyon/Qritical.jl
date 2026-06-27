using Qritical
using Test
using TensorOperations
using LinearAlgebra

# Aqua quality checks are deferred to v1.0 (stale_deps will fire until all
# planned dependencies are actually used).

@testset "Qritical.jl" begin
    # include("test_site_types.jl")
    # include("test_backend.jl")
    # include("test_finite_mpo.jl")
    # include("test_tebd.jl")
    include("test_geometry.jl")
    include("test_dof.jl")
    include("test_operator.jl")
    include("test_power_method.jl")
    include("test_tebd.jl")
    include("test_quench.jl")
    include("test_ed.jl")
    include("test_disorder.jl")
    include("test_svd.jl")
    include("test_indices.jl")
    include("test_qtensor.jl")
    include("test_spectrum.jl")
    include("test_io_state.jl")
    include("test_mps.jl")
    include("test_canonicalize.jl")
    include("test_vidal.jl")
    include("test_observables.jl")
    include("test_correlators.jl")
end
