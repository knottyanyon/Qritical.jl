using Qritical
using Test
using TensorOperations
using LinearAlgebra
using Statistics

# Pkg.test() compatibility shim: redirect @testitem to @testset so the test bodies
# actually execute when run via `Pkg.test()`. VS Code's Julia extension runs @testitem
# natively via TestItems.jl; this shim only activates in the Pkg.test() path, where
# @testitem would otherwise register tests for a runner that never fires.
#
# The project uses nested @testitem (outer = section header, inner = test case), which
# maps cleanly to nested @testset — both accept (name::String, body::Expr).
macro testitem(args...)
    return esc(:(@testset $(args...)))
end

@testset "Qritical.jl" begin
    include("test_packaging.jl")
    include("test_dof.jl")
    include("test_symmetries.jl")
    include("test_operator.jl")
    include("test_power_method.jl")
    include("test_tebd.jl")
    include("test_study.jl")
    include("test_evolution.jl")
    include("test_ed.jl")
    include("test_ed_time.jl")
    include("test_disorder.jl")
    include("test_svd.jl")
    include("test_tix.jl")
    include("test_multix.jl")
    include("test_partition.jl")
    include("test_tensor_utils.jl")
    include("test_qtensor.jl")
    include("test_bond.jl")
    include("test_ortho_center.jl")
    include("test_spectrum.jl")
    include("test_min_model_kit/test_lattice/test_lattice_graph.jl")
    include("test_io_state.jl")
    include("test_mps.jl")
    include("test_convention.jl")
    include("test_canonicalize.jl")
    include("test_vidal.jl")
    include("test_observables.jl")
    include("test_correlators.jl")
end
