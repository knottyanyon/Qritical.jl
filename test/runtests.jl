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
    # Everything below except the core/ tests is commented out: Qritical.jl is reduced to
    # loading only the `core` submodule for now, pending the DoF API migration. Re-enable
    # these alongside the corresponding src/Qritical.jl includes once that lands.
    # include("test_packaging.jl")
    include("core/test_core.jl")
    include("core/test_structure_traits.jl")
    include("processes/test_processes.jl")
    include("simstudy/test_simstudy.jl")
    include("subroutines/test_subroutines.jl")
    include("operations/test_operations.jl")
end
