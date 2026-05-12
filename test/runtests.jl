using Qritical
using Test
using Aqua

@testset "Qritical.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Qritical)
    end

    include("test_tensor_index.jl")
    include("test_site_types.jl")
end
