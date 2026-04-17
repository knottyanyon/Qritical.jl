using Qritical
using Test
using Aqua

@testset "Qritical.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Qritical)
    end
    # Write your tests here.
end
