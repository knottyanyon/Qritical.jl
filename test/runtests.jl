using Qritical
using Test
using Aqua

@testset "Qritical.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Qritical; persistent_tasks=false)
    end

    include("test_tensor_index.jl")
    include("test_site_types.jl")
    include("test_backend.jl")
end
