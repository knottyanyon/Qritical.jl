using Qritical
using Test
using Aqua

@testset "Qritical.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        on_ci = get(ENV, "CI", "false") == "true"
        Aqua.test_all(Qritical; persistent_tasks=(; broken=on_ci))
    end

    include("test_tensor_index.jl")
    include("test_site_types.jl")
end
