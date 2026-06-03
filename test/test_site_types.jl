# v0.3 physics behavior tests: AbstractDoF hierarchy and hilbert_space

@testset "AbstractDoF — hilbert_space (native mode)" begin
    @test hilbert_space(Spin{1//2}()) == 2
    @test hilbert_space(Spin{1}())    == 3
    @test hilbert_space(Spin{3//2}()) == 4
    @test hilbert_space(Fermionic())   == 2
    @test hilbert_space(HardCoreBoson()) == 2
end

@testset "AbstractDoF — type hierarchy" begin
    @test Spin{1//2}()    isa AbstractDoF
    @test Fermionic()     isa AbstractDoF
    @test HardCoreBoson() isa AbstractDoF
    @test StateSite{Spin{1//2}} <: AbstractSite
end
