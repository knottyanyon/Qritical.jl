@testitem "Correlator subtype relationships and stub accessors" begin
    c = Correlator()
    @test c isa Observable
    @test c isa AbstractProcess

    @test_throws ErrorException tensor(c)
    @test_throws ErrorException outputs(c)
    @test_throws ErrorException inputs(c)
end

@testitem "to_mpo is deliberately not defined for Correlator" begin
    @test_throws MethodError to_mpo(Correlator())
end
