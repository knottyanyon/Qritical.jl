@testitem "Hamiltonian subtype relationships and stub accessors" begin
    h = Hamiltonian()
    @test h isa Observable
    @test h isa AbstractProcess

    @test_throws ErrorException tensor(h)
    @test_throws ErrorException outputs(h)
    @test_throws ErrorException inputs(h)
end

@testitem "to_mpo(::Hamiltonian) is stubbed" begin
    @test_throws ErrorException to_mpo(Hamiltonian())
end
