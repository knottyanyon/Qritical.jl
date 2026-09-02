@testitem "Propagator construction, type-parameter dispatch, and stub accessors" begin
    p_real = Propagator{RealTime}(Hamiltonian(), 0.1)
    p_imag = Propagator{ImaginaryTime}(Hamiltonian(), 0.1)

    @test p_real isa AbstractProcess
    @test p_real.hamiltonian isa Hamiltonian
    @test p_real.dt == 0.1
    @test Propagator{RealTime} !== Propagator{ImaginaryTime}
    @test typeof(p_real) !== typeof(p_imag)

    @test_throws ErrorException tensor(p_real)
    @test_throws ErrorException outputs(p_real)
    @test_throws ErrorException inputs(p_real)
end

@testitem "trotterize is stubbed" begin
    p = Propagator{RealTime}(Hamiltonian(), 0.1)
    @test_throws ErrorException trotterize(p, SuzukiTrotter())
end
