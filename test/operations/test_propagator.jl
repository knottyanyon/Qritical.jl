@testitem "Propagator construction, type-parameter dispatch, and stub accessors" begin
    using TensorKit

    V = ComplexSpace(2)
    H = Hamiltonian(AutomatonTerm[], 3, V)
    p_real = Propagator{RealTime}(H, 0.1)
    p_imag = Propagator{ImaginaryTime}(H, 0.1)

    @test p_real isa AbstractProcess
    @test p_real.hamiltonian isa Hamiltonian
    @test p_real.dt == 0.1
    @test Propagator{RealTime} !== Propagator{ImaginaryTime}
    @test typeof(p_real) !== typeof(p_imag)

    @test_throws ErrorException tensor(p_real)
    @test_throws ErrorException outputs(p_real)
    @test_throws ErrorException inputs(p_real)
end

@testitem "propagator(H, dt; kind) constructs the right Propagator{T}" begin
    using TensorKit

    V = ComplexSpace(2)
    H = Hamiltonian(AutomatonTerm[], 3, V)

    p_default = propagator(H, 0.1)
    @test p_default isa Propagator{RealTime}
    @test p_default.hamiltonian === H
    @test p_default.dt == 0.1

    p_imag = propagator(H, 0.2; kind=ImaginaryTime)
    @test p_imag isa Propagator{ImaginaryTime}
    @test p_imag.dt == 0.2
end

@testitem "trotterize on an empty Hamiltonian produces an empty TrotterStep" begin
    using TensorKit

    V = ComplexSpace(2)
    H = Hamiltonian(AutomatonTerm[], 3, V)
    p = Propagator{RealTime}(H, 0.1)
    step = trotterize(p, SuzukiTrotter())
    @test isempty(step.block.gates)
    @test step.dt == 0.1
    @test step.num_steps == 1
end
