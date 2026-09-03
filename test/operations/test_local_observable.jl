@testitem "LocalObservable subtype relationships and stub accessors" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    obs = LocalObservable(1, sz)
    @test obs isa Observable
    @test obs isa AbstractProcess
    @test obs.ops == [1 => sz]

    @test_throws ErrorException tensor(obs)
    @test_throws ErrorException outputs(obs)
    @test_throws ErrorException inputs(obs)
end

@testitem "to_mpo is deliberately not defined for LocalObservable" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    @test_throws MethodError to_mpo(LocalObservable(1, sz))
end

@testitem "LocalObservable: multi-op constructor mirrors AutomatonTerm.ops shape" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    obs = LocalObservable([1 => sz, 10 => sz])   # long-range, non-contiguous - allowed
    @test obs.ops == [1 => sz, 10 => sz]
end

@testitem "local_expectation_value: single-site dense cross-check" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    ψtensor = TensorMap(zeros(ComplexF64, 2, 2), V ⊗ V ← one(V))
    ψtensor[1, 1] = 0.6
    ψtensor[1, 2] = 0.8im
    ψtensor[2, 1] = -0.3
    ψtensor[2, 2] = 0.1
    ψtensor = ψtensor / TensorKit.norm(ψtensor)
    mps = to_mps(State(ψtensor))

    result = local_expectation_value(mps, [1 => sz])
    @test result isa Scalar

    zdiag = [1, -1]
    ψarr = convert(Array, ψtensor)[:, :, 1]
    expected = sum(zdiag[s1] * abs2(ψarr[s1, s2]) for s1 in 1:2, s2 in 1:2)
    @test value(result) ≈ expected
end

@testitem "local_expectation_value: non-contiguous two-point correlator dense cross-check" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    ψtensor = randn(ComplexF64, (V ⊗ V ⊗ V) ← one(V))
    ψtensor = ψtensor / TensorKit.norm(ψtensor)
    mps = to_mps(State(ψtensor))

    result = local_expectation_value(mps, [1 => sz, 3 => sz])

    zdiag = [1, -1]
    ψarr = convert(Array, ψtensor)[:, :, :, 1]
    expected = sum(
        zdiag[s1] * zdiag[s3] * abs2(ψarr[s1, s2, s3]) for s1 in 1:2, s2 in 1:2, s3 in 1:2
    )
    @test value(result) ≈ expected
end
