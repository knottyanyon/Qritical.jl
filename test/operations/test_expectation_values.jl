# `import`/method-extension of a `using`-imported generic function only works at true top-level
# module scope, not inside a `@testset`/`@testitem` body (see test/simstudy/test_collectors.jl) -
# so the collector used below lives at the top of the file, outside any @testitem.
import Qritical: step!, finalize!

struct SnapshotEcho <: Qritical.AbstractCollector
    seen::Vector{Any}
end
SnapshotEcho() = SnapshotEcho(Any[])
step!(::Qritical.Active, c::SnapshotEcho, ctx::NamedTuple) = push!(c.seen, ctx.snapshot)
finalize!(::Qritical.Active, c::SnapshotEcho) = c.seen

@testitem "ExpectationValueSnapshot payload" begin
    snap = ExpectationValueSnapshot(Dict(:mag => 1.0, :stag => -1.0))
    @test snap.values == Dict(:mag => 1.0, :stag => -1.0)
end

@testitem "collector plug-in: an Active collector receives the snapshot via step!" begin
    collector = SnapshotEcho()
    snap = ExpectationValueSnapshot(Dict(:mag => 1.0))
    step!(collector, (; snapshot=snap))
    @test finalize!(collector) == [snap]
end

@testitem "evaluate_expectation_value: Hamiltonian×VidalGauge and LocalObservable×VidalGauge throw when λs is unpopulated" begin
    using TensorKit

    L = 3
    d = 2
    V = TensorKit.ComplexSpace(d)
    arr = randn(ComplexF64, ntuple(_ -> d, L)...)
    ψtensor = TensorKit.TensorMap(arr, reduce(⊗, ntuple(_ -> V, L)) ← one(V))
    ψ = State(ψtensor)

    u, _, _ = svd_compact(randn(ComplexF64, (V ⊗ V) ← V))
    site = QProcess(u; output_roles=(VirtualLeg(), PhysicalLeg()), input_roles=VirtualLeg())
    vidal_chain = MPState([site, site], VidalGauge(), 0, 0, nothing, 0.0)

    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    H = Hamiltonian(AutomatonTerm[], L, V)
    obs = LocalObservable(1, sz)
    @test_throws ArgumentError evaluate_expectation_value(H, vidal_chain)
    @test_throws ArgumentError evaluate_expectation_value(obs, vidal_chain)
end

@testitem "evaluate_expectation_value: Hamiltonian/LocalObservable × VidalGauge match LeftCanonical/dense" begin
    using TensorKit

    L = 3
    d = 2
    V = TensorKit.ComplexSpace(d)
    arr = randn(ComplexF64, ntuple(_ -> d, L)...)
    ψtensor = TensorKit.TensorMap(arr, reduce(⊗, ntuple(_ -> V, L)) ← one(V))
    ψ = State(ψtensor)
    chain = to_mps(ψ; form=:left)
    vidal_chain = to_vidal(chain)

    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    obs = LocalObservable(2, sz)
    H = xxz_hamiltonian(L, V; Jxy=1.0, Jz=0.5, h=0.1)

    @test value(evaluate_expectation_value(obs, vidal_chain)) ≈
        value(evaluate_expectation_value(obs, chain))
    @test value(evaluate_expectation_value(H, vidal_chain)) ≈
        value(evaluate_expectation_value(H, chain))
end

@testitem "evaluate_expectation_value: LocalObservable×{Left,Right,Mixed}Canonical now succeed" begin
    using TensorKit

    L = 3
    d = 2
    V = TensorKit.ComplexSpace(d)
    arr = randn(ComplexF64, ntuple(_ -> d, L)...)
    ψtensor = TensorKit.TensorMap(arr, reduce(⊗, ntuple(_ -> V, L)) ← one(V))
    ψ = State(ψtensor)

    left_chain = to_mps(ψ; form=:left)
    right_chain = to_mps(ψ; form=:right)
    mixed_chain = canonicalize(left_chain, MixedCanonicalize(2))

    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    obs = LocalObservable(1, sz)
    for chain in (left_chain, right_chain, mixed_chain)
        @test evaluate_expectation_value(obs, chain) isa Scalar
    end
end

@testitem "evaluate_expectation_value: Hamiltonian×{Left,Right,Mixed}Canonical match dense ⟨ψ|H|ψ⟩" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    sx = QProcess(
        TensorMap(ComplexF64[0 1; 1 0], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    L = 2
    terms = [AutomatonTerm(-1.0, [1 => sz, 2 => sz]), AutomatonTerm(0.5, [1 => sx])]
    H = Hamiltonian(terms, L, V)

    ψtensor = TensorMap(zeros(ComplexF64, 2, 2), V ⊗ V ← one(V))
    ψtensor[1, 1] = 0.6
    ψtensor[1, 2] = 0.8im
    ψtensor[2, 1] = -0.3
    ψtensor[2, 2] = 0.1
    ψtensor = ψtensor / TensorKit.norm(ψtensor)

    left_chain = to_mps(State(ψtensor); form=:left)
    right_chain = to_mps(State(ψtensor); form=:right)
    mixed_chain = canonicalize(left_chain, MixedCanonicalize(1))

    # brute-force dense ⟨ψ|H|ψ⟩: H = -Z⊗Z + 0.5·X⊗I, built via explicit loops (s1 fastest,
    # matching TensorKit's own convert(Array,...) axis order) rather than kron/vec index games.
    d = 2
    Zm = ComplexF64[1 0; 0 -1]
    Xm = ComplexF64[0 1; 1 0]
    I2 = ComplexF64[1 0; 0 1]
    Hmat = zeros(ComplexF64, d^2, d^2)
    for s1k in 1:d, s2k in 1:d, s1b in 1:d, s2b in 1:d
        ki = s1k + (s2k - 1) * d
        qi = s1b + (s2b - 1) * d
        Hmat[ki, qi] = -Zm[s1k, s1b] * Zm[s2k, s2b] + 0.5 * Xm[s1k, s1b] * I2[s2k, s2b]
    end
    ψarr = convert(Array, ψtensor)[:, :, 1]
    ψvec = zeros(ComplexF64, d^2)
    for s1 in 1:d, s2 in 1:d
        ψvec[s1 + (s2 - 1) * d] = ψarr[s1, s2]
    end
    expected = ψvec' * Hmat * ψvec

    for chain in (left_chain, right_chain, mixed_chain)
        result = evaluate_expectation_value(H, chain)
        @test result isa Scalar
        @test value(result) ≈ expected
    end
end

@testitem "evaluate_expectation_values batch/collector plumbing" begin
    using TensorKit

    L = 3
    d = 2
    V = TensorKit.ComplexSpace(d)
    arr = randn(ComplexF64, ntuple(_ -> d, L)...)
    ψtensor = TensorKit.TensorMap(arr, reduce(⊗, ntuple(_ -> V, L)) ← one(V))
    ψ = State(ψtensor)
    chain = to_mps(ψ; form=:left)

    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    observables = Dict(
        :mag => LocalObservable(1, sz), :energy => Hamiltonian(AutomatonTerm[], L, V)
    )

    results = evaluate_expectation_values(observables, chain)
    @test results[:mag] isa Scalar
    @test results[:energy] isa Scalar

    @test RecordingTrait(NoOpCollector()) isa Inactive
end
