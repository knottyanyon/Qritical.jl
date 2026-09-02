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

@testitem "evaluate_expectation_value dispatch matrix - all 8 combinations resolve" begin
    using TensorKit

    L = 3
    d = 2
    V = TensorKit.ComplexSpace(d)
    arr = randn(ComplexF64, ntuple(_ -> d, L)...)
    ψtensor = TensorKit.TensorMap(arr, reduce(⊗, ntuple(_ -> V, L)) ← one(V))
    ψ = State(ψtensor)

    left_chain = to_mps(ψ; form=:left)
    right_chain = to_mps(ψ; form=:right)
    mixed_chain = canonicalize(left_chain, SiteCanonicalize(2))

    u, _, _ = svd_compact(randn(ComplexF64, (V ⊗ V) ← V))
    site = QProcess(u; output_roles=(VirtualLeg(), PhysicalLeg()), input_roles=VirtualLeg())
    vidal_chain = MPState([site, site], VidalGauge(), 0, 0, nothing, 0.0)

    for observable in (Hamiltonian(), Correlator())
        for chain in (left_chain, right_chain, mixed_chain, vidal_chain)
            @test_throws ErrorException evaluate_expectation_value(observable, chain)
        end
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

    observables = Dict(:mag => Correlator(), :energy => Hamiltonian())

    # the batch function's own dict/collector plumbing runs; the still-stubbed
    # evaluate_expectation_value call underneath is what actually throws.
    @test_throws ErrorException evaluate_expectation_values(observables, chain)

    @test RecordingTrait(NoOpCollector()) isa Inactive
end
