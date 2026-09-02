# `import`/method-extension of a `using`-imported generic function only works at true top-level
# module scope, not inside a `@testset`/`@testitem` body (see test/simstudy/test_collectors.jl) -
# so the collector used below lives at the top of the file, outside any @testitem.
import Qritical: step!, finalize!

struct SpectrumEcho <: Qritical.AbstractCollector
    seen::Vector{Any}
end
SpectrumEcho() = SpectrumEcho(Any[])
step!(::Qritical.Active, c::SpectrumEcho, ctx::NamedTuple) = push!(c.seen, ctx.spectrum)
finalize!(::Qritical.Active, c::SpectrumEcho) = c.seen

@testitem "factorize_tensor dispatch" begin
    using TensorKit

    V = TensorKit.ComplexSpace(4)
    W = TensorKit.ComplexSpace(2)
    A = randn(ComplexF64, V ← W)

    U, S, Vd, ε = factorize_tensor(A, SVDFACTORIZER())
    @test ε == 0.0
    @test A ≈ U * S * Vd

    Q, R = factorize_tensor(A, QRFACTORIZER())
    @test A ≈ Q * R
    @test TensorKit.norm(TensorKit.adjoint(Q) * Q - TensorKit.id(TensorKit.domain(Q))) <
        1e-10

    B = randn(ComplexF64, W ← V)
    L, Qm = factorize_tensor(B, LQFACTORIZER())
    @test B ≈ L * Qm

    Uh, Sh, Vdh, εh = factorize_tensor(A, HasEntanglementSpectrum())
    @test Uh * Sh * Vdh ≈ A
    Q2, R2 = factorize_tensor(A, NoEntanglementSpectrum())
    @test A ≈ Q2 * R2
end

@testitem "bond_cutoff truncation" begin
    using TensorKit

    V = TensorKit.ComplexSpace(4)
    A = randn(ComplexF64, V ← V)
    U, S, Vd, ε = factorize_tensor(A, SVDFACTORIZER(); bond_cutoff=2)
    @test TensorKit.dim(TensorKit.domain(U)) == 2
    @test ε >= 0.0
end

@testitem "advance_bond! with collector and accumulator" begin
    using TensorKit

    V = TensorKit.ComplexSpace(2)
    T = randn(ComplexF64, (V ⊗ V) ← V)

    collector = SpectrumEcho()
    accumulator = QuadratureTruncationErrorAccumulator()
    site_tensor, remainder = advance_bond!(LeftRight(), T, 1; collector, accumulator)
    @test length(finalize!(collector)) == 1
    @test finalize!(collector)[1] isa SingValSpectrum
    @test finalize!(accumulator) >= 0.0

    # NoOpCollector never builds a spectrum
    site_tensor2, remainder2 = advance_bond!(LeftRight(), T, 1)
    @test site_tensor ≈ site_tensor2
end
