@testitem "adjoint / dagger" begin
    using TensorKit

    V = ComplexSpace(2)
    ψ = zeros(ComplexF64, V ← one(V))
    ψ[1] = 1
    s = State(ψ)

    @test dagger === adjoint
    @test is_effect(s')
    @test is_state((s')')
    @test tensor((s')') == tensor(s)
    @test outputs((s')') == outputs(s)

    t = randn(ComplexF64, V ← V)
    p = QProcess(t)
    @test tensor(p'') == tensor(p)
    @test outputs(p') == inputs(p)
    @test inputs(p') == outputs(p)
end

@testitem "sequential composition ∘" begin
    using TensorKit

    V = ComplexSpace(2)
    ψ = zeros(ComplexF64, V ← one(V))
    ψ[1] = 1
    s = State(ψ)
    e = s'

    scalar = e ∘ s
    @test scalar isa Scalar
    @test value(scalar) ≈ 1.0 + 0.0im

    W = ComplexSpace(3)
    φ = zeros(ComplexF64, W ← one(W))
    φ[1] = 1
    other = State(φ)
    @test_throws ArgumentError other' ∘ s   # mismatched leg spaces
end

@testitem "parallel composition ⊗" begin
    using TensorKit

    V = ComplexSpace(2)
    ψ = zeros(ComplexF64, V ← one(V))
    ψ[1] = 1
    φ = zeros(ComplexF64, V ← one(V))
    φ[2] = 1
    s, r = State(ψ), State(φ)

    w = s ⊗ r
    @test w isa State
    @test length(outputs(w)) == 2
    @test outputs(w) == (outputs(s)..., outputs(r)...)
end

@testitem "identity_process is a two-sided unit" begin
    using TensorKit

    # AbstractProcess has no `==` of its own (only `equal_up_to_scalar`); composition results
    # are fresh objects wrapping fresh TensorMaps, so legs are compared exactly and the
    # underlying tensor approximately (floating point matrix products, not bit-for-bit).
    procapprox(p, q) =
        outputs(p) == outputs(q) && inputs(p) == inputs(q) && tensor(p) ≈ tensor(q)

    V = ComplexSpace(2)
    ix = TIx(V)
    idp = identity_process(ix)
    t = randn(ComplexF64, V ← V)
    p = QProcess(t)

    @test procapprox(idp ∘ p, p)
    @test procapprox(p ∘ idp, p)
end

@testitem "is_isometry / is_unitary" begin
    using TensorKit

    V = ComplexSpace(4)
    W = ComplexSpace(2)
    u, _, _ = svd_compact(randn(ComplexF64, V ← W))   # a genuine isometry, not unitary (4 != 2)
    @test is_isometry(QProcess(u))
    @test !is_unitary(QProcess(u))

    q, _ = qr_compact(randn(ComplexF64, V ← V))
    @test is_isometry(QProcess(q))
    @test is_unitary(QProcess(q))

    @test !is_isometry(QProcess(randn(ComplexF64, V ← W)))
end

@testitem "governing categorical equations" begin
    using TensorKit

    procapprox(p, q) =
        outputs(p) == outputs(q) && inputs(p) == inputs(q) && tensor(p) ≈ tensor(q)

    V = ComplexSpace(2)
    W = ComplexSpace(2)
    f, g, h = (QProcess(randn(ComplexF64, V ← V)) for _ in 1:3)
    f2, g2 = (QProcess(randn(ComplexF64, W ← W)) for _ in 1:2)

    @testitem "associativity of ∘ and ⊗" begin
        @test procapprox((h ∘ g) ∘ f, h ∘ (g ∘ f))
        @test procapprox((f ⊗ f2) ⊗ g2, f ⊗ (f2 ⊗ g2))
    end

    @testitem "interchange law" begin
        @test procapprox((g ⊗ g2) ∘ (f ⊗ f2), (g ∘ f) ⊗ (g2 ∘ f2))
    end

    @testitem "dagger-functor laws" begin
        @test procapprox((g ∘ f)', f' ∘ g')
        @test procapprox((f ⊗ f2)', f' ⊗ f2')
    end
end
