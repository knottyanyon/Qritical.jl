@testitem "GaugeFreedom" begin
    @test GaugeFreedom(LeftCanonical) isa Fixed
    @test GaugeFreedom(LeftCanonical()) isa Fixed
    @test GaugeFreedom(RightCanonical) isa Fixed
    @test GaugeFreedom(MixedCanonical) isa Fixed
    @test GaugeFreedom(VidalGauge) isa Fixed
    @test GaugeFreedom(UnknownGauge) isa Free
    @test GaugeFreedom(UnknownGauge()) isa Free
end

@testitem "is_canonical / is_gauge_fixed per gauge tag" begin
    using TensorKit

    # A 2-site chain: llim=2 means only site 1 is checked (1:(llim-1) = 1:1); site 2 (the
    # orthogonality centre) is never checked, matching the legacy convention where the
    # norm-carrying/centre site is deliberately excluded.
    V = TensorKit.ComplexSpace(2)
    u, _, _ = svd_compact(randn(ComplexF64, (V ⊗ V) ← V))   # a genuine left-isometric site tensor
    good1 = QProcess(
        u; output_roles=(VirtualLeg(), PhysicalLeg()), input_roles=VirtualLeg()
    )
    bad1 = QProcess(
        randn(ComplexF64, (V ⊗ V) ← V);
        output_roles=(VirtualLeg(), PhysicalLeg()),
        input_roles=VirtualLeg(),
    )
    site2 = QProcess(
        randn(ComplexF64, (V ⊗ V) ← V);
        output_roles=(VirtualLeg(), PhysicalLeg()),
        input_roles=VirtualLeg(),
    )

    left_good = MPState([good1, site2], LeftCanonical(), 2, 3, 2, 0.0)
    @test is_canonical(left_good)
    @test is_gauge_fixed(left_good)

    left_bad = MPState([bad1, site2], LeftCanonical(), 2, 3, 2, 0.0)
    @test !is_canonical(left_bad)

    unknown = MPState([bad1, site2], UnknownGauge(), 0, 0, nothing, 0.0)
    @test !is_canonical(unknown)
    @test !is_gauge_fixed(unknown)

    vidal = MPState([bad1, site2], VidalGauge(), 0, 0, nothing, 0.0)
    @test is_canonical(vidal)   # VidalGauge's own invariants aren't rechecked here
    @test is_gauge_fixed(vidal)
end

@testitem "MPState boundary validation" begin
    using TensorKit

    V = TensorKit.ComplexSpace(2)
    site = QProcess(
        randn(ComplexF64, (V ⊗ V) ← V);
        output_roles=(VirtualLeg(), PhysicalLeg()),
        input_roles=VirtualLeg(),
    )

    @test_throws ArgumentError MPState([site], LeftCanonical(), 1, 2, 5, 0.0)   # center out of range
    @test MPState([site], LeftCanonical(), 1, 2, 1, 0.0) isa
        MPState{LeftCanonical,FiniteSupport}
    @test MPState([site], LeftCanonical(), InfiniteSupport(), 1, 2, 1, 0.0) isa
        MPState{LeftCanonical,InfiniteSupport}
end
