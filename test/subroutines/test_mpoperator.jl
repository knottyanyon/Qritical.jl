@testitem "is_canonical / is_gauge_fixed for MPOperator" begin
    using TensorKit
    import Qritical.Subroutines: _restore_site_shape

    # A genuine left-isometric bulk MPOperator site: build U isometric on the (vL,ket,bra)|(vR)
    # grouping directly (via svd_compact), then reshape into operator-style storage via the same
    # _restore_site_shape helper canonicalize's left-sweep uses internally.
    V = TensorKit.ComplexSpace(2)
    u, _, _ = svd_compact(randn(ComplexF64, (V ⊗ V ⊗ V) ← V))
    good1 = QProcess(
        _restore_site_shape(Val(2), u);
        output_roles=(VirtualLeg(), PhysicalLeg()),
        input_roles=(VirtualLeg(), PhysicalLeg()),
    )
    bad1 = QProcess(
        randn(ComplexF64, (V ⊗ V) ← (V ⊗ V));
        output_roles=(VirtualLeg(), PhysicalLeg()),
        input_roles=(VirtualLeg(), PhysicalLeg()),
    )
    site2 = QProcess(
        randn(ComplexF64, (V ⊗ V) ← (V ⊗ V));
        output_roles=(VirtualLeg(), PhysicalLeg()),
        input_roles=(VirtualLeg(), PhysicalLeg()),
    )

    left_good = MPOperator([good1, site2], LeftCanonical(), 2, 3, 2, 0.0)
    @test left_good isa MPOperator{LeftCanonical,FiniteSupport}
    @test is_canonical(left_good)
    @test is_gauge_fixed(left_good)

    left_bad = MPOperator([bad1, site2], LeftCanonical(), 2, 3, 2, 0.0)
    @test !is_canonical(left_bad)

    unknown = MPOperator([bad1, site2], UnknownGauge(), 0, 0, nothing, 0.0)
    @test !is_canonical(unknown)
    @test !is_gauge_fixed(unknown)

    vidal = MPOperator([bad1, site2], VidalGauge(), 0, 0, nothing, 0.0)
    @test is_canonical(vidal)
    @test is_gauge_fixed(vidal)
end

@testitem "MPOperator boundary validation" begin
    using TensorKit

    V = TensorKit.ComplexSpace(2)
    site = QProcess(
        randn(ComplexF64, (V ⊗ V) ← (V ⊗ V));
        output_roles=(VirtualLeg(), PhysicalLeg()),
        input_roles=(VirtualLeg(), PhysicalLeg()),
    )

    @test_throws ArgumentError MPOperator([site], LeftCanonical(), 1, 2, 5, 0.0)
    @test MPOperator([site], LeftCanonical(), 1, 2, 1, 0.0) isa
        MPOperator{LeftCanonical,FiniteSupport}
    @test MPOperator([site], LeftCanonical(), InfiniteSupport(), 1, 2, 1, 0.0) isa
        MPOperator{LeftCanonical,InfiniteSupport}
end

@testitem "canonicalize: MPOperator MixedCanonicalize round-trip" begin
    using TensorKit

    L = 3
    V = TensorKit.ComplexSpace(2)
    Vb = oneunit(V)
    raw = QProcess[]
    prev = Vb
    for i in 1:L
        nxt = i == L ? Vb : V
        t = randn(ComplexF64, (prev ⊗ V) ← (nxt ⊗ V))
        push!(
            raw,
            QProcess(
                t;
                output_roles=(VirtualLeg(), PhysicalLeg()),
                input_roles=(VirtualLeg(), PhysicalLeg()),
            ),
        )
        prev = nxt
    end
    chain = MPOperator(raw, UnknownGauge(), 0, 0, nothing, 0.0)

    for k in 1:L
        mchain = canonicalize(chain, MixedCanonicalize(k))
        @test mchain isa MPOperator{MixedCanonical,FiniteSupport}
        @test is_canonical(mchain)
        @test mchain.orthogonality_center == k
    end
end

@testitem "canonicalize: MPOperator left-canonical and perturbation" begin
    using TensorKit

    L = 3
    V = TensorKit.ComplexSpace(2)
    Vb = oneunit(V)
    raw = QProcess[]
    prev = Vb
    for i in 1:L
        nxt = i == L ? Vb : V
        t = randn(ComplexF64, (prev ⊗ V) ← (nxt ⊗ V))
        push!(
            raw,
            QProcess(
                t;
                output_roles=(VirtualLeg(), PhysicalLeg()),
                input_roles=(VirtualLeg(), PhysicalLeg()),
            ),
        )
        prev = nxt
    end
    chain = MPOperator(raw, UnknownGauge(), 0, 0, nothing, 0.0)

    lchain = canonicalize(chain, LeftCanonicalize())
    @test lchain isa MPOperator{LeftCanonical,FiniteSupport}
    @test is_canonical(lchain)
    @test lchain.orthogonality_center == L

    bad_sites = copy(lchain.sites)
    perturbed_tensor =
        tensor(lchain.sites[1]) +
        randn(ComplexF64, TensorKit.space(tensor(lchain.sites[1])))
    bad_sites[1] = QProcess(
        perturbed_tensor;
        output_roles=(VirtualLeg(), PhysicalLeg()),
        input_roles=(VirtualLeg(), PhysicalLeg()),
    )
    bad_chain = MPOperator(
        bad_sites,
        LeftCanonical(),
        lchain.llim,
        lchain.rlim,
        lchain.orthogonality_center,
        lchain.ε,
    )
    @test !is_canonical(bad_chain)
end

@testitem "to_mpo / to_choi / to_operator are not yet implemented" begin
    using TensorKit

    V = TensorKit.ComplexSpace(2)
    site = QProcess(
        randn(ComplexF64, (V ⊗ V) ← (V ⊗ V));
        output_roles=(VirtualLeg(), PhysicalLeg()),
        input_roles=(VirtualLeg(), PhysicalLeg()),
    )
    Ô = QProcess(
        randn(ComplexF64, (V ⊗ V) ← (V ⊗ V));
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )

    @test_throws ErrorException to_choi(site)
    @test_throws ErrorException to_operator(site)
    @test_throws ErrorException to_mpo(Ô)
end
