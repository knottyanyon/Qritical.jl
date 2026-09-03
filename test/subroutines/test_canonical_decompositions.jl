@testitem "to_mps: left and right canonical, exact reconstruction" begin
    using TensorKit
    using LinearAlgebra: norm

    L = 4
    d = 2
    V = TensorKit.ComplexSpace(d)
    arr = randn(ComplexF64, ntuple(_ -> d, L)...)
    ψtensor = TensorKit.TensorMap(arr, reduce(⊗, ntuple(_ -> V, L)) ← one(V))
    ψ = State(ψtensor)

    chain = to_mps(ψ; form=:left)
    @test length(chain.sites) == L
    @test chain isa MPState{LeftCanonical,FiniteSupport}
    @test (chain.llim, chain.rlim) == (L, L + 1)
    @test chain.orthogonality_center == L
    @test is_canonical(chain)
    @test is_gauge_fixed(chain)

    rchain = to_mps(ψ; form=:right)
    @test rchain isa MPState{RightCanonical,FiniteSupport}
    @test (rchain.llim, rchain.rlim) == (0, 1)
    @test rchain.orthogonality_center == 1
    @test is_canonical(rchain)

    @test_throws ArgumentError to_mps(ψ; form=:bogus)
end

@testitem "canonicalize: re-gauging an existing chain" begin
    using TensorKit

    L = 4
    d = 2
    V = TensorKit.ComplexSpace(d)
    arr = randn(ComplexF64, ntuple(_ -> d, L)...)
    ψtensor = TensorKit.TensorMap(arr, reduce(⊗, ntuple(_ -> V, L)) ← one(V))
    ψ = State(ψtensor)
    chain = to_mps(ψ; form=:left)

    rchain = canonicalize(chain, RightCanonicalize())
    @test is_canonical(rchain)
    @test rchain.orthogonality_center == 1

    lchain = canonicalize(rchain, LeftCanonicalize())
    @test is_canonical(lchain)
    @test lchain.orthogonality_center == L

    for k in 1:L
        mchain = canonicalize(chain, MixedCanonicalize(k))
        @test mchain isa MPState{MixedCanonical,FiniteSupport}
        @test is_canonical(mchain)
        @test mchain.orthogonality_center == k
    end
end

@testitem "to_vidal: reconstruction and is_canonical" begin
    using TensorKit
    using LinearAlgebra: norm

    L = 3
    d = 2
    V = TensorKit.ComplexSpace(d)
    arr = randn(ComplexF64, ntuple(_ -> d, L)...)
    ψtensor = TensorKit.TensorMap(arr, reduce(⊗, ntuple(_ -> V, L)) ← one(V))
    ψ = State(ψtensor)
    chain = to_mps(ψ; form=:left)

    vidal_chain = to_vidal(chain)
    @test vidal_chain isa MPState{VidalGauge,FiniteSupport}
    Γs, λs = vidal_chain.sites, vidal_chain.λs
    @test length(Γs) == L
    @test length(λs) == L - 1

    full = tensor(Γs[1])
    for i in 1:(L - 1)
        n = TensorKit.numind(full)
        full = TensorKit.permute(full, (Tuple(1:(n - 1)), (n,)))
        full = full * λs[i]
        full = full * TensorKit.permute(tensor(Γs[i + 1]), ((1,), (2, 3)))
    end
    full = TensorKit.permute(full, (Tuple(1:TensorKit.numind(full)), ()))
    arr_rec = vec(convert(Array, full))
    arr_orig = vec(convert(Array, ψtensor)) / norm(ψtensor)
    phase = (arr_orig' * arr_rec) / abs(arr_orig' * arr_rec)
    @test norm(arr_rec ./ phase .- arr_orig) < 1e-8

    rchain = to_mps(ψ; form=:right)
    @test_throws ArgumentError to_vidal(rchain)
end

@testitem "is_canonical: false after an arbitrary perturbation" begin
    using TensorKit

    L = 3
    d = 2
    V = TensorKit.ComplexSpace(d)
    arr = randn(ComplexF64, ntuple(_ -> d, L)...)
    ψtensor = TensorKit.TensorMap(arr, reduce(⊗, ntuple(_ -> V, L)) ← one(V))
    ψ = State(ψtensor)
    chain = to_mps(ψ; form=:left)
    @test is_canonical(chain)

    perturbed_tensor =
        tensor(chain.sites[1]) + randn(ComplexF64, TensorKit.space(tensor(chain.sites[1])))
    bad_sites = copy(chain.sites)
    bad_sites[1] = QProcess(
        perturbed_tensor;
        output_roles=(VirtualLeg(), PhysicalLeg()),
        input_roles=VirtualLeg(),
    )
    bad_chain = MPState(
        bad_sites,
        LeftCanonical(),
        chain.llim,
        chain.rlim,
        chain.orthogonality_center,
        chain.ε,
    )
    @test !is_canonical(bad_chain)
end

@testitem "_validate_lambda: λs only meaningful for VidalGauge, length checked" begin
    using TensorKit

    V = TensorKit.ComplexSpace(2)
    u, S, _ = svd_compact(randn(ComplexF64, (V ⊗ V) ← V))
    site = QProcess(u; output_roles=(VirtualLeg(), PhysicalLeg()), input_roles=VirtualLeg())

    @test_throws ArgumentError MPState([site, site], LeftCanonical(), 2, 3, 1, 0.0, [S])

    @test MPState([site, site], VidalGauge(), 0, 0, nothing, 0.0, [S]) isa
        MPState{VidalGauge,FiniteSupport}

    @test_throws ArgumentError MPState(
        [site, site], VidalGauge(), 0, 0, nothing, 0.0, [S, S]
    )
end
