using HalfIntegers: half

@testset "TensorIndex" begin

    # ── Direction enum: UpIndex / DownIndex ────────────────────────────

    @testset "IndexDirection enum" begin
        @test UpIndex != DownIndex
        @test flip(UpIndex) == DownIndex
        @test flip(DownIndex) == UpIndex
    end

    # ── Index structs: PhysicalIndex, BondIndex ──────────────────────────────

    @testset "BondIndex" begin
        α = BondIndex(:α, 2, 3, 4, UpIndex)
        @test α.label == :α
        @test α.from == 2
        @test α.to == 3
        @test α.dim == 4
        @test α.dir == UpIndex
        @test is_bond(α)
        @test !is_physical(α)
    end

    # ── Step 4: PhysicalIndex with AbstractSite + local_hilbert_dim ─────────

    @testset "PhysicalIndex with site" begin
        σ = PhysicalIndex(:σ, SpinSite(half(1), 1), UpIndex)
        @test σ.label == :σ
        @test σ.site === SpinSite(half(1), 1)
        @test σ.site.lattice_ordinal == 1
        @test σ.dir == UpIndex
        @test is_physical(σ)
        @test !is_bond(σ)
    end

    @testset "local_hilbert_dim on indices" begin
        @test local_hilbert_dim(PhysicalIndex(:σ, SpinSite(half(1), 1), UpIndex)) == 2
        @test local_hilbert_dim(PhysicalIndex(:σ, SpinSite(half(2), 1), DownIndex)) == 3
        @test local_hilbert_dim(
            PhysicalIndex(:n, SpinlessBosonicSite(1; n_max_occ=3), DownIndex)
        ) == 4
        @test local_hilbert_dim(BondIndex(:α, 2, 3, 16, DownIndex)) == 16
    end

    @testset "dual preserves site" begin
        σ = PhysicalIndex(:σ, SpinSite(half(1), 3), UpIndex)
        @test dual(σ).dir == DownIndex
        @test dual(σ).site === σ.site
        @test dual(dual(σ)) == σ
    end

    # ── Index algebra: dual, adjoint, direction helpers ──────────────────────

    @testset "dual and adjoint" begin
        α = BondIndex(:α, 2, 3, 4, UpIndex)
        @test dual(α).dir == DownIndex
        @test dual(α).label == α.label
        @test dual(α).from == α.from
        @test dual(α).to == α.to
        @test dual(α).dim == α.dim
        @test α' == dual(α)
        @test dual(dual(α)) == α

        σ = PhysicalIndex(:σ, SpinSite(half(1), 1), DownIndex)
        @test dual(σ).dir == UpIndex
        @test dual(σ).site === σ.site
        @test local_hilbert_dim(dual(σ)) == local_hilbert_dim(σ)
    end

    @testset "isdual" begin
        α = BondIndex(:α, 2, 3, 4, UpIndex)
        @test isdual(α, dual(α))
        @test !isdual(α, α)
        @test !isdual(α, BondIndex(:α, 2, 3, 4, UpIndex))     # same direction
        @test !isdual(α, BondIndex(:α, 2, 3, 3, DownIndex))   # different dim
        @test !isdual(α, BondIndex(:β, 2, 3, 4, DownIndex))   # different label
        @test !isdual(α, BondIndex(:α, 3, 4, 4, DownIndex))   # different bond endpoints
        σ = PhysicalIndex(:σ, SpinSite(half(3), 1), DownIndex)   # spin-3/2: dim 4
        @test !isdual(α, σ)                                 # different kind

        site1 = SpinSite(half(1), 1)
        σ_up = PhysicalIndex(:σ, site1, UpIndex)
        σ_dn = PhysicalIndex(:σ, site1, DownIndex)
        @test isdual(σ_up, σ_dn)                                                 # same site, opposite dir
        @test !isdual(σ_up, σ_up)                                                 # same dir
        # SpinSite is immutable: === is value equality, so same-field sites are ===
        @test isdual(σ_up, PhysicalIndex(:σ, SpinSite(half(1), 1), DownIndex))  # value-equal sites are ===
        @test !isdual(σ_up, PhysicalIndex(:σ, SpinSite(half(1), 2), DownIndex))  # different ordinal
        @test !isdual(σ_up, PhysicalIndex(:n, site1, DownIndex))                  # different label
    end

    @testset "as_up and as_down" begin
        α = BondIndex(:α, 2, 3, 4, UpIndex)
        @test as_down(α).dir == DownIndex
        @test as_down(dual(α)) == dual(α)   # idempotent
        @test as_up(α) == α                 # idempotent
        @test as_up(dual(α)).dir == UpIndex
    end

end
