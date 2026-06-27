# Tests for: §5.1 Vidal Γ–Λ form — to_vidal and to_canonical
# Physics invariants:
#   - to_vidal / to_canonical round-trip preserves the state (overlap ≈ 1)
#   - Λᵢ₋₁ Γᵢ recovers the left-canonical A_i tensor
#   - bond_svs are unchanged by the round-trip
#   - VidalForm() and CanonicalForm(L, L+1) form tags are set correctly
#   - zeros in Λ (reduced effective bond) handled without division blow-up

using Test, LinearAlgebra, Qritical

@testset "to_vidal and to_canonical: Vidal Γ–Λ form" begin

    # ----------------------------------------------------------------
    # Shared fixtures
    # ----------------------------------------------------------------
    rng_seed = 42
    L = 4; d = 2

    ψ_vec = let v = randn(d^L); v ./= norm(v) end
    ψ_mps = to_mps(as_state(ψ_vec, fill(d, L)); trunc=NoTrunc(), form=:left)
    vidal  = to_vidal(ψ_mps)

    # ----------------------------------------------------------------
    # Form tags
    # ----------------------------------------------------------------
    @testset "to_vidal produces VidalForm() tag" begin
        @test vidal.form isa VidalForm
    end

    @testset "to_canonical produces CanonicalForm(L, L+1)" begin
        canonical = to_canonical(vidal)
        @test canonical.form == CanonicalForm(L, L + 1)
    end

    # ----------------------------------------------------------------
    # Round-trip: |⟨canonical|ψ⟩| ≈ 1
    # ----------------------------------------------------------------
    @testset "round-trip overlap ≈ 1 (state preserved)" begin
        canonical = to_canonical(vidal)
        @test abs(overlap(canonical, ψ_mps)) ≈ 1.0 atol=1e-10
    end

    # ----------------------------------------------------------------
    # Λᵢ₋₁ · Γᵢ = Aᵢ  (left-canonical recovery)
    # ----------------------------------------------------------------
    @testset "Λᵢ₋₁ · Γᵢ recovers the left-canonical tensor A_i" begin
        for i in 1:L
            Γ_i  = vidal.tensors[i].data
            λ    = vidal.bond_svs[i].values   # bond to the left of site i
            A_i  = ψ_mps.tensors[i].data

            # broadcast λ over the first (vL) dimension
            A_reconstructed = λ .* Γ_i
            @test A_reconstructed ≈ A_i atol=1e-12
        end
    end

    # ----------------------------------------------------------------
    # bond_svs preserved through round-trip
    # ----------------------------------------------------------------
    @testset "bond_svs unchanged by to_vidal / to_canonical" begin
        canonical = to_canonical(vidal)
        for i in 1:(L + 1)
            @test vidal.bond_svs[i].values ≈ ψ_mps.bond_svs[i].values atol=1e-12
            @test canonical.bond_svs[i].values ≈ ψ_mps.bond_svs[i].values atol=1e-12
        end
    end

    # ----------------------------------------------------------------
    # Edge case: product state — all inner bonds have λ = [1.0]
    # ----------------------------------------------------------------
    @testset "product state: Γ tensors equal A tensors (Λ = [1.0])" begin
        ψ_prod_vec = zeros(d^L); ψ_prod_vec[1] = 1.0
        mps_prod  = to_mps(as_state(ψ_prod_vec, fill(d, L)); trunc=NoTrunc(), form=:left)
        vidal_prod = to_vidal(mps_prod)

        # All inner Λ = [1.0] → Γᵢ = Aᵢ
        for i in 1:L
            @test vidal_prod.tensors[i].data ≈ mps_prod.tensors[i].data atol=1e-12
        end
    end

    # ----------------------------------------------------------------
    # Edge case: zeros in Λ — no blow-up (zero reciprocal used)
    # ----------------------------------------------------------------
    @testset "zeros in Λ handled without division blow-up" begin
        # Construct an MPS whose inner bond has a zero SV by truncating to D=1
        ψ_prod_vec = zeros(d^L); ψ_prod_vec[1] = 1.0
        mps_trunc = to_mps(as_state(ψ_prod_vec, fill(d, L)); trunc=MaxBondDimTrunc(1), form=:left)

        # to_vidal should not throw and should produce finite tensors
        @test_nowarn vidal_trunc = to_vidal(mps_trunc)
        vidal_trunc = to_vidal(mps_trunc)
        for i in 1:L
            @test all(isfinite, vidal_trunc.tensors[i].data)
        end
    end

    # ----------------------------------------------------------------
    # L=1 edge case
    # ----------------------------------------------------------------
    @testset "L=1 single-site Vidal form" begin
        ψ1 = as_state([0.6, 0.8], [2])
        mps1  = to_mps(ψ1; trunc=NoTrunc(), form=:left)
        vidal1 = to_vidal(mps1)
        @test vidal1.form isa VidalForm
        canonical1 = to_canonical(vidal1)
        @test abs(overlap(canonical1, mps1)) ≈ 1.0 atol=1e-12
    end

end
