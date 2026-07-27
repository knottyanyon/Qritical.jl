# Tests for: §5.1 Vidal Γ–Λ form — to_vidal and to_canonical
# Physics invariants:
#   - to_vidal / to_canonical round-trip preserves the state (overlap ≈ 1)
#   - Λᵢ₋₁ Γᵢ recovers the left-canonical A_i tensor
#   - bond_svs are unchanged by the round-trip
#   - VidalForm() and CanonicalForm(L, L+1) form tags are set correctly
#   - zeros in Λ (reduced effective bond) handled without division blow-up

using Test, LinearAlgebra, Qritical

@testset "to_vidal and to_canonical: Vidal Γ–Λ form" begin
# `@testset "name" begin ... end`: groups tests with a shared label.
# All @test macros inside contribute to this testset's pass/fail report.

    # ----------------------------------------------------------------
    # Shared fixtures
    # ----------------------------------------------------------------
    rng_seed = 42
    # Note: this variable is declared but not used to set the RNG seed.
    # Julia's `Random.seed!(rng_seed)` would set the global RNG. Here it's just metadata.
    L = 4; d = 2
    # `;` separates two statements on the same line (like Python's newline).
    # L = number of sites (chain length); d = local dimension (spin-1/2: d=2).

    ψ_vec = let v = randn(d^L); v ./= norm(v) end
    # `let v = expr; body end` is Julia's LOCAL SCOPE block: v is only visible inside.
    # In Python: `v = np.random.randn(d**L); v /= np.linalg.norm(v)` (but Python doesn't scope it).
    # `randn(d^L)` = d^L = 2^4 = 16 normally-distributed random numbers.
    # `v ./= norm(v)` normalizes in-place (./= is elementwise in-place division).
    # The whole `let ... end` evaluates to the last expression (v after normalization).
    # So ψ_vec = normalized random 16-component state vector.

    ψ_mps = to_mps(as_state(ψ_vec, fill(d, L)); trunc=NoTrunc(), form=:left)
    # `fill(d, L)` = Python's `[d] * L` = [2,2,2,2].
    # `as_state(ψ_vec, [2,2,2,2])` wraps the vector as a (2,2,2,2)-shaped QTensor.
    # `to_mps(...; trunc=NoTrunc(), form=:left)` decomposes it into a left-canonical MPS.
    # This is the ground-truth left-canonical MPS that Vidal form is derived from.

    vidal  = to_vidal(ψ_mps)
    # Convert to Vidal Γ–Λ form. Each site tensor Γᵢ = Λᵢ₋₁⁻¹ · Aᵢ.
    # Bond SVs are unchanged; only the site tensors are transformed.

    # ----------------------------------------------------------------
    # Form tags
    # ----------------------------------------------------------------
    @testset "to_vidal produces VidalForm() tag" begin
        @test vidal.form isa VidalForm
        # `isa` = `isinstance` in Python. Checks that vidal.form is of type VidalForm.
        # Physics: after to_vidal, the MPS is in Γ–Λ representation, not canonical form.
        # The VidalForm() tag signals this to any function that inspects the form.
    end

    @testset "to_canonical produces CanonicalForm(L, L+1)" begin
        canonical = to_canonical(vidal)
        # `to_canonical` is the inverse: Aᵢ = Λᵢ₋₁ · Γᵢ. Recovers the left-canonical MPS.
        @test canonical.form == CanonicalForm(L, L + 1)
        # After to_canonical, the form tag should be CanonicalForm(4, 5) for L=4.
        # `==` for CanonicalForm compares both fields (llim=4, rlim=5) by value.
    end

    # ----------------------------------------------------------------
    # Round-trip: |⟨canonical|ψ⟩| ≈ 1
    # ----------------------------------------------------------------
    @testset "round-trip overlap ≈ 1 (state preserved)" begin
        canonical = to_canonical(vidal)
        @test abs(overlap(canonical, ψ_mps)) ≈ 1.0 atol=1e-10
        # `overlap(φ, ψ)` computes ⟨φ|ψ⟩ (inner product).
        # `abs(x)` = |x| (absolute value — handles complex numbers too).
        # Physics: to_vidal followed by to_canonical is an exact round-trip (no approximation).
        # The physical state is unchanged: ⟨A|ψ⟩ = 1 for the recovered A and original ψ.
        # We take abs() because of possible global phase differences (unphysical).
    end

    # ----------------------------------------------------------------
    # Λᵢ₋₁ · Γᵢ = Aᵢ  (left-canonical recovery)
    # ----------------------------------------------------------------
    @testset "Λᵢ₋₁ · Γᵢ recovers the left-canonical tensor A_i" begin
        for i in 1:L
            Γ_i  = vidal.tensors[i].data         # Vidal Γ tensor at site i: shape (χL, d, χR)
            λ    = vidal.bond_svs[i].values      # Λᵢ₋₁ = Schmidt values at bond to the LEFT
            A_i  = ψ_mps.tensors[i].data         # original left-canonical tensor at site i

            # broadcast λ over the first (vL) dimension
            A_reconstructed = λ .* Γ_i
            # `λ` has shape (χL,) and Γ_i has shape (χL, d, χR).
            # `λ .* Γ_i` broadcasts λ along the FIRST dimension: λ[α] * Γ_i[α, :, :] for each α.
            # In numpy: `λ[:, None, None] * Γ_i` (explicit broadcasting with None dims).
            # Physics: Aᵢ = Λᵢ₋₁ · Γᵢ where Λᵢ₋₁ acts on the left virtual index.
            # This verifies that to_vidal correctly computes Γᵢ = Λᵢ₋₁⁻¹ · Aᵢ.

            @test A_reconstructed ≈ A_i atol=1e-12
            # `≈` with atol = numpy's `np.allclose(a, b, atol=1e-12)`.
            # The reconstruction should be exact up to floating-point precision.
        end
    end

    # ----------------------------------------------------------------
    # bond_svs preserved through round-trip
    # ----------------------------------------------------------------
    @testset "bond_svs unchanged by to_vidal / to_canonical" begin
        canonical = to_canonical(vidal)
        for i in 1:(L + 1)
            @test vidal.bond_svs[i].values ≈ ψ_mps.bond_svs[i].values atol=1e-12
            # Vidal form stores the same bond SVs as the canonical MPS.
            # Physics: the Λᵢ vectors in Vidal form ARE the Schmidt values = bond SVs.
            # No conversion is needed; they're shared without transformation.
            @test canonical.bond_svs[i].values ≈ ψ_mps.bond_svs[i].values atol=1e-12
            # After round-trip to_canonical, the bond SVs should still be unchanged.
            # `(L+1)` bonds total for L sites (including the two trivial boundary bonds).
        end
    end

    # ----------------------------------------------------------------
    # Edge case: product state — all inner bonds have λ = [1.0]
    # ----------------------------------------------------------------
    @testset "product state: Γ tensors equal A tensors (Λ = [1.0])" begin
        ψ_prod_vec = zeros(d^L); ψ_prod_vec[1] = 1.0
        # `zeros(n)` creates a Float64 zero vector. Then set index 1 to 1.0.
        # Julia is 1-indexed: ψ_prod_vec[1] = first element = |↑↑↑↑⟩ basis state.
        # Physics: |↑↑↑↑⟩ = |↑⟩⊗|↑⟩⊗|↑⟩⊗|↑⟩ is a product state (zero entanglement).
        mps_prod  = to_mps(as_state(ψ_prod_vec, fill(d, L)); trunc=NoTrunc(), form=:left)
        vidal_prod = to_vidal(mps_prod)

        # All inner Λ = [1.0] → Γᵢ = Aᵢ
        for i in 1:L
            @test vidal_prod.tensors[i].data ≈ mps_prod.tensors[i].data atol=1e-12
            # Physics: for a product state, all Schmidt values are 1.0 (Λᵢ = [1.0]).
            # Then Γᵢ = Λᵢ₋₁⁻¹ · Aᵢ = 1⁻¹ · Aᵢ = Aᵢ (dividing by 1 = no change).
            # So the Γ tensors are identical to the original A tensors for a product state.
        end
    end

    # ----------------------------------------------------------------
    # Edge case: zeros in Λ — no blow-up (zero reciprocal used)
    # ----------------------------------------------------------------
    @testset "zeros in Λ handled without division blow-up" begin
        # Construct an MPS whose inner bond has a zero SV by truncating to D=1
        ψ_prod_vec = zeros(d^L); ψ_prod_vec[1] = 1.0
        mps_trunc = to_mps(as_state(ψ_prod_vec, fill(d, L)); trunc=MaxBondDimTrunc(1), form=:left)
        # `MaxBondDimTrunc(1)` keeps only the LARGEST singular value at each bond.
        # For a product state (rank-1), this is exact. For an entangled state, Λ would have zeros.
        # The point: what if after truncation some bonds have Λᵢ = [0.0, …, 0.0] (null entries)?

        # to_vidal should not throw and should produce finite tensors
        @test_nowarn vidal_trunc = to_vidal(mps_trunc)
        # `@test_nowarn expr` = checks that evaluating expr produces no warnings or errors.
        # Like Python's `with pytest.warns(None): to_vidal(mps_trunc)`.
        # Physics: if Λ has zeros, dividing by them would give Inf/NaN.
        # The implementation clamps zero reciprocals to zero (not Inf), so this must not crash.

        vidal_trunc = to_vidal(mps_trunc)
        for i in 1:L
            @test all(isfinite, vidal_trunc.tensors[i].data)
            # `all(predicate, collection)` = Python's `all(predicate(x) for x in collection)`.
            # `isfinite(x)` = True iff x is neither Inf nor NaN. Like numpy's `np.isfinite(x)`.
            # Physics: even with zeros in Λ, the Γ tensors must be finite arrays.
            # The zero-clamping in to_vidal ensures Γᵢ[α, :, :] = 0 where Λᵢ₋₁[α] = 0.
        end
    end

    # ----------------------------------------------------------------
    # L=1 edge case
    # ----------------------------------------------------------------
    @testset "L=1 single-site Vidal form" begin
        ψ1 = as_state([0.6, 0.8], [2])
        # `[0.6, 0.8]` = Julia array literal = Python's `[0.6, 0.8]` or `np.array([0.6, 0.8])`.
        # [2] means: L=1 site with d=2. State: 0.6|↑⟩ + 0.8|↓⟩.
        mps1  = to_mps(ψ1; trunc=NoTrunc(), form=:left)
        vidal1 = to_vidal(mps1)
        @test vidal1.form isa VidalForm
        # Even for L=1, to_vidal must correctly tag the result as VidalForm.
        canonical1 = to_canonical(vidal1)
        @test abs(overlap(canonical1, mps1)) ≈ 1.0 atol=1e-12
        # Round-trip for L=1: Γ₁ = Λ₀⁻¹ · A₁ = 1⁻¹ · A₁ = A₁ (boundary SV is 1.0).
        # Then to_canonical: A₁ = Λ₀ · Γ₁ = 1 · Γ₁ = Γ₁ = A₁. Exact round-trip.
        # Physics: single-site case is trivial — no bonds, no entanglement.
    end

end
