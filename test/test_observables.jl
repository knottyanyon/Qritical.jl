# Tests for: §4.1 single-site expectation values (local_expectation)
# Physics: expectation values ⟨ψ|O_i|ψ⟩ are the bridge between MPS and measurement.
# TDD Red Phase: all tests must fail before any implementation is written.

using Test, LinearAlgebra, Qritical

# ============================================================================
# Spin operators (spin-1/2 basis: |↑⟩ = [1,0], |↓⟩ = [0,1])
# ============================================================================
const σz = [1.0  0.0; 0.0 -1.0]
const σx = [0.0  1.0; 1.0  0.0]
const σy = [0.0 -im;  im   0.0]
const Id = [1.0  0.0; 0.0  1.0]

# Build the all-up product state |↑↑⋯↑⟩ as an MPS.
function all_up_mps(L::Int)
    ψ_vec = zeros(2^L)
    ψ_vec[1] = 1.0
    ψ = as_state(ψ_vec, fill(2, L))
    return to_mps(ψ; trunc=NoTrunc(), form=:left)
end

# Build the all-plus eigenstate |+⟩⊗L (eigenstate of σˣ with eigenvalue +1).
function all_plus_mps(L::Int)
    plus = [1.0, 1.0] / sqrt(2)
    ψ_vec = foldl(kron, fill(plus, L))
    ψ = as_state(ψ_vec, fill(2, L))
    return to_mps(ψ; trunc=NoTrunc(), form=:left)
end

# ============================================================================
# §4.1  local_expectation
# ============================================================================

@testset "local_expectation: single-site observables" begin

    # ------------------------------------------------------------------
    # Test 1 — ⟨σᶻ⟩ = +1 on every site of |↑↑↑↑⟩
    # Physical invariant: σᶻ|↑⟩ = +|↑⟩, so ⟨↑|σᶻ|↑⟩ = +1.
    # ------------------------------------------------------------------
    @testset "⟨σᶻ⟩ = +1 on all sites of |↑↑↑↑⟩" begin
        L   = 4
        mps = all_up_mps(L)
        for i in 1:L
            @test local_expectation(mps, σz, i) ≈ 1.0 atol=1e-12
        end
    end

    # ------------------------------------------------------------------
    # Test 2 — ⟨σˣ⟩ = +1 on every site of |+⟩⊗L
    # Physical invariant: σˣ|+⟩ = +|+⟩, so ⟨+|σˣ|+⟩ = +1.
    # ------------------------------------------------------------------
    @testset "⟨σˣ⟩ = +1 on all sites of |+⟩⊗L" begin
        L   = 4
        mps = all_plus_mps(L)
        for i in 1:L
            @test real(local_expectation(mps, σx, i)) ≈ 1.0 atol=1e-12
        end
    end

    # ------------------------------------------------------------------
    # Test 3 — ⟨σᶻ⟩ = −1 on every site of |↓↓↓↓⟩
    # ------------------------------------------------------------------
    @testset "⟨σᶻ⟩ = −1 on all sites of |↓↓↓↓⟩" begin
        L = 4
        ψ_vec = zeros(2^L)
        ψ_vec[end] = 1.0          # last basis state = |↓↓↓↓⟩
        mps = to_mps(as_state(ψ_vec, fill(2, L)); trunc=NoTrunc(), form=:left)
        for i in 1:L
            @test local_expectation(mps, σz, i) ≈ -1.0 atol=1e-12
        end
    end

    # ------------------------------------------------------------------
    # Test 4 — ⟨I⟩ = 1 (identity operator) for any normalised MPS
    # A normalised state must give ⟨ψ|I|ψ⟩ = 1.
    # ------------------------------------------------------------------
    @testset "⟨I⟩ = 1 for a normalised MPS (identity check)" begin
        L   = 5
        ψ_vec = randn(2^L)
        ψ_vec ./= norm(ψ_vec)
        mps = to_mps(as_state(ψ_vec, fill(2, L)); trunc=NoTrunc(), form=:left)
        for i in 1:L
            @test real(local_expectation(mps, Id, i)) ≈ 1.0 atol=1e-10
        end
    end

    # ------------------------------------------------------------------
    # Test 5 — Result matches brute-force dense calculation
    # ⟨ψ|I⊗⋯⊗O_i⊗⋯⊗I|ψ⟩ computed from the full state vector.
    # ------------------------------------------------------------------
    @testset "local_expectation matches full-state brute-force" begin
        L     = 4
        ψ_vec = randn(2^L)
        ψ_vec ./= norm(ψ_vec)
        ψ     = as_state(ψ_vec, fill(2, L))
        mps   = to_mps(ψ; trunc=NoTrunc(), form=:left)

        for site in 1:L
            # as_state respects kron ordering: site 1 is MSB (most significant),
            # so kron_pos == site.
            ops    = [i == site ? σz : Id for i in 1:L]
            O_full = foldl(kron, ops)

            expected = dot(ψ_vec, O_full * ψ_vec)
            @test local_expectation(mps, σz, site) ≈ expected atol=1e-10
        end
    end

    # ------------------------------------------------------------------
    # Test 6 — Center-site shortcut agrees with full contraction
    # For a mixed-canonical MPS with centre at site k, ⟨O_k⟩ reduces to
    # a single 3-tensor contraction (the environments collapse to I).
    # The result must equal the full left-to-right environment contraction.
    # ------------------------------------------------------------------
    @testset "center-site result equals full contraction (mixed canonical)" begin
        L   = 5
        k   = 3
        ψ_vec = randn(2^L)
        ψ_vec ./= norm(ψ_vec)
        mps = to_mps(as_state(ψ_vec, fill(2, L)); trunc=NoTrunc(), form=:left)
        mps_mixed = canonicalize(mps, BondCanonical(k))

        # Both calls must agree — full contraction vs. center shortcut.
        @test local_expectation(mps_mixed, σz, k) ≈ local_expectation(mps, σz, k) atol=1e-10
    end

    # ------------------------------------------------------------------
    # §4.1 edge cases: mismatched-length / mismatched-dim overlap errors
    # ------------------------------------------------------------------
    @testset "overlap: mismatched lengths throw ArgumentError" begin
        mps4 = all_up_mps(4)
        mps3 = all_up_mps(3)
        @test_throws ArgumentError overlap(mps4, mps3)
        @test_throws ArgumentError overlap(mps3, mps4)
    end

    @testset "overlap: mismatched physical dims throw ArgumentError" begin
        # Build a d=2 MPS and a d=3 MPS of the same length (L=2).
        ψ2 = as_state(randn(4), [2, 2])
        ψ3 = as_state(randn(9), [3, 3])
        mps2 = to_mps(ψ2; trunc=NoTrunc(), form=:left)
        mps3 = to_mps(ψ3; trunc=NoTrunc(), form=:left)
        @test_throws ArgumentError overlap(mps2, mps3)
    end

end
