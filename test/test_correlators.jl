# Tests for: §5.2 two-point correlators — two_point(mps, op_i, op_j, i, j)
# Physics: ⟨ψ|O_i O_j|ψ⟩ connects measurements at two sites via transfer matrices.
# TDD Red Phase: all tests must fail before any implementation is written.

using Test, LinearAlgebra, Qritical

# ============================================================================
# Spin operators (spin-1/2: |↑⟩=[1,0], |↓⟩=[0,1])
# ============================================================================
const σz_c = [1.0  0.0; 0.0 -1.0]
const σx_c = [0.0  1.0; 1.0  0.0]
const Id_c = [1.0  0.0; 0.0  1.0]

# Néel state |↑↓↑↓⟩ for L=4 (site 1=↑, site 2=↓, site 3=↑, site 4=↓).
# Julia column-major: site 1 is the fast (LSB) index.
# |↑⟩=1 (index 1), |↓⟩=2 (index 2).
# Linear index for (σ1, σ2, σ3, σ4) = σ1 + 2*(σ2-1) + 4*(σ3-1) + 8*(σ4-1)
# Néel (1,2,1,2): 1 + 2*1 + 4*0 + 8*1 = 1 + 2 + 0 + 8 = 11 → ψ_vec[11] = 1
function neel_mps(L::Int)
    ψ_vec = zeros(2^L)
    # Build Néel pattern: site i has spin ↑ if i is odd, ↓ if i is even
    # In column-major: index = 1 + sum_{i=1}^{L} (spin_i - 1) * 2^{i-1}
    idx = 1 + sum((isodd(i) ? 0 : 1) * 2^(i - 1) for i in 1:L)
    ψ_vec[idx] = 1.0
    return to_mps(as_state(ψ_vec, fill(2, L)); trunc=NoTrunc(), form=:left)
end

function all_up_mps_c(L::Int)
    ψ_vec = zeros(2^L); ψ_vec[1] = 1.0
    return to_mps(as_state(ψ_vec, fill(2, L)); trunc=NoTrunc(), form=:left)
end

# ============================================================================
# §5.2  two_point correlator
# ============================================================================

@testset "two_point: two-site correlators" begin

    # ------------------------------------------------------------------
    # Test 1 — ⟨σᶻ_i σᶻ_{i+1}⟩ = −1 on Néel state |↑↓↑↓⟩
    # Adjacent sites have opposite spins, so σᶻ_i = +1, σᶻ_{i+1} = −1
    # → product = −1.
    # ------------------------------------------------------------------
    @testset "⟨σᶻ_i σᶻ_{i+1}⟩ = −1 on Néel |↑↓↑↓⟩ (alternating spins)" begin
        L   = 4
        mps = neel_mps(L)
        for i in 1:(L - 1)
            @test two_point(mps, σz_c, σz_c, i, i + 1) ≈ -1.0 atol=1e-10
        end
    end

    # ------------------------------------------------------------------
    # Test 2 — ⟨σᶻ_i σᶻ_{i+1}⟩ = +1 on |↑↑↑↑⟩ (parallel spins)
    # ------------------------------------------------------------------
    @testset "⟨σᶻ_i σᶻ_{i+1}⟩ = +1 on |↑↑↑↑⟩ (ferromagnetic)" begin
        L   = 4
        mps = all_up_mps_c(L)
        for i in 1:(L - 1)
            @test two_point(mps, σz_c, σz_c, i, i + 1) ≈ 1.0 atol=1e-10
        end
    end

    # ------------------------------------------------------------------
    # Test 3 — two_point(ψ, I, O, i, j) == local_expectation(ψ, O, j)
    # Inserting identity on one side reduces to a single-site observable.
    # ------------------------------------------------------------------
    @testset "two_point with identity collapses to local_expectation" begin
        L     = 5
        ψ_vec = randn(2^L); ψ_vec ./= norm(ψ_vec)
        mps   = to_mps(as_state(ψ_vec, fill(2, L)); trunc=NoTrunc(), form=:left)
        for j in 2:L
            tp  = two_point(mps, Id_c, σz_c, 1, j)
            le  = local_expectation(mps, σz_c, j)
            @test tp ≈ le atol=1e-10
        end
    end

    # ------------------------------------------------------------------
    # Test 4 — Brute-force agreement on a random state
    # ⟨ψ|σᶻ_i ⊗ σᶻ_j|ψ⟩ computed via full 2^L operator embedding.
    # as_state respects kron ordering: MPS site s → kron position s.
    # ------------------------------------------------------------------
    @testset "two_point matches full-state brute-force on random state" begin
        L     = 4
        ψ_vec = randn(2^L); ψ_vec ./= norm(ψ_vec)
        mps   = to_mps(as_state(ψ_vec, fill(2, L)); trunc=NoTrunc(), form=:left)

        for i in 1:L, j in (i + 1):L
            ops    = [k == i ? σz_c : (k == j ? σz_c : Id_c) for k in 1:L]
            O_full = foldl(kron, ops)
            expected = dot(ψ_vec, O_full * ψ_vec)
            @test two_point(mps, σz_c, σz_c, i, j) ≈ expected atol=1e-10
        end
    end

    # ------------------------------------------------------------------
    # Test 5 — Connected correlator ⟨O_i O_j⟩ − ⟨O_i⟩⟨O_j⟩ = 0
    # for a product state (sites are unentangled → no correlations).
    # ------------------------------------------------------------------
    @testset "connected correlator vanishes for a product state" begin
        L   = 5
        mps = all_up_mps_c(L)
        for i in 1:L, j in (i + 1):L
            corr      = two_point(mps, σz_c, σz_c, i, j)
            exp_i     = local_expectation(mps, σz_c, i)
            exp_j     = local_expectation(mps, σz_c, j)
            connected = corr - exp_i * exp_j
            @test abs(connected) < 1e-10
        end
    end

end
