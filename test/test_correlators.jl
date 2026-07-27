# Tests for: §5.2 two-point correlators — two_site_op(mps, op_i, op_j, i, j)
# Physics: ⟨ψ|O_i O_j|ψ⟩ connects measurements at two sites via transfer matrices.
# TDD Red Phase: all tests must fail before any implementation is written.

using Test, LinearAlgebra, Qritical   # `using` imports exported names; multiple packages on one line 

# ============================================================================
# Spin operators (spin-1/2: |↑⟩=[1,0], |↓⟩=[0,1])
# ============================================================================
const σz_c = [1.0 0.0; 0.0 -1.0]   # `const` makes this binding immutable after assignment. σz in the convention where eigenvalues are ±1 (not ±½); note: this is 2Sz
const σx_c = [0.0 1.0; 1.0 0.0]   # Pauli σx; used for transverse correlators
const Id_c = [1.0 0.0; 0.0 1.0]   # identity matrix; used to isolate single-site contributions

# Néel state |↑↓↑↓⟩ for L=4 (site 1=↑, site 2=↓, site 3=↑, site 4=↓).
# Julia column-major: site 1 is the fast (LSB) index.
# |↑⟩=1 (index 1), |↓⟩=2 (index 2).
# Linear index for (σ1, σ2, σ3, σ4) = σ1 + 2*(σ2-1) + 4*(σ3-1) + 8*(σ4-1)
# Néel (1,2,1,2): 1 + 2*1 + 4*0 + 8*1 = 1 + 2 + 0 + 8 = 11 → ψ_vec[11] = 1
function neel_mps(L::Int)   # helper function: builds the Néel product state |↑↓↑↓…⟩ as an MPS
    ψ_vec = zeros(2^L)   # allocate Hilbert space vector, all zeros; `zeros(n)` = Python `np.zeros(n)` with Float64 dtype
    # Build Néel pattern: site i has spin ↑ if i is odd, ↓ if i is even
    # In column-major: index = 1 + sum_{i=1}^{L} (spin_i - 1) * 2^{i-1}
    idx = 1 + sum((isodd(i) ? 0 : 1) * 2^(i - 1) for i in 1:L)   # `isodd(i)` is true if i is odd. column-major index calculation: sum of (spin_i - 1) × 2^{i-1} + 1; spin_i = 1 for ↑ (odd sites), spin_i = 2 for ↓ (even sites)
    ψ_vec[idx] = 1.0   # set the Néel state coefficient to 1 (all other entries are 0)
    return to_mps(as_state(ψ_vec, fill(2, L)); trunc=NoTrunc(), form=:left)   # `fill(2, L)` = [2,2,...,2] of length L. converts to left-canonical MPS
end

function all_up_mps_c(L::Int)   # helper: ferromagnetic state |↑↑↑…↑⟩
    ψ_vec = zeros(2^L);
    ψ_vec[1] = 1.0   # index 1 = all spins up in column-major: σ_1=1,σ_2=1,...→ index 1+0+0+...=1
    return to_mps(as_state(ψ_vec, fill(2, L)); trunc=NoTrunc(), form=:left)
end

# ============================================================================
# §5.2  two_site_op correlator
# ============================================================================

@testset "two_site_op: two-site correlators" begin

    # ------------------------------------------------------------------
    # Test 1 — ⟨σᶻ_i σᶻ_{i+1}⟩ = −1 on Néel state |↑↓↑↓⟩
    # Adjacent sites have opposite spins, so σᶻ_i = +1, σᶻ_{i+1} = −1
    # → product = −1.
    # ------------------------------------------------------------------
    @testset "⟨σᶻ_i σᶻ_{i+1}⟩ = −1 on Néel |↑↓↑↓⟩ (alternating spins)" begin
        L = 4
        mps = neel_mps(L)   # Néel state as MPS
        for i in 1:(L - 1)   # loop over all adjacent pairs (1,2), (2,3), (3,4)
            @test two_site_op(mps, σz_c, σz_c, i, i + 1) ≈ -1.0 atol=1e-10   # ⟨σz_i σz_{i+1}⟩ = (+1)·(−1) = −1 (adjacent sites have opposite Néel pattern); note σz_c has eigenvalues ±1, not ±½
        end
    end

    # ------------------------------------------------------------------
    # Test 2 — ⟨σᶻ_i σᶻ_{i+1}⟩ = +1 on |↑↑↑↑⟩ (parallel spins)
    # ------------------------------------------------------------------
    @testset "⟨σᶻ_i σᶻ_{i+1}⟩ = +1 on |↑↑↑↑⟩ (ferromagnetic)" begin
        L = 4
        mps = all_up_mps_c(L)   # all-up ferromagnetic state
        for i in 1:(L - 1)
            @test two_site_op(mps, σz_c, σz_c, i, i + 1) ≈ 1.0 atol=1e-10   # ⟨σz_i σz_{i+1}⟩ = (+1)·(+1) = +1 for parallel spins
        end
    end

    # ------------------------------------------------------------------
    # Test 3 — two_site_op(ψ, I, O, i, j) == local_expectation(ψ, O, j)
    # Inserting identity on one side reduces to a single-site observable.
    # ------------------------------------------------------------------
    @testset "two_site_op with identity collapses to local_expectation" begin
        L = 5
        ψ_vec = randn(2^L);
        ψ_vec ./= norm(ψ_vec)   # random normalised vector; `/=` in-place division 
        mps = to_mps(as_state(ψ_vec, fill(2, L)); trunc=NoTrunc(), form=:left)
        for j in 2:L   # test all sites j from 2 to L
            tp = two_site_op(mps, Id_c, σz_c, 1, j)   # ⟨ψ|I₁·σz_j|ψ⟩ = ⟨ψ|σz_j|ψ⟩ (identity on left = marginalize site 1)
            le = local_expectation(mps, σz_c, j)   # single-site expectation ⟨ψ|σz_j|ψ⟩
            @test tp ≈ le atol=1e-10   # inserting identity on one leg should reduce to a single-site measurement; physics: I_i⊗O_j on an unentangled site i → just O_j
        end
    end

    # ------------------------------------------------------------------
    # Test 4 — Brute-force agreement on a random state
    # ⟨ψ|σᶻ_i ⊗ σᶻ_j|ψ⟩ computed via full 2^L operator embedding.
    # as_state respects kron ordering: MPS site s → kron position s.
    # ------------------------------------------------------------------
    @testset "two_site_op matches full-state brute-force on random state" begin
        L = 4
        ψ_vec = randn(2^L);
        ψ_vec ./= norm(ψ_vec)   # random state
        mps = to_mps(as_state(ψ_vec, fill(2, L)); trunc=NoTrunc(), form=:left)

        for i in 1:L, j in (i + 1):L   # iterate over all pairs i < j; `(i+1):L` starts from i+1 to ensure i < j
            ops = [k == i ? σz_c : (k == j ? σz_c : Id_c) for k in 1:L]   # list comprehension: operator at each site (σz at sites i,j; identity elsewhere); this builds the full 2^L-dimensional operator embedding
            O_full = foldl(kron, ops)   # `foldl(kron, [A, B, C])` = A⊗B⊗C; `foldl` = reduce left-to-right ; builds the full Kronecker product
            expected = dot(ψ_vec, O_full * ψ_vec)   # brute-force ⟨ψ|O_full|ψ⟩
            @test two_site_op(mps, σz_c, σz_c, i, j) ≈ expected atol=1e-10   # MPS transfer-matrix method must match the brute-force full-matrix calculation
        end
    end

    # ------------------------------------------------------------------
    # Test 5 — Connected correlator ⟨O_i O_j⟩ − ⟨O_i⟩⟨O_j⟩ = 0
    # for a product state (sites are unentangled → no correlations).
    # ------------------------------------------------------------------
    @testset "connected correlator vanishes for a product state" begin
        L = 5
        mps = all_up_mps_c(L)   # all-up product state — maximally unentangled
        for i in 1:L, j in (i + 1):L   # all pairs i < j
            corr = two_site_op(mps, σz_c, σz_c, i, j)   # ⟨σz_i σz_j⟩
            exp_i = local_expectation(mps, σz_c, i)   # ⟨σz_i⟩
            exp_j = local_expectation(mps, σz_c, j)   # ⟨σz_j⟩
            connected = corr - exp_i * exp_j   # connected correlator C_c(i,j) = ⟨O_i O_j⟩ − ⟨O_i⟩⟨O_j⟩
            @test abs(connected) < 1e-10   # `abs(x)` = absolute value; for a product state, sites are statistically independent → zero connected correlator; physics: entanglement is what drives connected correlations
        end
    end
end
