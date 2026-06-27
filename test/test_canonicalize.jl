# Tests for: §3.1 (CanonicalizeSweep iterator) and §3.2 (canonical_error / is_canonical)
# Physics: canonical form is the gauging discipline that makes MPS contraction O(L).
# TDD Red Phase: all tests must fail before any implementation is written.

using Test, LinearAlgebra, Qritical

# ============================================================================
# Helpers shared across tests
# ============================================================================

# Builds a random normalised L-site spin-1/2 state and converts to MPS.
function random_spin_half_mps(L::Int; trunc=NoTrunc(), form=:left)
    ψ_vec = randn(2^L)
    ψ_vec ./= norm(ψ_vec)
    ψ = as_state(ψ_vec, fill(2, L))
    return to_mps(ψ; trunc, form)
end

# Left-isometry error for a single site tensor A with shape (χL, d, χR):
#   err = ‖A†A − I‖_F     (should be ≈ 0 for left-canonical)
function _left_isometry_error(A::AbstractArray{<:Number,3})
    χL, d, χR = size(A)
    M = reshape(A, χL * d, χR)
    return norm(M' * M - I(χR))
end

# Right-isometry error for a site tensor B with shape (χL, d, χR):
#   err = ‖BB† − I‖_F     (should be ≈ 0 for right-canonical)
function _right_isometry_error(B::AbstractArray{<:Number,3})
    χL, d, χR = size(B)
    M = reshape(B, χL, d * χR)
    return norm(M * M' - I(χL))
end

# ============================================================================
# §3.2  canonical_error / is_canonical
# ============================================================================

@testset "canonical_error and is_canonical" begin

    # ------------------------------------------------------------------
    # Test 1 — canonical_error ≈ 0 on a left-canonical tensor
    # Each A_i (sites 1..L-1) from to_mps(form=:left) satisfies A†A ≈ I.
    # canonical_error(A_i) must agree with our ground-truth helper.
    # ------------------------------------------------------------------
    @testset "canonical_error is ~0 for left-canonical site tensors" begin
        mps = random_spin_half_mps(4; form=:left)
        L = length(mps.tensors)

        for i in 1:(L - 1)
            A_i = mps.tensors[i].data
            @test canonical_error(A_i) < 1e-10
        end
    end

    # ------------------------------------------------------------------
    # Test 2 — canonical_error grows after a deliberate perturbation
    # Adding noise to a canonical tensor breaks isometry.
    # ------------------------------------------------------------------
    @testset "canonical_error grows after perturbation" begin
        mps = random_spin_half_mps(4; form=:left)
        A1 = mps.tensors[1].data

        ε_before = canonical_error(A1)
        noise = 0.5 * randn(size(A1)...)
        ε_after = canonical_error(A1 + noise)

        @test ε_before < 1e-10
        @test ε_after > 1e-3          # perturbation must register
    end

    # ------------------------------------------------------------------
    # Test 3 — is_canonical returns true for a fully left-canonical MPS
    # to_mps(form=:left) should satisfy is_canonical(ψ) = true.
    # ------------------------------------------------------------------
    @testset "is_canonical → true for to_mps output (left)" begin
        mps = random_spin_half_mps(5; form=:left)
        @test is_canonical(mps)
    end

    # ------------------------------------------------------------------
    # Test 4 — is_canonical returns true for a fully right-canonical MPS
    # ------------------------------------------------------------------
    @testset "is_canonical → true for to_mps output (right)" begin
        mps = random_spin_half_mps(5; form=:right)
        @test is_canonical(mps)
    end

    # ------------------------------------------------------------------
    # Test 5 — is_canonical returns false after perturbing a site tensor
    # Replacing one tensor with noise breaks the isometry condition.
    # ------------------------------------------------------------------
    @testset "is_canonical → false after perturbing a site tensor" begin
        mps = random_spin_half_mps(4; form=:left)
        t = mps.tensors[2]
        noisy = QTensor(t.data + 0.5 * randn(size(t.data)...), t.indices)
        broken = FiniteMPS(
            [mps.tensors[1], noisy, mps.tensors[3], mps.tensors[4]],
            mps.bond_svs,
            mps.form,
            mps.ε,
        )
        @test !is_canonical(broken)
    end
end

# ============================================================================
# §3.1  CanonicalizeConfig + canonicalize
# ============================================================================

@testset "CanonicalizeConfig and canonicalize" begin

    # ------------------------------------------------------------------
    # Test 6 — LeftCanonical() config produces left-canonical form
    # canonicalize(mps, LeftCanonical()) must tag the result with
    # CanonicalForm(L, L+1) and satisfy is_canonical.
    # ------------------------------------------------------------------
    @testset "canonicalize with LeftCanonical() produces left-canonical MPS" begin
        mps = random_spin_half_mps(4; form=:right)   # start non-left
        result = canonicalize(mps, LeftCanonical())
        L = length(result.tensors)

        @test result.form == CanonicalForm(L, L + 1)
        @test is_canonical(result)
    end

    # ------------------------------------------------------------------
    # Test 7 — RightCanonical() config produces right-canonical form
    # ------------------------------------------------------------------
    @testset "canonicalize with RightCanonical() produces right-canonical MPS" begin
        mps = random_spin_half_mps(4; form=:left)    # start non-right
        result = canonicalize(mps, RightCanonical())

        @test result.form == CanonicalForm(0, 1)
        @test is_canonical(result)
    end

    # ------------------------------------------------------------------
    # Test 8 — double sweep (L→R then R→L) preserves the quantum state
    # Physical invariant: ⟨ψ_final|ψ_original⟩ ≈ ‖ψ_original‖² = 1
    # for a normalised input (since we preserved normalisation).
    # ------------------------------------------------------------------
    @testset "L→R then R→L sweep preserves the state" begin
        ψ_vec = randn(2^4)
        ψ_vec ./= norm(ψ_vec)
        ψ = as_state(ψ_vec, fill(2, 4))
        mps = to_mps(ψ; trunc=NoTrunc(), form=:left)

        # Two-sweep round-trip
        mps_r = canonicalize(mps, RightCanonical())
        mps_l = canonicalize(mps_r, LeftCanonical())

        # Overlap ⟨final|original⟩ must equal ⟨original|original⟩ = 1.
        ov = overlap(mps_l, mps)
        @test abs(ov) ≈ 1.0 atol=1e-10
    end

    # ------------------------------------------------------------------
    # Test 9 — canonicalize on an already-canonical MPS is a near-no-op
    # Applying LeftCanonical() to an already left-canonical MPS should
    # return ε ≈ 0 and keep is_canonical true.
    # ------------------------------------------------------------------
    @testset "canonicalize is near-no-op on already-canonical input" begin
        mps = random_spin_half_mps(4; form=:left)
        result = canonicalize(mps, LeftCanonical())

        @test result.ε ≈ 0.0 atol=1e-12
        @test is_canonical(result)
    end

    # ------------------------------------------------------------------
    # Test 10 — BondCanonical(k) produces mixed canonical form:
    #   sites 1..k-1 are left-canonical,  sites k+1..L are right-canonical.
    # ------------------------------------------------------------------
    @testset "BondCanonical(k) gives mixed canonical form" begin
        L = 5
        k = 3
        mps = random_spin_half_mps(L; form=:left)
        result = canonicalize(mps, BondCanonical(k))

        # form tag must target bond k
        @test result.form == CanonicalForm(k, k + 1)

        # Sites left of k must be left-isometric
        for i in 1:(k - 1)
            @test _left_isometry_error(result.tensors[i].data) < 1e-10
        end

        # Sites right of k must be right-isometric
        for i in (k + 1):L
            @test _right_isometry_error(result.tensors[i].data) < 1e-10
        end
    end

    # ------------------------------------------------------------------
    # Test 11 — truncating sweep accumulates ε monotonically
    # After a truncating canonicalize, mps.ε ≥ 0 and equals the sum of
    # per-bond truncation errors (same invariant as to_mps).
    # ------------------------------------------------------------------
    @testset "truncating canonicalize accumulates ε from per-bond errors" begin
        mps = random_spin_half_mps(5; form=:right)
        result = canonicalize(mps, LeftCanonical(MaxBondDimTrunc(2)))

        per_bond_ε = sum(result.bond_svs[i].ε for i in 2:(length(result.bond_svs) - 1))
        @test result.ε ≥ 0.0
        @test result.ε ≈ per_bond_ε atol=1e-14
    end
end



# **Left-canonical** means every site tensor *except the last* is a left-isometry:

# ```
# A_i† A_i = I    for i = 1, …, L-1
# ```

# The **norm (orthogonality center) sits at the rightmost site** `A_L`, which is *not* constrained to be isometric — it carries the full norm of the state.

# **Right-canonical** is the mirror: every site except the first satisfies `B_i B_i† = I`, and the norm sits at site 1.

# **Mixed canonical at site k** is what actually places the orthogonality center at a specific site:
# - Sites `1, …, k-1` are left-isometric (A-tensors)  
# - Sites `k+1, …, L` are right-isometric (B-tensors)
# - Site `k` is the **orthogonality center** — it carries the norm, and expectation values `⟨O_k⟩` reduce to a single-site contraction there

# So:
# | Form | Ortho center | `CanonicalForm` tag |
# |------|-------------|---------------------|
# | Left-canonical | site `L` (rightmost) | `CanonicalForm(L, L+1)` |
# | Right-canonical | site `1` (leftmost) | `CanonicalForm(0, 1)` |
# | Mixed at site `k` | site `k` | `CanonicalForm(k, k)` |

# The intuition: in a left sweep, you push the "weight" rightward bond by bond (each SVD makes the left factor isometric), so it piles up at the right end. The name "left-canonical" refers to *which tensors are left-isometric*, not where the center lives.
