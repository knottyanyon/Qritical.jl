# Tests for: §3.1 (CanonicalizeSweep iterator) and §3.2 (canonical_error / is_canonical)
# Physics: canonical form is the gauging discipline that makes MPS contraction O(L).
# TDD Red Phase: all tests must fail before any implementation is written.

using Test, LinearAlgebra, Qritical

# Helpers shared across tests

# Builds a random normalised L-site spin-1/2 state and converts to MPS.
function random_spin_half_mps(L::Int; trunc=NoTrunc(), form=:left)
    # `L::Int` = type annotation: L must be an integer. Julia will throw a MethodError otherwise.
    # `; trunc=NoTrunc(), form=:left` = keyword arguments with default values.
    # In Python: `def random_spin_half_mps(L: int, *, trunc=NoTrunc(), form=':left'):`
    ψ_vec = randn(2^L)
    # `randn(n)` = numpy's `np.random.randn(n)` — vector of n standard normal random numbers.
    # `2^L` = 2**L in Python. For L=4: 16 components (2⁴-dimensional Hilbert space).
    ψ_vec ./= norm(ψ_vec)
    # Normalize in-place: `./=` broadcasts the division across all elements.
    # `norm(v)` = ||v||₂ (Euclidean norm). Like numpy's `np.linalg.norm(v)`.
    # After this: ||ψ_vec||₂ = 1 (unit norm = normalized quantum state).
    ψ = as_state(ψ_vec, fill(2, L))
    # `fill(2, L)` = Python's `[2] * L` — vector of L twos (all spins are spin-1/2, d=2).
    # `as_state(vector, dims)` wraps the flat vector as a QTensor with physical shape (2,2,...,2).
    return to_mps(ψ; trunc, form)
    # Note the SHORTHAND keyword syntax: `trunc` (not `trunc=trunc`). Julia 1.5+ feature.
    # In Python there's no equivalent; you must write `to_mps(ψ, trunc=trunc, form=form)`.
end

# Left-isometry error for a single site tensor A with shape (χL, d, χR):
#   err = ‖A†A − I‖_F     (should be ≈ 0 for left-canonical)
function _left_isometry_error(A::AbstractArray{<:Number,3})
    # `AbstractArray{<:Number,3}` constrains A to be a 3D array of any numeric type.
    # `<:Number` means "a subtype of Number" — works for Float64, Complex128, etc.
    χL, d, χR = size(A)
    M = reshape(A, χL * d, χR)
    # "Left unfolding": group (χL, d) → rows, keep χR as columns.
    # Physics: left-canonical means U†U=I, i.e., the columns of M are orthonormal.
    return norm(M' * M - I(χR))
    # `M' * M` = (χR×χL*d) × (χL*d×χR) → (χR×χR). Should equal I(χR) if left-isometric.
    # `I(χR)` = χR×χR identity matrix (from LinearAlgebra module).
    # `norm(...)` = Frobenius norm = sqrt(Σ|aᵢⱼ|²). Zero iff M has orthonormal columns.
end

# Right-isometry error for a site tensor B with shape (χL, d, χR):
#   err = ‖BB† − I‖_F     (should be ≈ 0 for right-canonical)
function _right_isometry_error(B::AbstractArray{<:Number,3})
    # Mirror of _left_isometry_error for right-canonical tensors.
    χL, d, χR = size(B)
    M = reshape(B, χL, d * χR)
    # "Right unfolding": keep χL as rows, group (d, χR) → columns.
    # Physics: right-canonical means MM†=I, i.e., the rows of M are orthonormal.
    return norm(M * M' - I(χL))
    # `M * M'` = (χL×d*χR) × (d*χR×χL) → (χL×χL). Should equal I(χL) if right-isometric.
end

# §3.2  canonical_error / is_canonical

@testitem "canonical_error and is_canonical" begin

    # ------------------------------------------------------------------
    # Test 1 — canonical_error ≈ 0 on a left-canonical tensor
    # Each A_i (sites 1..L-1) from to_mps(form=:left) satisfies A†A ≈ I.
    # canonical_error(A_i) must agree with our ground-truth helper.
    # ------------------------------------------------------------------
    @testitem "canonical_error is ~0 for left-canonical site tensors" begin
        mps = random_spin_half_mps(4; form=:left)
        # Calling with keyword arg: `random_spin_half_mps(4; form=:left)`.
        # The `;` is required in Julia when passing keyword args separately from positional args.
        L = length(mps.tensors)

        for i in 1:(L - 1)
            A_i = mps.tensors[i].data   # raw array of the i-th site tensor
            @test canonical_error(A_i) < 1e-10
            # `canonical_error` checks ||A†A - I||_F. Should be near machine precision.
            # Physics: left-canonical sites (all but the last) satisfy A†A = I exactly
            # in exact arithmetic; numerically, errors are at floating-point level (~1e-14).
        end
    end

    # ------------------------------------------------------------------
    # Test 2 — canonical_error grows after a deliberate perturbation
    # Adding noise to a canonical tensor breaks isometry.
    # ------------------------------------------------------------------
    @testitem "canonical_error grows after perturbation" begin
        mps = random_spin_half_mps(4; form=:left)
        A1 = mps.tensors[1].data   # extract first site tensor's data

        ε_before = canonical_error(A1)   # should be ≈ 0 (left-canonical)
        noise = 0.5 * randn(size(A1)...)
        # `size(A1)` = Tuple of dimensions, e.g. (1, 2, 2).
        # `size(A1)...` unpacks the tuple: `randn(1, 2, 2)` — makes a noise array of the same shape.
        # `0.5 * randn(...)` = random noise with std=0.5. Like numpy's `0.5 * np.random.randn(...)`.
        ε_after = canonical_error(A1 + noise)
        # `A1 + noise` adds elementwise: like numpy's `A1 + noise`. Not in-place (creates new array).

        @test ε_before < 1e-10          # before: nearly exact isometry
        @test ε_after > 1e-3            # after: perturbation must register
        # Physics: isometry is fragile — adding even moderate noise destroys A†A=I.
        # The error grows from ~1e-14 to O(1) after noise of amplitude ~0.5.
        # This test verifies that canonical_error is sensitive to deviations.
    end

    # ------------------------------------------------------------------
    # Test 3 — is_canonical returns true for a fully left-canonical MPS
    # to_mps(form=:left) should satisfy is_canonical(ψ) = true.
    # ------------------------------------------------------------------
    @testitem "is_canonical → true for to_mps output (left)" begin
        mps = random_spin_half_mps(5; form=:left)
        @test is_canonical(mps)
        # `is_canonical` checks all sites according to mps.form.
        # For CanonicalForm(5, 6): checks sites 1..4 for left-isometry.
        # Physics: to_mps produces a correctly left-canonical MPS, so is_canonical must return true.
    end

    # ------------------------------------------------------------------
    # Test 4 — is_canonical returns true for a fully right-canonical MPS
    # ------------------------------------------------------------------
    @testitem "is_canonical → true for to_mps output (right)" begin
        mps = random_spin_half_mps(5; form=:right)
        @test is_canonical(mps)
        # For CanonicalForm(0, 1): checks sites 1..5 for right-isometry.
        # Physics: all sites satisfy BB†=I in a right-canonical MPS.
    end

    # ------------------------------------------------------------------
    # Test 5 — is_canonical returns false after perturbing a site tensor
    # Replacing one tensor with noise breaks the isometry condition.
    # ------------------------------------------------------------------
    @testitem "is_canonical → false after perturbing a site tensor" begin
        mps = random_spin_half_mps(4; form=:left)
        t = mps.tensors[2]   # pick the second site tensor
        noisy = QTensor(t.data + 0.5 * randn(size(t.data)...), t.indices)
        # `t.data + 0.5 * randn(size(t.data)...)`: add noise to the raw data array.
        # `t.indices` keeps the same index metadata (the noise only affects the data, not the indices).
        # `QTensor(data, indices)` wraps the noisy data in a new QTensor.
        broken = FiniteMPS(
            [mps.tensors[1], noisy, mps.tensors[3], mps.tensors[4]],
            mps.bond_svs,
            mps.form,   # the form tag STILL says CanonicalForm(4,5) — a lie after perturbation
            mps.ε,
        )
        # Constructing a FiniteMPS with a perturbed tensor but the same form tag.
        # Array literal: `[a, b, c, d]` creates a Vector{QTensor}.
        # The form tag is misleading (says canonical but isn't) — is_canonical must detect this.
        @test !is_canonical(broken)
        # `!x` = `not x` in Python. The `!` is the boolean NOT operator.
        # Physics: is_canonical checks each site tensor; the perturbed site will fail the isometry test.
    end
end

# §3.1  CanonicalizeConfig + canonicalize

@testitem "CanonicalizeConfig and canonicalize" begin

    # ------------------------------------------------------------------
    # Test 6 — LeftCanonical() config produces left-canonical form
    # canonicalize(mps, LeftCanonical()) must tag the result with
    # CanonicalForm(L, L+1) and satisfy is_canonical.
    # ------------------------------------------------------------------
    @testitem "canonicalize with LeftCanonical() produces left-canonical MPS" begin
        mps = random_spin_half_mps(4; form=:right)   # start non-left
        # We start with a RIGHT-canonical MPS to make the test non-trivial.
        # canonicalize must re-sweep and produce a LEFT-canonical MPS.
        result = canonicalize(mps, LeftCanonical())
        # `LeftCanonical()` uses the outer constructor (no-arg) which sets trunc=NoTrunc().
        L = length(result.tensors)

        @test result.form == CanonicalForm(L, L + 1)
        # After a full left sweep: llim=L, rlim=L+1.
        # `==` compares CanonicalForm structs field by field (value equality).
        @test is_canonical(result)
        # Physical sanity check: the form tag must agree with the actual tensor data.
    end

    # ------------------------------------------------------------------
    # Test 7 — RightCanonical() config produces right-canonical form
    # ------------------------------------------------------------------
    @testitem "canonicalize with RightCanonical() produces right-canonical MPS" begin
        mps = random_spin_half_mps(4; form=:left)    # start non-right
        result = canonicalize(mps, RightCanonical())

        @test result.form == CanonicalForm(0, 1)
        # After full right sweep: llim=0 (no left-canonical sites), rlim=1 (all right-canonical).
        @test is_canonical(result)
    end

    # ------------------------------------------------------------------
    # Test 8 — double sweep (L→R then R→L) preserves the quantum state
    # Physical invariant: ⟨ψ_final|ψ_original⟩ ≈ ‖ψ_original‖² = 1
    # for a normalised input (since we preserved normalisation).
    # ------------------------------------------------------------------
    @testitem "L→R then R→L sweep preserves the state" begin
        ψ_vec = randn(2^4)
        ψ_vec ./= norm(ψ_vec)   # normalize: ||ψ_vec||₂ = 1
        ψ = as_state(ψ_vec, fill(2, 4))
        mps = to_mps(ψ; trunc=NoTrunc(), form=:left)

        # Two-sweep round-trip
        mps_r = canonicalize(mps, RightCanonical())    # left → right canonical
        mps_l = canonicalize(mps_r, LeftCanonical())   # right → left canonical again

        # Overlap ⟨final|original⟩ must equal ⟨original|original⟩ = 1.
        ov = overlap(mps_l, mps)
        # `overlap(ψ, φ)` computes ⟨ψ|φ⟩ — the inner product of two MPS.
        @test abs(ov) ≈ 1.0 atol=1e-10
        # `abs(ov)` = |⟨ψ_final|ψ_original⟩|. We take absolute value because canonicalization
        # can introduce a global phase (the state is the same up to an overall phase factor).
        # Physics: canonicalization is a gauge transformation — it changes the individual tensors
        # but not the physical state. The overlap (inner product) must be invariant.
    end

    # ------------------------------------------------------------------
    # Test 9 — canonicalize on an already-canonical MPS is a near-no-op
    # Applying LeftCanonical() to an already left-canonical MPS should
    # return ε ≈ 0 and keep is_canonical true.
    # ------------------------------------------------------------------
    @testitem "canonicalize is near-no-op on already-canonical input" begin
        mps = random_spin_half_mps(4; form=:left)
        result = canonicalize(mps, LeftCanonical())

        @test result.ε ≈ 0.0 atol=1e-12
        # Physics: re-canonicalizing an already-canonical MPS shouldn't lose any precision.
        # The singular values are already sorted and exact (up to floating-point), so
        # re-sweeping with NoTrunc should discard zero singular values only → ε ≈ 0.
        @test is_canonical(result)
    end

    # ------------------------------------------------------------------
    # Test 10 — BondCanonical(k) produces mixed canonical form:
    #   sites 1..k-1 are left-canonical,  sites k+1..L are right-canonical.
    # ------------------------------------------------------------------
    @testitem "BondCanonical(k) gives mixed canonical form" begin
        L = 5
        k = 3   # orthogonality centre at site 3
        mps = random_spin_half_mps(L; form=:left)
        result = canonicalize(mps, BondCanonical(k))
        # BondCanonical(3): left sweep up to site 2, right sweep from site 4 down to 5.
        # Site 3 holds the gauge weight (the singular values).

        # form tag must target bond k
        @test result.form == CanonicalForm(k, k + 1)
        # CanonicalForm(3, 4): sites 1..2 left-canonical, sites 4..5 right-canonical.

        # Sites left of k must be left-isometric
        for i in 1:(k - 1)
            @test _left_isometry_error(result.tensors[i].data) < 1e-10
            # Sites 1 and 2 should satisfy A†A = I (left-canonical).
        end

        # Sites right of k must be right-isometric
        for i in (k + 1):L
            @test _right_isometry_error(result.tensors[i].data) < 1e-10
            # Sites 4 and 5 should satisfy BB† = I (right-canonical).
            # `(k+1):L` = Python's `range(k+1, L+1)` = [4, 5] for k=3, L=5.
        end
    end

    # ------------------------------------------------------------------
    # Test 11 — truncating sweep accumulates ε in quadrature
    # After a truncating canonicalize, mps.ε ≥ 0 and equals the quadrature
    # sum of per-bond truncation errors (same invariant as to_mps).
    # ------------------------------------------------------------------
    @testitem "truncating canonicalize accumulates ε in quadrature from per-bond errors" begin
        mps = random_spin_half_mps(5; form=:right)
        result = canonicalize(mps, LeftCanonical(MaxBondDimTrunc(2)))
        # `LeftCanonical(MaxBondDimTrunc(2))` uses the 1-arg outer constructor.
        # `MaxBondDimTrunc(2)` truncates to at most 2 singular values per bond.
        # Physics: truncation introduces approximation — the reconstructed state
        # is an approximation of the original, with error bounded by mps.ε.

        per_bond_ε = sqrt(
            sum(result.bond_svs[i].ε^2 for i in 2:(length(result.bond_svs) - 1))
        )
        # Quadrature sum from bond 2 to bond L (skip trivial boundary bonds 1 and L+1).
        # Each bond's ε is a 2-NORM of discarded singular values, so ε² is the weight it
        # threw away; weights add, norms do not. Hence sqrt of the sum of squares.
        @test result.ε ≥ 0.0
        # Non-negativity: truncation error is always non-negative (you discard weight, not gain it).
        @test result.ε ≈ per_bond_ε atol=1e-14
        # The total error must equal the quadrature sum of the individual bond errors.
        # Physics: the DISCARDED WEIGHT ε² is what is additive across bonds, not ε itself.
    end

    # ------------------------------------------------------------------
    # Test 12 — canonicalize preserves error already carried by the input
    # REGRESSION TEST: every `canonicalize` method used to return only the
    # error of ITS OWN sweep, silently wiping whatever the input had already
    # accumulated. A TEBD loop canonicalizes every step, so the whole history
    # was erased each time round.
    # ------------------------------------------------------------------
    @testitem "non-truncating canonicalize carries the input's ε through" begin
        base = random_spin_half_mps(5; form=:right)
        ε₀ = 0.15   # a known, nonzero error the state is pretending to arrive with
        mps = FiniteMPS(base.tensors, base.bond_svs, base.form, ε₀)
        # Rebuild the struct with an injected ε — the tensors are untouched, so the only
        # thing under test is the bookkeeping.

        result = canonicalize(mps, LeftCanonical(NoTrunc()))
        # `NoTrunc()` discards nothing, so this sweep contributes exactly zero new error.
        @test result.ε ≈ ε₀ atol=1e-12
        # A pure gauge transformation cannot undo an approximation that was already made.
    end

    # ------------------------------------------------------------------
    # Test 13 — a truncating canonicalize ADDS its sweep to the input's ε
    # ------------------------------------------------------------------
    @testitem "truncating canonicalize adds its sweep to the input ε in quadrature" begin
        base = random_spin_half_mps(5; form=:right)
        ε₀ = 0.15
        mps = FiniteMPS(base.tensors, base.bond_svs, base.form, ε₀)

        result = canonicalize(mps, LeftCanonical(MaxBondDimTrunc(2)))
        ε_sweep = sqrt(sum(result.bond_svs[i].ε^2 for i in 2:(length(result.bond_svs) - 1)))
        # The error this sweep alone introduced, read back off the bond spectra it wrote.

        @test result.ε ≈ hypot(ε₀, ε_sweep) atol=1e-12
        # `hypot(a, b)` = sqrt(a² + b²) computed without spurious overflow.
        # Physics: the total discarded weight is the sum of the two discarded weights,
        # so the 2-norms compose in quadrature.
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
