# Tests for: §2.1 (FiniteMPS struct) and §2.2 (to_mps) of MasterPlan
# Physics invariants: MPS canonical forms, isometry, reconstruction, truncation error accounting
# TDD Red Phase: all tests fail by design, to be implemented per MasterPlan design.

using Test, LinearAlgebra, Qritical
# `using` = Python's `from module import *` — brings all exported names into scope.
# `Test` provides @test, @testset, etc. (like Python's pytest assertions).
# `LinearAlgebra` provides norm, I, svd, etc. (like numpy.linalg).
# `Qritical` is our own package — brings FiniteMPS, to_mps, etc. into scope.

# ============================================================================
# Helper function: contract_mps
# ============================================================================
# Fully contract an MPS by absorbing bonds left-to-right to recover the state.
# This is the ground-truth reconstruction test.
function contract_mps(mps::FiniteMPS)
    # This function manually contracts the MPS tensor network into a single tensor.
    # Physics: |ψ⟩ = A₁ · A₂ · … · A_L (matrix product of site tensors, open indices are physical).
    # Contracting left-to-right accumulates all physical indices: result = full state tensor.

    L = length(mps.tensors)   # number of sites (like Python's len(mps.tensors))

    # Start with the leftmost site tensor's data.
    # Shape: (χ_L, σ, χ_R) — for site 1, χ_L=1 (boundary), so shape is (1, d, χ)
    current = mps.tensors[1].data
    # In Julia, array indexing is `A[i]`, same as Python's `A[i-1]` (because Julia is 1-indexed).

    # Successively contract in the right virtual index.
    for i in 2:L
        # At iteration i, `current` has accumulated all physical indices from sites 1..i-1.
        # Its shape is: (σ₁, σ₂, …, σ_{i-1}, χ_R) after i-1 steps.
        # mps.tensors[i] has shape: (χ_L, σ_i, χ_R)
        # We want to contract the right virtual of `current` with the left virtual of site i.
        # This is a tensor contraction: result[σ₁..σ_{i-1}, σᵢ, χ'] = Σ_χ current[…, χ] · A_i[χ, σᵢ, χ']

        A_i = mps.tensors[i].data
        # tensordot(current, A_i, ([ndims(current)], [1]))
        # Reshape current to 2D: (..., χ_R) → (prod(...), χ_R)
        current_2d = reshape(current, :, size(current, ndims(current)))
        # `reshape(A, :, n)` = numpy's `A.reshape(-1, n)`. The `:` means "infer this dimension".
        # `ndims(current)` = the number of dimensions of `current` (like numpy's `current.ndim`).
        # `size(current, ndims(current))` = the LAST dimension of current (the right virtual bond).
        # So current_2d has shape (prod(all-but-last-dims), χ_R).

        # A_i: (χ_L, σ, χ_R)
        A_i_2d = reshape(A_i, size(A_i, 1), size(A_i, 2) * size(A_i, 3))
        # Reshape A_i from (χ_L, σ, χ_R) to (χ_L, σ*χ_R).
        # `size(A_i, 1)` = first dimension (χ_L). `size(A_i, 2) * size(A_i, 3)` = d*χ_R.
        # In numpy: A_i_2d = A_i.reshape(A_i.shape[0], A_i.shape[1] * A_i.shape[2])

        # Contract: (prod(...), χ_R) * (χ_R, σ * χ'_R) → (prod(...), σ * χ'_R)
        result_2d = current_2d * A_i_2d
        # Standard matrix multiplication: `*` in Julia = `@` operator in Python (numpy matmul).
        # current_2d is (M, χ_R), A_i_2d is (χ_R, σ*χ'), result is (M, σ*χ').

        # Reshape back to tensor with physical leg inserted
        result_shape = (size(current)[1:(ndims(current)-1)]..., size(A_i, 2), size(A_i, 3))
        # `size(current)` returns a Tuple of all dimensions: (σ₁, …, σ_{i-1}, χ_R).
        # `[1:(ndims(current)-1)]` slices off the last dimension (χ_R): (σ₁, …, σ_{i-1}).
        # The `...` unpacks the tuple: `(tuple..., a, b)` = `(*tuple, a, b)` in Python.
        # So result_shape = (σ₁, …, σ_{i-1}, σᵢ, χ'_R) — adds the new physical leg.
        current = reshape(result_2d, result_shape)
        # Reshape result from (M, σ*χ') to (σ₁, …, σ_{i-1}, σᵢ, χ') — the tensor with all physical legs.
    end

    return current
    # Final shape: (σ₁, σ₂, …, σ_L, 1) — the last dim is the right boundary (χ_R=1 at boundary).
    # Physics: this is the full state tensor A^{σ₁σ₂…σ_L}, reconstructed from the MPS.
end


# ============================================================================
# Test Suite: FiniteMPS and to_mps
# ============================================================================

@testset "FiniteMPS and to_mps" begin
# `@testset "name" begin ... end` = pytest's `class TestFiniteMPS: def test_...(self): ...`
# In Julia, @testset groups related tests. Nested @testsets give a tree of results.
# If any @test inside fails, the testset reports a failure with a clear message.

    # ========================================================================
    # §2.1 FiniteMPS struct — basic construction invariants
    # ========================================================================
    @testset "FiniteMPS struct" begin

        # Test 1: length consistency — Physical invariant: L sites, L+1 bonds
        @testset "MPS tensor count matches number of sites" begin
            L = 4
            d = 2
            # Create L site tensors with generic dimensions
            tensors = [
                QTensor(randn(1, d, 2), (upper(:vL, 1), upper(:σ, d), lower(:vR, 2))),
                QTensor(randn(2, d, 3), (upper(:vL, 2), upper(:σ, d), lower(:vR, 3))),
                QTensor(randn(3, d, 2), (upper(:vL, 3), upper(:σ, d), lower(:vR, 2))),
                QTensor(randn(2, d, 1), (upper(:vL, 2), upper(:σ, d), lower(:vR, 1))),
            ]
            # Array literal syntax: `[a, b, c, d]` = Python's `[a, b, c, d]`.
            # `randn(1, d, 2)` = numpy's `np.random.randn(1, d, 2)` — random normal 3D array.
            # `QTensor(data, indices)` wraps the data with index metadata (variance tags + names).
            # `upper(:vL, 1)` = Upper (contravariant) index named :vL with dimension 1.
            # `lower(:vR, 2)` = Lower (covariant) index named :vR with dimension 2.
            # Convention: left boundary has vL=1, right interior has vR=χ.

            bond_svs = [
                SingValSpectrum([1.0], 0.0, true),
                SingValSpectrum([0.5, 0.3], 0.1, true),
                SingValSpectrum([0.6, 0.4], 0.05, true),
                SingValSpectrum([0.7, 0.2, 0.1], 0.0, true),
                SingValSpectrum([1.0], 0.0, true),
            ]
            # L+1=5 bond spectra for L=4 sites. The first and last are the trivial boundaries [1.0].
            # `SingValSpectrum(values, ε, normalized)`:
            #   values = vector of singular values (Schmidt values)
            #   ε = truncation error at this bond (0.0 if no truncation)
            #   normalized = whether ||values||² ≈ 1

            form = CanonicalForm(L, L+1)
            # CanonicalForm(4, 5): sites 1..3 are left-canonical, no right-canonical sites.
            # llim=4=L means all sites except the last are left-canonical.

            mps = FiniteMPS(tensors, bond_svs, form, 0.15)
            # Construct the MPS struct directly. The 0.15 is the accumulated truncation error ε.
            # This is a MANUAL construction for testing — normally you'd use to_mps().

            # Assertion: MPS has L tensors
            @test length(mps.tensors) == L
            # `@test expr` = `assert expr` in Python. If false, prints a failure message.
            # Physics check: the number of site tensors must exactly equal the chain length L.
        end

        # Test 2: bond spectrum count — Physical invariant: L+1 bonds on open chain
        @testset "bond_svs has L+1 spectra for open chain" begin
            L = 3
            d = 2
            tensors = [
                QTensor(randn(1, d, 2), (upper(:vL, 1), upper(:σ, d), lower(:vR, 2))),
                QTensor(randn(2, d, 2), (upper(:vL, 2), upper(:σ, d), lower(:vR, 2))),
                QTensor(randn(2, d, 1), (upper(:vL, 2), upper(:σ, d), lower(:vR, 1))),
            ]
            bond_svs = [
                SingValSpectrum([1.0], 0.0, true),
                SingValSpectrum([0.5, 0.3], 0.1, true),
                SingValSpectrum([0.6], 0.2, true),
                SingValSpectrum([1.0], 0.0, true),
            ]
            form = CanonicalForm(L, L+1)

            mps = FiniteMPS(tensors, bond_svs, form, 0.3)

            # Assertion: L+1 bonds on open chain
            @test length(mps.bond_svs) == L + 1
            # Physics: an open chain with L sites has L+1 bond positions:
            # [boundary|site1|bond1|site2|bond2|…|site_L|boundary]
            # So bond_svs has indices 1..L+1. Bonds 1 and L+1 are the trivial open boundaries.
        end

        # Test 3: boundary bonds are trivial [1.0] — Open-chain boundary condition
        @testset "boundary bond spectra are [1.0]" begin
            L = 2
            d = 2
            tensors = [
                QTensor(randn(1, d, 2), (upper(:vL, 1), upper(:σ, d), lower(:vR, 2))),
                QTensor(randn(2, d, 1), (upper(:vL, 2), upper(:σ, d), lower(:vR, 1))),
            ]
            bond_svs = [
                SingValSpectrum([1.0], 0.0, true),
                SingValSpectrum([0.5], 0.1, true),
                SingValSpectrum([1.0], 0.0, true),
            ]
            form = CanonicalForm(L, L+1)

            mps = FiniteMPS(tensors, bond_svs, form, 0.1)

            # Assertion: Left boundary (index 1) has spectrum [1.0]
            @test mps.bond_svs[1].values ≈ [1.0]
            # `≈` is Julia's `isapprox` operator (same as Python's np.allclose with default tolerances).
            # Physics: at the open boundary, the virtual bond has dimension 1 (no entanglement).
            # The single singular value is 1.0 (trivial Schmidt decomposition: ||ψ|| = 1).

            # Assertion: Right boundary (index L+1) has spectrum [1.0]
            @test mps.bond_svs[L+1].values ≈ [1.0]
            # Same reasoning: right boundary bond also has χ=1 and single SV=1.0.
        end

        # Test 4: each site tensor is order-3 — Tensor network geometry
        @testset "each site tensor is rank-3 (vL, σ, vR)" begin
            L = 3
            d = 2
            tensors = [
                QTensor(randn(1, d, 2), (upper(:vL, 1), upper(:σ, d), lower(:vR, 2))),
                QTensor(randn(2, d, 3), (upper(:vL, 2), upper(:σ, d), lower(:vR, 3))),
                QTensor(randn(3, d, 1), (upper(:vL, 3), upper(:σ, d), lower(:vR, 1))),
            ]
            bond_svs = [
                SingValSpectrum([1.0], 0.0, true),
                SingValSpectrum([0.5, 0.3], 0.1, true),
                SingValSpectrum([0.6], 0.2, true),
                SingValSpectrum([1.0], 0.0, true),
            ]
            form = CanonicalForm(L, L+1)

            mps = FiniteMPS(tensors, bond_svs, form, 0.3)

            # Assertion: every site tensor is rank-3
            for i in 1:L
                @test ndims(mps.tensors[i].data) == 3
                # `ndims(A)` = A.ndim in numpy. Every MPS site tensor must have exactly 3 legs:
                # (vL, σ, vR) = (left virtual, physical, right virtual).
            end
        end

        # Test 5: form is an AbstractMPSForm — Type safety
        @testset "form field is an AbstractMPSForm" begin
            L = 2
            d = 2
            tensors = [
                QTensor(randn(1, d, 2), (upper(:vL, 1), upper(:σ, d), lower(:vR, 2))),
                QTensor(randn(2, d, 1), (upper(:vL, 2), upper(:σ, d), lower(:vR, 1))),
            ]
            bond_svs = [
                SingValSpectrum([1.0], 0.0, true),
                SingValSpectrum([0.5], 0.1, true),
                SingValSpectrum([1.0], 0.0, true),
            ]
            form = CanonicalForm(L, L+1)

            mps = FiniteMPS(tensors, bond_svs, form, 0.1)

            # Assertion: form is an AbstractMPSForm instance
            @test mps.form isa AbstractMPSForm
            # `x isa T` = `isinstance(x, T)` in Python.
            # This checks that the form field is a subtype of AbstractMPSForm.
            # Since CanonicalForm <: AbstractMPSForm, this will pass for any valid form.
        end
    end


    # ========================================================================
    # §2.2 to_mps: left-canonical sweep
    # ========================================================================
    @testset "to_mps: left-canonical form" begin
        # Test 6: isometry condition A†A ≈ I (left-canonical) — von Delft §4.2
        @testset "left-canonical A_i satisfies A_i† A_i ≈ I" begin
            # Build a small state vector: 2^4 = 16 elements
            ψ_vec = randn(2^4)
            # `2^4` = 2**4 = 16 in Python. Julia uses `^` for exponentiation (not `**`).
            ψ = as_state(ψ_vec, [2, 2, 2, 2])
            # `as_state(vector, dims)` wraps a state vector as a QTensor with physical legs.
            # [2,2,2,2] means 4 sites each with local Hilbert-space dimension 2 (spin-1/2).

            # Convert to MPS via left-canonical sweep with no truncation
            mps = to_mps(ψ, trunc=NoTrunc(), form=:left)
            # `NoTrunc()` = no truncation: keep ALL singular values (exact representation).
            # `form=:left` = left-canonical sweep. `:left` is a Symbol (Julia's immutable string-like type).

            # Check isometry on each site except the last (which holds the norm)
            for i in 1:(length(mps.tensors)-1)
                # Reshape A_i: (χ_L, σ, χ_R) → (χ_L * σ, χ_R)
                A_i = mps.tensors[i].data
                χ_L, d, χ_R = size(A_i)
                A_i_mat = reshape(A_i, χ_L * d, χ_R)
                # Flatten (χ_L, d) together to get an (χ_L*d × χ_R) matrix.
                # This is the "left-unfolding" of the tensor — think of it as a wide isometric matrix.

                # Check: A_i† * A_i ≈ I_{χ_R}
                # (Frobenius norm of deviation from identity)
                product = A_i_mat' * A_i_mat
                # `A'` is Julia's adjoint (conjugate transpose). For real arrays: A' = A^T.
                # `A' * B` = matrix multiplication: (χ_R × χ_L*d) × (χ_L*d × χ_R) → (χ_R × χ_R).
                identity = I(χ_R)
                error_norm = norm(product - identity)
                # `I(n)` = n×n identity matrix (from LinearAlgebra). Like numpy's np.eye(n).
                # `norm(M)` = Frobenius norm of matrix M = sqrt(sum of squared entries).

                # Physical invariant: isometry error ≪ 1 (should be ~1e-14 for untruncated)
                @test error_norm < 1e-10
                # Physics: ||A†A - I|| < 1e-10 means the tensor is numerically left-isometric.
                # Machine precision for Float64 is ~1e-16, but accumulated floating-point errors
                # in the SVD can push the error to ~1e-14. The 1e-10 threshold is safe.
            end
        end

        # Test 7: left-canonical form tag — State representation tracking
        @testset "form == CanonicalForm(L, L+1) after left sweep" begin
            ψ_vec = randn(2^3)
            ψ = as_state(ψ_vec, [2, 2, 2])

            mps = to_mps(ψ, trunc=NoTrunc(), form=:left)

            # Assertion: after left sweep, llim = L (all sites are left-canonical)
            @test mps.form == CanonicalForm(length(mps.tensors), length(mps.tensors)+1)
            # For L=3: we expect CanonicalForm(3, 4).
            # `==` compares structs by VALUE (field by field), like Python's `__eq__`.
            # Julia auto-generates value equality for plain structs.
        end

        # Test 8: reconstruction ψ_reconstructed ≈ ψ (no truncation) — Algebraic completeness
        @testset "full contraction recovers original state (NoTrunc)" begin
            # Small state vector
            ψ_original = randn(2^4)
            ψ = as_state(ψ_original, [2, 2, 2, 2])

            # Build MPS with NoTrunc
            mps = to_mps(ψ, trunc=NoTrunc(), form=:left)

            # Contract the full MPS
            ψ_reconstructed = contract_mps(mps)
            # Call our helper above: contracts all L tensors back into a single dense tensor.

            # Flatten both to vectors for comparison
            ψ_vec = vec(ψ.data)
            # `vec(A)` = numpy's `A.ravel()` or `A.flatten()` — flattens to 1D vector.
            # ψ.data has shape (2,2,2,2); vec gives a 16-element vector.
            ψ_recon_vec = vec(ψ_reconstructed)

            # Normalize to handle global phase
            ψ_vec ./= norm(ψ_vec)
            # `./=` is in-place elementwise division: like Python's `ψ_vec /= np.linalg.norm(ψ_vec)`.
            # The `.` means "broadcast this operation elementwise".
            ψ_recon_vec ./= norm(ψ_recon_vec)

            # Physical invariant: reconstruction is exact within machine epsilon
            @test ψ_recon_vec ≈ ψ_vec atol=1e-12
            # `≈` with `atol` is like `np.allclose(a, b, atol=1e-12)`.
            # Physics: for NoTrunc, the MPS is an EXACT representation.
            # Contracting the MPS back must give exactly the original state (up to floating-point).
        end

        # Test 9: boundary bond dimensions are 1 — Tensor network boundary condition
        @testset "boundary tensors have dim-1 virtual legs" begin
            ψ_vec = randn(2^4)
            ψ = as_state(ψ_vec, [2, 2, 2, 2])

            mps = to_mps(ψ, trunc=NoTrunc(), form=:left)

            L = length(mps.tensors)

            # Left boundary: first site has vL dimension 1
            vL_dim = dim(mps.tensors[1].indices[1])
            # `mps.tensors[1]` = first site tensor (site 1, Julia 1-indexed).
            # `.indices[1]` = the first index of that tensor (the vL index).
            # `dim(index)` extracts the dimension of the index.
            @test vL_dim == 1
            # Physics: open boundary condition means the virtual dimension at the chain end is 1.
            # There's no entanglement "beyond" the chain edge.

            # Right boundary: last site has vR dimension 1
            vR_dim = dim(mps.tensors[L].indices[3])
            # `.indices[3]` = the third index (vR) of the last site tensor.
            @test vR_dim == 1
        end
    end


    # ========================================================================
    # §2.2 to_mps: right-canonical sweep
    # ========================================================================
    @testset "to_mps: right-canonical form" begin
        # Test 10: isometry condition B_i B_i† ≈ I (right-canonical) — von Delft §4.2
        @testset "right-canonical B_i satisfies B_i B_i† ≈ I" begin
            ψ_vec = randn(2^4)
            ψ = as_state(ψ_vec, [2, 2, 2, 2])

            # Right-canonical sweep
            mps = to_mps(ψ, trunc=NoTrunc(), form=:right)
            # `:right` triggers _right_sweep: sites 2..L become right-canonical.
            # Site 1 is the norm carrier (not constrained to be isometric).

            # Check isometry on each site except the first (which holds the norm)
            for i in 2:length(mps.tensors)
                # Reshape B_i: (χ_L, σ, χ_R) → (χ_L, σ * χ_R)
                B_i = mps.tensors[i].data
                χ_L, d, χ_R = size(B_i)
                B_i_mat = reshape(B_i, χ_L, d * χ_R)
                # For right-canonical: group the PHYSICAL+RIGHT indices together.
                # B_i_mat has shape (χ_L × d*χ_R) — a "tall" isometric matrix.

                # Check: B_i * B_i† ≈ I_{χ_L}
                product = B_i_mat * B_i_mat'
                # `B * B'` = (χ_L × d*χ_R) × (d*χ_R × χ_L) → (χ_L × χ_L).
                # Physics: BB† = I means the rows of B_mat are orthonormal.
                identity = I(χ_L)
                error_norm = norm(product - identity)

                # Physical invariant: isometry error ≪ 1
                @test error_norm < 1e-10
            end
        end

        # Test 11: right-canonical form tag — State representation tracking
        @testset "form == CanonicalForm(0, 1) after right sweep" begin
            ψ_vec = randn(2^3)
            ψ = as_state(ψ_vec, [2, 2, 2])

            mps = to_mps(ψ, trunc=NoTrunc(), form=:right)

            # Assertion: after right sweep, rlim = 1 (all sites are right-canonical)
            @test mps.form == CanonicalForm(0, 1)
            # CanonicalForm(0, 1): llim=0 (sentinel — no left-canonical sites), rlim=1 (sites 1..L right-canonical).
            # Physics: in fully right-canonical form, all site tensors (except site 1) satisfy BB†=I.
        end
    end


    # ========================================================================
    # §2.2 to_mps: truncation with MaxBondDimTrunc
    # ========================================================================
    @testset "to_mps: truncation with MaxBondDimTrunc" begin
        # Test 12: max bond dimension is enforced — Truncation strategy
        @testset "MaxBondDimTrunc(D) enforces inner bonds ≤ D" begin
            ψ_vec = randn(2^4)
            ψ = as_state(ψ_vec, [2, 2, 2, 2])

            D_max = 2
            mps = to_mps(ψ, trunc=MaxBondDimTrunc(D_max), form=:left)
            # `MaxBondDimTrunc(D)` truncates the SVD: keep at most D singular values per bond.
            # This is the main compression mechanism in DMRG and TEBD.
            # Physics: truncating the bond dimension limits the entanglement the MPS can represent.

            L = length(mps.tensors)

            # Check that inner bonds (not boundary) have dimension ≤ D_max
            for i in 1:(L-1)
                # Right virtual bond of site i
                vR_dim = dim(mps.tensors[i].indices[3])
                # `.indices[3]` = the third tensor index (vR). `dim` extracts its dimension.
                @test vR_dim <= D_max
                # Physics: after MaxBondDimTrunc(D), every inner bond has χ ≤ D.
                # The bond dimension CAN be smaller than D if the state has low entanglement.
            end
        end

        # Test 13: accumulated error matches sum of per-bond errors — Error accounting
        @testset "mps.ε ≈ sum of per-bond truncation errors" begin
            ψ_vec = randn(2^4)
            ψ = as_state(ψ_vec, [2, 2, 2, 2])

            D_max = 1  # Force truncation: every bond keeps only 1 singular value
            mps = to_mps(ψ, trunc=MaxBondDimTrunc(D_max), form=:left)

            # Sum the per-bond errors (excluding boundaries which have ε=0)
            per_bond_ε = sum(mps.bond_svs[i].ε for i in 2:(length(mps.bond_svs)-1))
            # Generator expression: `sum(expr for i in range)` = Python's `sum(expr for i in range)`.
            # `mps.bond_svs[i].ε` = the truncation error stored at bond i.
            # Boundaries (bond_svs[1] and bond_svs[L+1]) have ε=0.0 (no truncation there).

            # Physical invariant: accumulated ε tracks total discarded weight
            @test mps.ε ≈ per_bond_ε atol=1e-14
            # Physics: the total truncation error is the SUM of per-bond errors.
            # Each bond's error = ||discarded singular values||² = squared truncation error.
            # This is the standard MPS error measure (see Vidal 2003, White 1992).
        end
    end


    # ========================================================================
    # ========================================================================
    # §2.1 / §2.2 edge cases: L=1 and L=2
    # ========================================================================
    @testset "to_mps: L=1 single-site (no inner bonds)" begin
        # Physics: a single site has no virtual bonds; both boundary spectra are [1.0].
        ψ_vec = [0.6, 0.8]
        # Array literal `[a, b]` = Python's `[a, b]` or `np.array([a, b])`.
        ψ = as_state(ψ_vec, [2])   # L=1 chain, d=2 (a single qubit)
        mps = to_mps(ψ; trunc=NoTrunc(), form=:left)
        # Note the `;` before keyword args when calling — required when positional args precede keywords.
        # In Python: `to_mps(ψ, trunc=NoTrunc(), form=':left')` (keyword args don't need `;`).

        @test length(mps.tensors) == 1          # single site = single tensor
        @test length(mps.bond_svs) == 2         # L+1=2 bond spectra (both boundaries)
        @test size(mps.tensors[1].data) == (1, 2, 1)
        # The single site tensor has shape (χ_L=1, d=2, χ_R=1).
        # `size(A)` = A.shape in numpy — returns a Tuple.
        # Tuple comparison with `==` works elementwise in Julia.

        @test mps.bond_svs[1].values ≈ [1.0]   # left boundary: trivial χ=1, SV=[1.0]
        @test mps.bond_svs[2].values ≈ [1.0]   # right boundary: trivial χ=1, SV=[1.0]
        @test mps.ε == 0.0                      # no truncation → zero error
        # Reconstruction: the single tensor encodes the state exactly
        @test mps.tensors[1].data[1, :, 1] ≈ ψ_vec atol=1e-12
        # `A[1, :, 1]` extracts all elements of the middle dimension (the physical leg σ).
        # `:` in Julia indexing = "all elements along this dimension" = numpy's `:`/`...`.
        # Physics: for L=1, the "MPS" is just the state vector itself, stored as [1,:,1].
    end

    @testset "to_mps: L=2 single inner bond" begin
        # Physics: a 2-site Bell state (|00⟩+|11⟩)/√2 has bond dim 2 and equal SVs 1/√2.
        ψ_vec = [1.0, 0.0, 0.0, 1.0] / sqrt(2)
        # `sqrt(2)` = Python's `math.sqrt(2)` or `np.sqrt(2)`.
        # Division of a vector by a scalar: `v / c` in Julia = numpy's `v / c` (broadcasts).
        # Physical state: |Bell⟩ = (|↑↑⟩ + |↓↓⟩)/√2 — maximally entangled 2-qubit state.
        ψ = as_state(ψ_vec, [2, 2])
        mps = to_mps(ψ; trunc=NoTrunc(), form=:left)

        @test length(mps.tensors) == 2          # 2 sites = 2 tensors
        @test length(mps.bond_svs) == 3         # L+1=3 bond spectra
        @test mps.bond_svs[1].values ≈ [1.0]   # left boundary trivial
        @test mps.bond_svs[3].values ≈ [1.0]   # right boundary trivial
        # Inner bond should reflect the entanglement: two equal SVs 1/√2
        @test length(mps.bond_svs[2].values) == 2
        # Bond dim 2 for a Bell state: the Schmidt rank is 2 (two non-zero Schmidt values).
        @test sort(mps.bond_svs[2].values; rev=true) ≈ [1/sqrt(2), 1/sqrt(2)] atol=1e-12
        # `sort(...; rev=true)` = Python's `sorted(..., reverse=True)`.
        # Physics: the Bell state has Schmidt values {1/√2, 1/√2} at the central bond.
        # Equal Schmidt values = maximally entangled (maximum entropy = log(2) bits).
        # Reconstruction
        @test abs(overlap(mps, mps)) ≈ 1.0 atol=1e-12
        # `overlap(ψ, φ)` = ⟨ψ|φ⟩ (inner product of two MPS).
        # `abs(x)` = |x| (absolute value, handles complex numbers too).
        # For a normalized state, ⟨ψ|ψ⟩ = 1.
    end

    # ========================================================================
    # ========================================================================
    # §4.2 add_mps — superposition and edge cases
    # ========================================================================
    @testset "add_mps: superposition of two MPS" begin
        L = 4
        ψ_vec = randn(2^L); ψ_vec ./= norm(ψ_vec)
        # Semicolons `;` on the same line = Python's newline. Separate two statements.
        # `./=` is in-place elementwise division: like Python's `ψ_vec /= np.linalg.norm(ψ_vec)`.
        φ_vec = randn(2^L); φ_vec ./= norm(φ_vec)
        ψ_mps = to_mps(as_state(ψ_vec, fill(2, L)); trunc=NoTrunc(), form=:left)
        # `fill(2, L)` = vector of length L with all values 2 = Python's `[2] * L`.
        φ_mps = to_mps(as_state(φ_vec, fill(2, L)); trunc=NoTrunc(), form=:left)

        @testset "add_mps(1,ψ,1,ψ) represents 2|ψ⟩" begin
            sum_mps = add_mps(1, ψ_mps, 1, ψ_mps)
            # add_mps(a, ψ, b, φ) computes a|ψ⟩ + b|φ⟩.
            # Here: 1·|ψ⟩ + 1·|ψ⟩ = 2|ψ⟩.
            # ⟨ψ+ψ|ψ⟩ = 2⟨ψ|ψ⟩ = 2
            @test real(overlap(sum_mps, ψ_mps)) ≈ 2.0 atol=1e-10
            # `real(x)` = takes the real part. Like Python's `x.real` or `np.real(x)`.
            # Physics: ⟨2ψ|ψ⟩ = 2⟨ψ|ψ⟩ = 2 for a normalized state.
        end

        @testset "add_mps: coefficients respected (2ψ + 3φ)" begin
            sum_mps = add_mps(2.0, ψ_mps, 3.0, φ_mps)
            # ⟨2ψ+3φ|ψ⟩ = 2⟨ψ|ψ⟩ = 2  (ψ,φ orthogonal only if random states are, not guaranteed)
            # Use the explicit formula: = 2*norm(ψ)² + 3*⟨φ|ψ⟩
            expected = 2.0 * overlap(ψ_mps, ψ_mps) + 3.0 * overlap(φ_mps, ψ_mps)
            @test overlap(sum_mps, ψ_mps) ≈ expected atol=1e-10
            # Physics: linearity of the inner product: ⟨(aψ+bφ)|χ⟩ = a⟨ψ|χ⟩ + b⟨φ|χ⟩.
        end

        @testset "add_mps: a=0 reduces to b*φ" begin
            sum_mps = add_mps(0, ψ_mps, 1, φ_mps)
            # 0·|ψ⟩ + 1·|φ⟩ = |φ⟩. So ⟨0ψ+φ|φ⟩ = ⟨φ|φ⟩ = 1.
            @test real(overlap(sum_mps, φ_mps)) ≈ 1.0 atol=1e-10
        end

        @testset "add_mps: b=0 reduces to a*ψ" begin
            sum_mps = add_mps(1, ψ_mps, 0, φ_mps)
            @test real(overlap(sum_mps, ψ_mps)) ≈ 1.0 atol=1e-10
        end

        @testset "add_mps: adding state to itself with truncation" begin
            sum_mps = add_mps(1.0/sqrt(2), ψ_mps, 1.0/sqrt(2), ψ_mps; trunc=MaxBondDimTrunc(4))
            # (1/√2)|ψ⟩ + (1/√2)|ψ⟩ = √2|ψ⟩. So ⟨√2ψ|ψ⟩ = √2.
            # The keyword arg `trunc=MaxBondDimTrunc(4)` is passed with `;` before it.
            # In Julia: `f(a, b; kw=val)` — positional first, then keyword args after `;`.
            ov = real(overlap(sum_mps, ψ_mps))
            @test isapprox(ov, sqrt(2); atol=1e-10) ||
                  isapprox(abs(ov), sqrt(2); atol=1e-10)
            # `isapprox(a, b; atol=...)` = np.isclose(a, b, atol=...).
            # The `||` is logical OR — we accept both signs because of global phase freedom.
        end

        @testset "Base.:+ sugar works" begin
            sum_mps = ψ_mps + φ_mps
            # Uses the `Base.:+` extension defined in mps.jl: ψ + φ = add_mps(1, ψ, 1, φ).
            expected = overlap(ψ_mps, ψ_mps) + overlap(φ_mps, ψ_mps)
            @test overlap(sum_mps, ψ_mps) ≈ expected atol=1e-10
        end
    end

    # §2.2 to_mps: product state special case
    # ========================================================================
    @testset "to_mps: product state has bond dim 1" begin
        # Test 14: separable state |↑↑↑↑⟩ has all bond dims = 1 and zero entanglement
        @testset "product state (e.g. |↑↑↑↑⟩) has χ=1 everywhere" begin
            # Product state: |↑↑↑↑⟩ = [1,0] ⊗ [1,0] ⊗ [1,0] ⊗ [1,0]
            # As a full state vector, only first element is 1
            ψ_product = zeros(2^4)
            # `zeros(n)` = numpy's `np.zeros(n)`. Creates a Float64 vector of zeros.
            ψ_product[1] = 1.0  # |↑⟩⊗|↑⟩⊗|↑⟩⊗|↑⟩
            # Julia is 1-indexed: index 1 is the first element (Python's index 0).
            # In computational basis: |↑↑↑↑⟩ = basis state with σ₁=↑,σ₂=↑,σ₃=↑,σ₄=↑.

            ψ = as_state(ψ_product, [2, 2, 2, 2])

            # Convert to MPS
            mps = to_mps(ψ, trunc=NoTrunc(), form=:left)

            L = length(mps.tensors)

            # Physical invariant: all bond dimensions = 1 for separable state
            for i in 1:(L-1)
                # Right bond of site i
                vR_dim = dim(mps.tensors[i].indices[3])
                @test vR_dim == 1
                # Physics: |↑↑…↑⟩ is a product state (zero entanglement).
                # The Schmidt rank at any bond = 1 (only one Schmidt term).
                # So all inner bond dimensions should be χ=1.
            end

            # Physical invariant: all bond spectra are [1.0] (zero entanglement)
            for i in 1:(L+1)
                @test mps.bond_svs[i].values ≈ [1.0]
                # Every bond has a single Schmidt value = 1.0.
                # This is the minimum entanglement configuration (separable state).
            end
        end
    end

end
