# Tests for: §2.1 (FiniteMPS struct) and §2.2 (to_mps) of MasterPlan
# Physics invariants: MPS canonical forms, isometry, reconstruction, truncation error accounting
# TDD Red Phase: all tests fail by design, to be implemented per MasterPlan design.

using Test, LinearAlgebra, Qritical

# ============================================================================
# Helper function: contract_mps
# ============================================================================
# Fully contract an MPS by absorbing bonds left-to-right to recover the state.
# This is the ground-truth reconstruction test.
function contract_mps(mps::FiniteMPS)
    L = length(mps.tensors)

    # Start with the leftmost site tensor's data.
    # Shape: (χ_L, σ, χ_R)
    current = mps.tensors[1].data

    # Successively contract in the right virtual index.
    for i in 2:L
        # current shape after i-1 steps: (σ_1, σ_2, …, σ_{i-1}, χ_R)
        # mps.tensors[i] shape: (χ_L, σ_i, χ_R)
        # Contract: current[…, χ] ⊗ A[χ, σ, χ'] → current[…, σ, χ']
        # Using tensordot: contract last axis of current with first axis of A[i]
        A_i = mps.tensors[i].data
        # tensordot(current, A_i, ([ndims(current)], [1]))
        # Reshape current to 2D: (..., χ_R) → (prod(...), χ_R)
        current_2d = reshape(current, :, size(current, ndims(current)))
        # A_i: (χ_L, σ, χ_R)
        A_i_2d = reshape(A_i, size(A_i, 1), size(A_i, 2) * size(A_i, 3))
        # Contract: (prod(...), χ_R) * (χ_R, σ * χ'_R) → (prod(...), σ * χ'_R)
        result_2d = current_2d * A_i_2d
        # Reshape back to tensor with physical leg inserted
        result_shape = (size(current)[1:(ndims(current)-1)]..., size(A_i, 2), size(A_i, 3))
        current = reshape(result_2d, result_shape)
    end

    return current
end


# ============================================================================
# Test Suite: FiniteMPS and to_mps
# ============================================================================

@testset "FiniteMPS and to_mps" begin

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
            bond_svs = [
                SingValSpectrum([1.0], 0.0, true),
                SingValSpectrum([0.5, 0.3], 0.1, true),
                SingValSpectrum([0.6, 0.4], 0.05, true),
                SingValSpectrum([0.7, 0.2, 0.1], 0.0, true),
                SingValSpectrum([1.0], 0.0, true),
            ]
            form = CanonicalForm(L, L+1)

            mps = FiniteMPS(tensors, bond_svs, form, 0.15)

            # Assertion: MPS has L tensors
            @test length(mps.tensors) == L
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
            # Assertion: Right boundary (index L+1) has spectrum [1.0]
            @test mps.bond_svs[L+1].values ≈ [1.0]
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
            ψ = as_state(ψ_vec, [2, 2, 2, 2])

            # Convert to MPS via left-canonical sweep with no truncation
            mps = to_mps(ψ, trunc=NoTrunc(), form=:left)

            # Check isometry on each site except the last (which holds the norm)
            for i in 1:(length(mps.tensors)-1)
                # Reshape A_i: (χ_L, σ, χ_R) → (χ_L * σ, χ_R)
                A_i = mps.tensors[i].data
                χ_L, d, χ_R = size(A_i)
                A_i_mat = reshape(A_i, χ_L * d, χ_R)

                # Check: A_i† * A_i ≈ I_{χ_R}
                # (Frobenius norm of deviation from identity)
                product = A_i_mat' * A_i_mat
                identity = I(χ_R)
                error_norm = norm(product - identity)

                # Physical invariant: isometry error ≪ 1 (should be ~1e-14 for untruncated)
                @test error_norm < 1e-10
            end
        end

        # Test 7: left-canonical form tag — State representation tracking
        @testset "form == CanonicalForm(L, L+1) after left sweep" begin
            ψ_vec = randn(2^3)
            ψ = as_state(ψ_vec, [2, 2, 2])

            mps = to_mps(ψ, trunc=NoTrunc(), form=:left)

            # Assertion: after left sweep, llim = L (all sites are left-canonical)
            @test mps.form == CanonicalForm(length(mps.tensors), length(mps.tensors)+1)
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

            # Flatten both to vectors for comparison
            ψ_vec = vec(ψ.data)
            ψ_recon_vec = vec(ψ_reconstructed)

            # Normalize to handle global phase
            ψ_vec ./= norm(ψ_vec)
            ψ_recon_vec ./= norm(ψ_recon_vec)

            # Physical invariant: reconstruction is exact within machine epsilon
            @test ψ_recon_vec ≈ ψ_vec atol=1e-12
        end

        # Test 9: boundary bond dimensions are 1 — Tensor network boundary condition
        @testset "boundary tensors have dim-1 virtual legs" begin
            ψ_vec = randn(2^4)
            ψ = as_state(ψ_vec, [2, 2, 2, 2])

            mps = to_mps(ψ, trunc=NoTrunc(), form=:left)

            L = length(mps.tensors)

            # Left boundary: first site has vL dimension 1
            vL_dim = dim(mps.tensors[1].indices[1])
            @test vL_dim == 1

            # Right boundary: last site has vR dimension 1
            vR_dim = dim(mps.tensors[L].indices[3])
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

            # Check isometry on each site except the first (which holds the norm)
            for i in 2:length(mps.tensors)
                # Reshape B_i: (χ_L, σ, χ_R) → (χ_L, σ * χ_R)
                B_i = mps.tensors[i].data
                χ_L, d, χ_R = size(B_i)
                B_i_mat = reshape(B_i, χ_L, d * χ_R)

                # Check: B_i * B_i† ≈ I_{χ_L}
                product = B_i_mat * B_i_mat'
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

            L = length(mps.tensors)

            # Check that inner bonds (not boundary) have dimension ≤ D_max
            for i in 1:(L-1)
                # Right virtual bond of site i
                vR_dim = dim(mps.tensors[i].indices[3])
                @test vR_dim <= D_max
            end
        end

        # Test 13: accumulated error matches sum of per-bond errors — Error accounting
        @testset "mps.ε ≈ sum of per-bond truncation errors" begin
            ψ_vec = randn(2^4)
            ψ = as_state(ψ_vec, [2, 2, 2, 2])

            D_max = 1  # Force truncation
            mps = to_mps(ψ, trunc=MaxBondDimTrunc(D_max), form=:left)

            # Sum the per-bond errors (excluding boundaries which have ε=0)
            per_bond_ε = sum(mps.bond_svs[i].ε for i in 2:(length(mps.bond_svs)-1))

            # Physical invariant: accumulated ε tracks total discarded weight
            @test mps.ε ≈ per_bond_ε atol=1e-14
        end
    end


    # ========================================================================
    # ========================================================================
    # §2.1 / §2.2 edge cases: L=1 and L=2
    # ========================================================================
    @testset "to_mps: L=1 single-site (no inner bonds)" begin
        # Physics: a single site has no virtual bonds; both boundary spectra are [1.0].
        ψ_vec = [0.6, 0.8]
        ψ = as_state(ψ_vec, [2])
        mps = to_mps(ψ; trunc=NoTrunc(), form=:left)

        @test length(mps.tensors) == 1
        @test length(mps.bond_svs) == 2
        @test size(mps.tensors[1].data) == (1, 2, 1)
        @test mps.bond_svs[1].values ≈ [1.0]
        @test mps.bond_svs[2].values ≈ [1.0]
        @test mps.ε == 0.0
        # Reconstruction: the single tensor encodes the state exactly
        @test mps.tensors[1].data[1, :, 1] ≈ ψ_vec atol=1e-12
    end

    @testset "to_mps: L=2 single inner bond" begin
        # Physics: a 2-site Bell state (|00⟩+|11⟩)/√2 has bond dim 2 and equal SVs 1/√2.
        ψ_vec = [1.0, 0.0, 0.0, 1.0] / sqrt(2)
        ψ = as_state(ψ_vec, [2, 2])
        mps = to_mps(ψ; trunc=NoTrunc(), form=:left)

        @test length(mps.tensors) == 2
        @test length(mps.bond_svs) == 3
        @test mps.bond_svs[1].values ≈ [1.0]
        @test mps.bond_svs[3].values ≈ [1.0]
        # Inner bond should reflect the entanglement: two equal SVs 1/√2
        @test length(mps.bond_svs[2].values) == 2
        @test sort(mps.bond_svs[2].values; rev=true) ≈ [1/sqrt(2), 1/sqrt(2)] atol=1e-12
        # Reconstruction
        @test abs(overlap(mps, mps)) ≈ 1.0 atol=1e-12
    end

    # ========================================================================
    # ========================================================================
    # §4.2 add_mps — superposition and edge cases
    # ========================================================================
    @testset "add_mps: superposition of two MPS" begin
        L = 4
        ψ_vec = randn(2^L); ψ_vec ./= norm(ψ_vec)
        φ_vec = randn(2^L); φ_vec ./= norm(φ_vec)
        ψ_mps = to_mps(as_state(ψ_vec, fill(2, L)); trunc=NoTrunc(), form=:left)
        φ_mps = to_mps(as_state(φ_vec, fill(2, L)); trunc=NoTrunc(), form=:left)

        @testset "add_mps(1,ψ,1,ψ) represents 2|ψ⟩" begin
            sum_mps = add_mps(1, ψ_mps, 1, ψ_mps)
            # ⟨ψ+ψ|ψ⟩ = 2⟨ψ|ψ⟩ = 2
            @test real(overlap(sum_mps, ψ_mps)) ≈ 2.0 atol=1e-10
        end

        @testset "add_mps: coefficients respected (2ψ + 3φ)" begin
            sum_mps = add_mps(2.0, ψ_mps, 3.0, φ_mps)
            # ⟨2ψ+3φ|ψ⟩ = 2⟨ψ|ψ⟩ = 2  (ψ,φ orthogonal only if random states are, not guaranteed)
            # Use the explicit formula: = 2*norm(ψ)² + 3*⟨φ|ψ⟩
            expected = 2.0 * overlap(ψ_mps, ψ_mps) + 3.0 * overlap(φ_mps, ψ_mps)
            @test overlap(sum_mps, ψ_mps) ≈ expected atol=1e-10
        end

        @testset "add_mps: a=0 reduces to b*φ" begin
            sum_mps = add_mps(0, ψ_mps, 1, φ_mps)
            # ⟨0ψ+φ|φ⟩ = ⟨φ|φ⟩ = 1
            @test real(overlap(sum_mps, φ_mps)) ≈ 1.0 atol=1e-10
        end

        @testset "add_mps: b=0 reduces to a*ψ" begin
            sum_mps = add_mps(1, ψ_mps, 0, φ_mps)
            @test real(overlap(sum_mps, ψ_mps)) ≈ 1.0 atol=1e-10
        end

        @testset "add_mps: adding state to itself with truncation" begin
            sum_mps = add_mps(1.0/sqrt(2), ψ_mps, 1.0/sqrt(2), ψ_mps; trunc=MaxBondDimTrunc(4))
            # sqrt(2) * ψ_mps after normalization: overlap with ψ should be sqrt(2)
            ov = real(overlap(sum_mps, ψ_mps))
            @test isapprox(ov, sqrt(2); atol=1e-10) ||
                  isapprox(abs(ov), sqrt(2); atol=1e-10)
        end

        @testset "Base.:+ sugar works" begin
            sum_mps = ψ_mps + φ_mps
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
            ψ_product[1] = 1.0  # |↑⟩⊗|↑⟩⊗|↑⟩⊗|↑⟩

            ψ = as_state(ψ_product, [2, 2, 2, 2])

            # Convert to MPS
            mps = to_mps(ψ, trunc=NoTrunc(), form=:left)

            L = length(mps.tensors)

            # Physical invariant: all bond dimensions = 1 for separable state
            for i in 1:(L-1)
                # Right bond of site i
                vR_dim = dim(mps.tensors[i].indices[3])
                @test vR_dim == 1
            end

            # Physical invariant: all bond spectra are [1.0] (zero entanglement)
            for i in 1:(L+1)
                @test mps.bond_svs[i].values ≈ [1.0]
            end
        end
    end

end
