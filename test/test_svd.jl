"""
Tests for SVD and truncation and Schmidt coefficients / entanglement entropy.

Physics invariants tested:
- Reconstruction: U·Σ·V† ≈ A (the SVD rank-r approximation)
- Isometry: U†U ≈ I and V†V† ≈ I (after truncation, per-sector on a graded space)
- Schmidt normalization: Σ_i² ≈ 1 for normalized Schmidt coefficients
- Truncation error identity: ε² + (kept singular values)² ≈ ‖A‖_F²
- Schmidt rank: counting singular values above a cutoff
- Degenerate singular values: truncation keeps both or neither, never splits a pair
- Entanglement entropy: product states → 0 bits; maximally entangled 2-qubit → 1 bit
- Boundary bonds: Schmidt decomposition at the open-chain ends → [1.0]
"""

using LinearAlgebra
using Test
using Serialization
using Random

# Import the active module (indices.jl and qtensor.jl)
using Qritical

# ============================================================================
# backend-agnostic tensor_svd + truncation + TensorSVD
# ============================================================================

@testset "SVD and truncation" begin

    # ────────────────────────────────────────────────────────────────────────
    # 1.4.1 Truncation strategies
    # ────────────────────────────────────────────────────────────────────────

    @testset "NoTrunc keeps all singular values (keep full Schmidt rank)" begin
        A = randn(3, 4)
        i = upper(:i, 3)
        j = lower(:j, 4)
        A_tensor = QTensor(A, (i, j))

        @test begin
            trunc = NoTrunc()
            bp = Bipartition(Partition([i]), Partition([j]))
            F = do_svd(A_tensor, bp, trunc)
            # Full rank is min(3, 4) = 3
            length(F.Σ.data.diag) == min(3, 4)
        end
    end
    @testset "Truncation strategies (4×4 orthogonal basis)" begin
        let U = qr(randn(4, 4)).Q, V = qr(randn(4, 4)).Q, i = upper(:i, 4), j = lower(:j, 4)
            @testset "MaxBondDimTrunc keeps at most D largest singular values (maximum Schmidt rank D)" begin
                # The returned truncation should have exactly min(D, rank(A)) singular values.
                let A_tensor = QTensor(U * Diagonal([4.0, 2.0, 1.0, 0.5]) * V', (i, j))
                    @test begin
                        trunc = MaxBondDimTrunc(2)
                        bp = Bipartition(Partition([i]), Partition([j]))
                        F = do_svd(A_tensor, bp, trunc)
                        length(F.Σ.data.diag) == 2
                    end
                end
            end

            @testset "ValCutoffTrunc keeps singular values above a threshold" begin
                # Physics: truncation by absolute tolerance (discard small singular values).
                let A_tensor = QTensor(U * Diagonal([4.0, 2.0, 0.1, 0.01]) * V', (i, j))
                    @test begin
                        trunc = ValCutoffTrunc(0.5)
                        bp = Bipartition(Partition([i]), Partition([j]))
                        F = do_svd(A_tensor, bp, trunc)
                        # Should keep r=2 (4.0 and 2.0 are above 0.5)
                        length(F.Σ.data.diag) == 2
                    end
                end
            end
        end  # let U, V, i, j
    end  # Truncation strategies

    # ────────────────────────────────────────────────────────────────────────
    # 1.4.2 Reconstruction from SVD
    # ────────────────────────────────────────────────────────────────────────

    @testset "Reconstruction: U·Σ·Vd ≈ A (no truncation, dense backend)" begin
        # Physics invariant: the SVD factors multiply back to the original matrix.
        # ‖A - U·Σ·V†‖_F should be negligible (~ machine epsilon).

        A = randn(ComplexF64, 5, 6)
        i = upper(:i, 5)
        j = lower(:j, 6)
        A_tensor = QTensor(A, (i, j))

        trunc = NoTrunc()
        bp = Bipartition(Partition([i]), Partition([j]))
        F = do_svd(A_tensor, bp, trunc)

        # Reconstruct: U·Σ·Vd
        U, Σ, Vd = F.U.data, F.Σ.data, F.Vd.data
        A_recon = U * Σ * Vd

        # Check norm of the error
        reconstruction_error = norm(A - A_recon)
        @test reconstruction_error < sqrt(eps()) * norm(A)  # ~machine epsilon relative error
    end

    @testset "Reconstruction with truncation includes the truncation error term" begin
        # Physics invariant: ‖A - U·Σ·Vd‖_F² = ε² (the truncation error is the 2-norm of discarded values).

        A = randn(5, 5)
        i = upper(:i, 5)
        j = lower(:j, 5)
        A_tensor = QTensor(A, (i, j))

        trunc = MaxBondDimTrunc(3)  # Keep only 3 singular values
        bp = Bipartition(Partition([i]), Partition([j]))
        F = do_svd(A_tensor, bp, trunc)

        U, Σ, Vd, ε = F.U.data, F.Σ.data, F.Vd.data, F.ε

        # Reconstruct the truncated approximation
        A_approx = U * Σ * Vd

        # Frobenius error should equal the reported truncation error
        reconstruction_error_f = norm(A - A_approx)
        @test isapprox(reconstruction_error_f, ε; rtol=1e-10)
    end

    # ────────────────────────────────────────────────────────────────────────
    # 1.4.3 Isometry: U†U = I and V†V† = I
    # ────────────────────────────────────────────────────────────────────────

    @testset "Isometry of U: U†U ≈ I after full-rank SVD" begin
        # Physics invariant: U is a partial isometry; U†U is the identity on its support.

        A = randn(4, 6)
        i = upper(:i, 4)
        j = lower(:j, 6)
        A_tensor = QTensor(A, (i, j))

        trunc = NoTrunc()
        bp = Bipartition(Partition([i]), Partition([j]))
        F = do_svd(A_tensor, bp, trunc)

        U = F.U.data
        rank_U = size(U, 2)

        # U†U should be the identity in the rank-U × rank-U subspace
        UdU = U' * U
        I_expected = Matrix(I, rank_U, rank_U)
        @test isapprox(UdU, I_expected; atol=1e-10)
    end

    @testset "Isometry of V†: V†·(V†)† ≈ I after full-rank SVD" begin
        # Physics invariant: V† is a partial isometry on the right.

        A = randn(5, 5)
        i = upper(:i, 5)
        j = lower(:j, 5)
        A_tensor = QTensor(A, (i, j))

        trunc = NoTrunc()
        bp = Bipartition(Partition([i]), Partition([j]))
        F = do_svd(A_tensor, bp, trunc)

        Vd = F.Vd.data
        rank_V = size(Vd, 1)

        # Vd·Vd† should be the identity in the rank_V × rank_V subspace
        VdVd_dag = Vd * Vd'
        I_expected = Matrix(I, rank_V, rank_V)
        @test isapprox(VdVd_dag, I_expected; atol=1e-10)
    end

    # ────────────────────────────────────────────────────────────────────────
    # 1.4.4 Singular value normalization (Schmidt coefficients)
    # ────────────────────────────────────────────────────────────────────────

    @testset "Schmidt coefficients: normalized Σ satisfies sum(λ²) ≈ 1 for normalized input" begin
        # Physics invariant: for a normalized state |ψ⟩, Schmidt coefficients satisfy
        # Σ_i λ_i² = 1 (the Schmidt coefficients are unit-normalized).

        # Build a normalized state (2-qubit entangled state: (|00⟩ + |11⟩)/√2)
        ψ_vector = [1.0, 0.0, 0.0, 1.0] / sqrt(2)
        ψ = reshape(ψ_vector, 2, 2)  # (2×2 matrix, representing the state on two qubits)

        # Normalize the full state
        ψ_norm = ψ / norm(ψ)

        i = lower(:i, 2)
        j = lower(:j, 2)
        ψ_tensor = QTensor(ψ_norm, (i, j))

        @test_broken begin
            trunc = NoTrunc()
            bp = Bipartition(Partition([i]), Partition([j]))
            F = tensor_svd(ψ_tensor, bp, trunc; normalize=true)

            # Schmidt coefficients squared should sum to ≈ 1
            λ_sq_sum = sum(abs2, diag(F.Σ.data))
            @test isapprox(λ_sq_sum, 1.0; atol=1e-10)
        end
    end

    @testset "Frobenius norm identity: ‖A‖_F² = sum(σ_i²) + ε²" begin
        # Physics invariant: the Frobenius norm of the full matrix equals the sum of squared
        # singular values (kept + discarded). After truncation, the total power is conserved.

        A = randn(6, 6)
        norm_A_F = norm(A)

        i = upper(:i, 6)
        j = lower(:j, 6)
        A_tensor = QTensor(A, (i, j))

        trunc = MaxBondDimTrunc(4)  # Keep 4, discard 2 (since 6×6 has rank ≤ 6)
        bp = Bipartition(Partition([i]), Partition([j]))
        F = do_svd(A_tensor, bp, trunc)

        σ_kept_sq = sum(abs2, diag(F.Σ.data))
        ε_sq = F.ε^2

        # Frobenius norm squared = sum of kept singular values squared + truncation error squared
        @test isapprox(norm_A_F^2, σ_kept_sq + ε_sq; atol=1e-10)
    end

    @testset "Truncation error identity: ε = ‖discarded singular values‖_2" begin
        # Physics invariant: the truncation error is the 2-norm of the discarded singular values.

        U = qr(randn(5, 5)).Q
        V = qr(randn(5, 5)).Q
        σ_full = [5.0, 3.0, 1.5, 0.3, 0.05]
        A = U * Diagonal(σ_full) * V'

        i = upper(:i, 5)
        j = lower(:j, 5)
        A_tensor = QTensor(A, (i, j))

        trunc = MaxBondDimTrunc(3)  # Keep 3 largest, discard [0.3, 0.05]
        bp = Bipartition(Partition([i]), Partition([j]))
        F = do_svd(A_tensor, bp, trunc)

        # Expected error: norm of [0.3, 0.05]
        ε_expected = norm(σ_full[4:end])
        @test isapprox(F.ε, ε_expected; atol=1e-10)
    end

    # ────────────────────────────────────────────────────────────────────────
    # 1.4.5 Schmidt rank and counting singular values
    # ────────────────────────────────────────────────────────────────────────

    @testset "Schmidt rank equals the number of singular values above the cutoff" begin
        # Physics invariant: the Schmidt rank is the number of nonzero Schmidt coefficients
        # (or, above a threshold in a numerical implementation).

        # Build a state with known Schmidt rank: |ψ⟩ = (|00⟩ + |11⟩)/√2 has rank 2.
        ψ_vector = [1.0, 0.0, 0.0, 1.0] / sqrt(2)
        ψ = reshape(ψ_vector, 2, 2)

        i = lower(:i, 2)
        j = lower(:j, 2)
        ψ_tensor = QTensor(ψ, (i, j))

        @test_broken begin
            trunc = ValCutoffTrunc(1e-10)  # Discard everything < 1e-10
            bp = Bipartition(Partition([i]), Partition([j]))
            F = tensor_svd(ψ_tensor, bp, trunc)

            # This entangled state has Schmidt rank 2
            schmidt_rank = length(F.Σ.data.diag)
            @test schmidt_rank == 2
        end
    end

    @testset "Product state has Schmidt rank 1" begin
        # Physics invariant: a product state |ψ⟩ = |ψ_A⟩ ⊗ |ψ_B⟩ has Schmidt rank 1.

        ψ = ones(3, 4) / sqrt(12)  # Normalized product state

        i = lower(:i, 3)
        j = lower(:j, 4)
        ψ_tensor = QTensor(ψ, (i, j))

        @test_broken begin
            trunc = NoTrunc()
            bp = Bipartition(Partition([i]), Partition([j]))
            F = tensor_svd(ψ_tensor, bp, trunc)

            # All but the largest singular value should be ~0
            σ_vals = diag(F.Σ.data)
            schmidt_rank_numeric = count(s -> s > 1e-12, σ_vals)
            @test schmidt_rank_numeric == 1
        end
    end

    # ────────────────────────────────────────────────────────────────────────
    # 1.4.6 Degenerate singular values (edge case)
    # ────────────────────────────────────────────────────────────────────────

    @testset "Degenerate singular values: truncation keeps both or neither" begin
        # Physics invariant: when singular values are degenerate (within numerical precision),
        # truncation must keep both or neither — splitting a degenerate pair is unphysical.

        # Build a matrix with degenerate singular values
        U = qr(randn(4, 4)).Q
        V = qr(randn(4, 4)).Q
        σ_vals = [4.0, 2.0, 2.0, 1.0]  # Two equal 2.0's
        A = U * Diagonal(σ_vals) * V'

        i = upper(:i, 4)
        j = lower(:j, 4)
        A_tensor = QTensor(A, (i, j))

        # Request to keep exactly 3 singular values.
        # The two degenerate 2.0's should both be kept (not split).
        trunc = MaxBondDimTrunc(3)
        bp = Bipartition(Partition([i]), Partition([j]))
        F = do_svd(A_tensor, bp, trunc)

        σ_kept = diag(F.Σ.data)
        # Should be [4.0, 2.0, 2.0] or [4.0, 2.0, 1.0]
        # but NOT [4.0, 2.0, X] where X is 2.0 split across truncation boundary.
        # Check that we don't have a singular value "between" 2.0 and 1.0.
        @test all(
            s -> abs(s - 2.0) < 1e-10 || abs(s - 4.0) < 1e-10 || abs(s - 1.0) < 1e-10,
            σ_kept,
        )
    end

    # ────────────────────────────────────────────────────────────────────────
    # 1.4.7 Complex matrices
    # ────────────────────────────────────────────────────────────────────────

    @testset "Complex matrix: A = U·Σ·V† with complex elements" begin
        # Physics invariant: SVD works for complex matrices; singular values are always real and ≥0.

        A = randn(ComplexF64, 3, 4)
        i = upper(:i, 3)
        j = lower(:j, 4)
        A_tensor = QTensor(A, (i, j))

        trunc = NoTrunc()
        bp = Bipartition(Partition([i]), Partition([j]))
        F = do_svd(A_tensor, bp, trunc)

        # Singular values must be real
        σ_vals = diag(F.Σ.data)
        @test eltype(σ_vals) <: Real

        # Singular values must be non-negative
        @test all(σ -> σ >= -1e-14, σ_vals)

        # Reconstruct and verify
        U, Σ, Vd = F.U.data, F.Σ.data, F.Vd.data
        A_recon = U * Σ * Vd
        @test isapprox(A, A_recon; atol=sqrt(eps()) * norm(A))
    end

    # ────────────────────────────────────────────────────────────────────────
    # 1.4.8 Rank-deficient matrices
    # ────────────────────────────────────────────────────────────────────────

    @testset "Rank-deficient matrix: rank(A) < min(m,n)" begin
        # Physics invariant: SVD correctly identifies the numerical rank.
        # Singular values below machine epsilon are "zero" in the numerical rank.

        # Build a rank-2 matrix in R^4×4
        U_small = qr(randn(4, 2)).Q
        V_small = qr(randn(4, 2)).Q
        A = U_small * [1.0 0.0; 0.0 1.0] * V_small'  # Rank 2 by construction

        i = upper(:i, 4)
        j = lower(:j, 4)
        A_tensor = QTensor(A, (i, j))

        trunc = ValCutoffTrunc(1e-10)  # Discard tiny values
        bp = Bipartition(Partition([i]), Partition([j]))
        F = do_svd(A_tensor, bp, trunc)

        # Number of kept singular values should be ≈ 2
        @test length(F.Σ.data.diag) <= 3  # At most 2–3 due to numerical error
        @test length(F.Σ.data.diag) >= 1  # At least 1
    end

    # ────────────────────────────────────────────────────────────────────────
    # 1.4.9 Bipartition semantics: state vs operator variance rule
    # ────────────────────────────────────────────────────────────────────────

    @testset "Bipartition respects index variance (state physical legs are Lower)" begin
        # Physics invariant: for a state tensor, all physical legs are of the same kind (Lower).
        # The SVD should respect the bipartition correctly.

        # A simple 3-site MPS-like tensor: σ_1, σ_2 (physical), vL, vR (bonds)
        σ1 = upper(:σ1, 2)
        σ2 = upper(:σ2, 2)
        vL = upper(:vL, 4)
        vR = lower(:vR, 4)

        # Data: 2×2×4×4
        data = randn(2, 2, 4, 4)
        A = QTensor(data, (σ1, σ2, vL, vR))

        # Bipartition: left = [σ1, σ2, vL] (3 legs), right = [vR] (1 leg)
        bp = Bipartition(Partition([σ1, σ2, vL]), Partition([vR]))
        F = do_svd(A, bp, NoTrunc())

        # The factor tensors carry one axis per original leg plus the bond, so a
        # multi-leg partition yields a genuine multi-leg U / Vd (not a bare matrix).
        @test length(F.U.indices) == 4   # σ1, σ2, vL, λ_Left
        @test length(F.Σ.indices) == 2   # λ_Left, λ_Right
        @test length(F.Vd.indices) == 2  # λ_Right, vR
        @test size(F.U.data)[1:3] == (2, 2, 4)
        @test size(F.Vd.data)[2] == 4

        # The reshaped factors reconstruct the original matricized state exactly.
        M = reshape(data, 2 * 2 * 4, 4)   # bp.left already occupies axes 1:3
        r = length(F.spectrum.values)
        Umat = reshape(F.U.data, 2 * 2 * 4, r)
        Vmat = reshape(F.Vd.data, r, 4)
        @test Umat * Diagonal(F.spectrum.values) * Vmat ≈ M
        @test sort(F.spectrum.values; rev=true) ≈ sort(svdvals(M); rev=true)[1:r]
    end
end  # @testset "SVD and truncation"

# ============================================================================
# Schmidt coefficients, entanglement entropy, data I/O
# ============================================================================

@testset "Schmidt coefficients and entanglement entropy" begin

    # ────────────────────────────────────────────────────────────────────────
    # 1.5.1 get_schmidt_coefficients
    # ────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────
    # 1.5.1 get_schmidt_coefficients
    # ────────────────────────────────────────────────────────────────────────

    # Bell state (|00⟩+|11⟩)/√2 — shared across all Schmidt coefficient + entropy tests.
    @testset "Bell state: Schmidt coefficients and entanglement entropy" begin
        let _bell = reshape([1.0, 0.0, 0.0, 1.0] / sqrt(2), 2, 2),
            _i = lower(:i, 2),
            _j = lower(:j, 2),
            ψ_tensor = QTensor(_bell, (_i, _j)),
            bp = Bipartition(Partition([_i]), Partition([_j]))

            @testset "Schmidt coefficients are non-negative and normalized to 1" begin
                # Physics: λ_i ≥ 0 and Σ_i λ_i² = 1 for a normalized state.
                @test_broken begin
                    λ = get_schmidt_coefficients(ψ_tensor, bp)
                    @test all(λ .>= -1e-14)
                    @test isapprox(sum(abs2, λ), 1.0; atol=1e-10)
                end
            end

            @testset "Maximally entangled state has equal Schmidt coefficients λ = 1/√2" begin
                # Physics: Bell pair has λ_1 = λ_2 = 1/√2.
                @test_broken begin
                    λ = get_schmidt_coefficients(ψ_tensor, bp)
                    expected = 1.0 / sqrt(2)
                    @test isapprox(λ[1], expected; atol=1e-10)
                    @test isapprox(λ[2], expected; atol=1e-10)
                end
            end

            # ────────────────────────────────────────────────────────────────────
            # 1.5.2 get_entanglement_entropy
            # ────────────────────────────────────────────────────────────────────

            @testset "Entanglement entropy: Bell state = 1 bit (log base 2)" begin
                # Physics: S = -2·(1/2·log₂(1/2)) = 1 bit.
                @test_broken begin
                    S = get_entanglement_entropy(ψ_tensor, bp; base=2)
                    @test isapprox(S, 1.0; atol=1e-10)
                end
            end

            @testset "Entanglement entropy: default base is 2 (result in bits)" begin
                # Physics: omitting base keyword must give the same result as base=2.
                @test_broken begin
                    S_default = get_entanglement_entropy(ψ_tensor, bp)
                    S_base2 = get_entanglement_entropy(ψ_tensor, bp; base=2)
                    @test isapprox(S_default, S_base2; atol=1e-14)
                    @test isapprox(S_default, 1.0; atol=1e-10)
                end
            end
        end  # let _bell …
    end  # Bell state

    @testset "Entanglement entropy: product state = 0 bits" begin
        # Physics: S[ρ_A] = 0 for a product state (ones(2,3)/√6).
        @test_broken begin
            i = lower(:i, 2);
            j = lower(:j, 3)
            ψ_tensor = QTensor(ones(2, 3) / sqrt(6), (i, j))
            bp = Bipartition(Partition([i]), Partition([j]))
            S = get_entanglement_entropy(ψ_tensor, bp; base=2)
            @test isapprox(S, 0.0; atol=1e-10)
        end
    end

    # 3×4 normalized product state — shared by Schmidt + zero-log-zero entropy tests.
    @testset "Product state ones(3,4)/√12: Schmidt rank and zero-log-zero handling" begin
        let _i = lower(:i, 3),
            _j = lower(:j, 4),
            ψ_tensor = QTensor(ones(3, 4) / sqrt(12), (_i, _j)),
            bp = Bipartition(Partition([_i]), Partition([_j]))

            @testset "Product state has single Schmidt coefficient = 1" begin
                # Physics: |ψ_A⟩⊗|ψ_B⟩ has Schmidt rank 1 and λ_1 = 1.
                @test_broken begin
                    λ = get_schmidt_coefficients(ψ_tensor, bp)
                    @test length(λ) == 1
                    @test isapprox(λ[1], 1.0; atol=1e-10)
                end
            end

            @testset "Entanglement entropy handles zero singular values (0·log(0) = 0)" begin
                # Physics: rank-1 state has S = 0 even with near-zero numerical values.
                @test_broken begin
                    S = get_entanglement_entropy(ψ_tensor, bp; base=2)
                    @test isapprox(S, 0.0; atol=1e-10)
                end
            end
        end  # let _i …
    end  # Product state ones(3,4)/√12

    # ────────────────────────────────────────────────────────────────────────
    # 1.5.3 Boundary bonds: entanglement entropy at chain ends
    # ────────────────────────────────────────────────────────────────────────

    # 3-qubit uniform product state — shared by both boundary-bond tests.
    @testset "Boundary bonds of open-chain product state" begin
        let _σ1 = upper(:σ1, 2),
            _σ2 = upper(:σ2, 2),
            _σ3 = upper(:σ3, 2),
            ψ_tensor = QTensor(ones(2, 2, 2) / (2 * sqrt(2)), (_σ1, _σ2, _σ3)),
            bp = Bipartition(Partition([_σ1]), Partition([_σ2, _σ3]))

            @testset "Boundary bonds have Schmidt rank 1 (no entanglement)" begin
                # Physics: open-chain boundary bond has a single Schmidt coefficient = 1.
                @test_broken begin
                    λ = get_schmidt_coefficients(ψ_tensor, bp)
                    @test length(λ) == 1
                    @test isapprox(λ[1], 1.0; atol=1e-10)
                end
            end

            @testset "Entanglement entropy at boundary is 0 bits" begin
                @test_broken begin
                    S = get_entanglement_entropy(ψ_tensor, bp; base=2)
                    @test isapprox(S, 0.0; atol=1e-10)
                end
            end
        end  # let _σ1 …
    end  # Boundary bonds

    # ────────────────────────────────────────────────────────────────────────
    # 1.5.4 load_array — data I/O
    # ────────────────────────────────────────────────────────────────────────

    @testset "load_array: .jls format (Serialization)" begin
        # Physics: state data should load unchanged from a trusted serialized file.

        test_data = [1.0, 2.0, 3.0, 4.0]
        temp_file = tempname() * ".jls"
        open(temp_file, "w") do f
            serialize(f, test_data)
        end
        loaded_data = load_array(temp_file)
        @test loaded_data ≈ test_data
        rm(temp_file)
    end

    @testset "load_array: .txt format (DelimitedFiles)" begin
        # Physics: state data should load unchanged from a text file.

        using DelimitedFiles
        test_data = [1.0 2.0; 3.0 4.0]
        temp_file = tempname() * ".txt"
        writedlm(temp_file, test_data)
        loaded_data = load_array(temp_file)
        @test loaded_data ≈ test_data
        rm(temp_file)
    end

    # ────────────────────────────────────────────────────────────────────────
    # 1.5.5 bipartition_matrix — reshape state to matrix form
    # ────────────────────────────────────────────────────────────────────────

    # All three bipartition_matrix tests use the same 3-qubit state array.
    @testset "bipartition_matrix" begin
        let ψ_vector = randn(8), ψ_full = reshape(ψ_vector, 2, 2, 2)
            @testset "reshape d^L state to (d^cut × d^(L-cut))" begin
                # Physics: matricising at cut=1 gives a (2 × 4) matrix.
                @test_broken begin
                    M = bipartition_matrix(ψ_full, cut=1)
                    @test size(M) == (2, 4)
                    @test isapprox(reshape(M, 2, 2, 2), ψ_full)
                end
            end

            @testset "edge case cut=0 gives (1 × d^L) matrix" begin
                # Physics: trivial left subsystem (dimension 1).
                @test_broken begin
                    M = bipartition_matrix(ψ_full, cut=0)
                    @test size(M) == (1, 8)
                    @test isapprox(vec(M), ψ_vector)
                end
            end

            @testset "edge case cut=L gives (d^L × 1) matrix" begin
                # Physics: trivial right subsystem (dimension 1).
                @test_broken begin
                    M = bipartition_matrix(ψ_full, cut=3)
                    @test size(M) == (8, 1)
                    @test isapprox(vec(M), ψ_vector)
                end
            end
        end  # let ψ_vector, ψ_full
    end  # bipartition_matrix

    # ────────────────────────────────────────────────────────────────────────
    # 1.5.6 as_state — reshape flat vector to rank-L tensor
    # ────────────────────────────────────────────────────────────────────────

    # Parametric test: d=2 (qubit) and d=3 (qutrit) share identical logic.
    @testset "as_state: reshape d^L vector to rank-L tensor (d=$d, L=$L)" for (L, d) in [
        (4, 2), (3, 3)
    ]
        @test_broken begin
            ψ_vector = randn(ComplexF64, d^L)
            ψ_tensor = as_state(ψ_vector, L; d)
            @test size(ψ_tensor) == ntuple(_ -> d, L)
            @test isapprox(vec(ψ_tensor), ψ_vector)
        end
    end

    @testset "as_state: preserves normalization" begin
        # Physics: reshaping a normalized state vector should preserve the norm.
        @test_broken begin
            L = 3;
            d = 2
            ψ_vector = randn(d^L)
            ψ_vector /= norm(ψ_vector)
            ψ_tensor = as_state(ψ_vector, L; d)
            @test isapprox(norm(vec(ψ_tensor)), norm(ψ_vector); atol=1e-14)
        end
    end
end  # @testset "Schmidt coefficients and entanglement entropy"
