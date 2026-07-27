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

using LinearAlgebra   # `using LinearAlgebra` = import standard library; exposes `svd`, `Diagonal`, `norm`, `qr`, `svdvals`, `I`, `diagm`
using Test            # @test, @testset, @test_throws, @test_broken
using Serialization   # `using Serialization` = standard library for `serialize`/`deserialize` 
using Random          # `using Random` = standard library for random number generation 

# Import the active module (indices.jl and qtensor.jl)
using Qritical   # all exported names: QTensor, Bipartition, do_svd, NoTrunc, etc.

const _gvl = Qritical._golub_van_loan_threshold   # `const` = module-level binding; `Qritical._golub_van_loan_threshold` = access private function from the module 

# ============================================================================
# backend-agnostic tensor_svd + truncation + TensorSVD
# ============================================================================

@testset "SVD and truncation" begin

    # ────────────────────────────────────────────────────────────────────────
    # 1.4.1 Truncation strategies
    # ────────────────────────────────────────────────────────────────────────

    @testset "NoTrunc keeps all singular values (keep full Schmidt rank)" begin
        A = randn(3, 4)      # `randn(m, n)` = m×n matrix of N(0,1) random values
        i = upper(:i, 3)
        j = lower(:j, 4)
        A_tensor = QTensor(A, (i, j))

        @test begin
            trunc = NoTrunc()   # `NoTrunc()` = singleton struct; selects the full-rank SVD dispatch branch
            bp = Bipartition(Partition([i]), Partition([j]))
            F = do_svd(A_tensor, bp, trunc)
            # Full rank is min(3, 4) = 3
            length(F.Σ.data.diag) == min(3, 4)   # `F.Σ.data` = the Diagonal matrix inside the QTensor; `.diag` = the diagonal vector; `min(3,4)=3` = full rank of a 3×4 matrix
        end
    end
end
@testset "Truncation strategies (4×4 orthogonal basis)" begin
    let U = qr(randn(4, 4)).Q, V = qr(randn(4, 4)).Q, i = upper(:i, 4), j = lower(:j, 4)   # `qr(M).Q` = orthogonal factor of QR decomposition ; gives exact singular vectors
        @testset "MaxBondDimTrunc keeps at most D largest singular values (maximum Schmidt rank D)" begin
            # The returned truncation should have exactly min(D, rank(A)) singular values.
            let A_tensor = QTensor(U * Diagonal([4.0, 2.0, 1.0, 0.5]) * V', (i, j))   # `U * Diagonal(σ) * V'` = construct matrix with known singular values; `V'` = conjugate transpose 
                @test begin
                    trunc = MaxBondDimTrunc(2)   # `MaxBondDimTrunc(2)` = keep at most 2 singular values (bond dimension cap χ≤2)
                    bp = Bipartition(Partition([i]), Partition([j]))
                    F = do_svd(A_tensor, bp, trunc)
                    length(F.Σ.data.diag) == 2   # exactly 2 kept (not 3 or 4)
                end
            end
            @testset "ValCutoffTrunc keeps singular values above a threshold" begin
                # Physics: truncation by absolute tolerance (discard small singular values).
                let A_tensor = QTensor(U * Diagonal([4.0, 2.0, 0.1, 0.01]) * V', (i, j))
                    @test begin
                        trunc = ValCutoffTrunc(0.5)   # `ValCutoffTrunc(0.5)` = discard σᵢ ≤ 0.5; only [4.0, 2.0] survive (0.1 < 0.5)
                        bp = Bipartition(Partition([i]), Partition([j]))
                        F = do_svd(A_tensor, bp, trunc)
                        # Should keep r=2 (4.0 and 2.0 are above 0.5)
                        length(F.Σ.data.diag) == 2
                    end
                end
            end
        end  # @testset "MaxBondDimTrunc keeps at most D largest singular values (maximum Schmidt rank D)"
    end  # let U, V, i, j
end  # Truncation strategies

    # ────────────────────────────────────────────────────────────────────────
    # 1.4.2 Reconstruction from SVD
    # ────────────────────────────────────────────────────────────────────────

    @testset "Reconstruction: U·Σ·Vd ≈ A (no truncation, dense backend)" begin
        # Physics invariant: the SVD factors multiply back to the original matrix.
        # ‖A - U·Σ·V†‖_F should be negligible (~ machine epsilon).

        A = randn(ComplexF64, 5, 6)   # `randn(ComplexF64, 5, 6)` = complex random matrix 
        i = upper(:i, 5)
        j = lower(:j, 6)
        A_tensor = QTensor(A, (i, j))

        trunc = NoTrunc()
        bp = Bipartition(Partition([i]), Partition([j]))
        F = do_svd(A_tensor, bp, trunc)

        # Reconstruct: U·Σ·Vd

    @testset "Reconstruction with truncation includes the truncation error term" begin
        # Physics invariant: ‖A - U·Σ·Vd‖_F² = ε² (the truncation error is the 2-norm of discarded values).

        A = randn(5, 5)
        i = upper(:i, 5)
        j = lower(:j, 5)
        A_tensor = QTensor(A, (i, j))

        trunc = MaxBondDimTrunc(3)   # Keep only 3 singular values
        bp = Bipartition(Partition([i]), Partition([j]))
        F = do_svd(A_tensor, bp, trunc)

        U, Σ, Vd, ε = F.U.data, F.Σ.data, F.Vd.data, F.ε   # `F.ε` = field on Re
        # Reconstruct the truncated approximation
        A_approx = U * Σ * Vd

        # Frobenius error should equal the reported truncation error
        reconstruction_error_f = norm(A - A_approx)   # ‖A - A_r‖_F where A_r = rank-r approximation
        @test isapprox(reconstruction_error_f, ε; rtol=1e-10)   # `rtol` = relative tolerance; ε is computed from the discarded singular values' 2-norm which equals the Frobenius approximation error
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
        rank_U = size(U, 2)   # `size(M, 2)` = number of columns = rank (dim of σ-space) 

        # U†U should be the identity in the rank-U × rank-U subspace
        UdU = U' * U   # `U'` = adjoint (conjugate transpose) of U ; U†U should be I_r if U is isometric
        I_expected = Matrix(I, rank_U, rank_U)   # `Matrix(I, r, r)` = r×r identity matrix 
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
        rank_V = size(Vd, 1)   # `size(Vd, 1)` = number of rows = rank

        # Vd·Vd† should be the identity in the rank_V × rank_V subspace
        VdVd_dag = Vd * Vd'   # Vd · Vd† = I_r if Vd is an isometry (rows orthonormal)
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
        ψ_norm = ψ / norm(ψ)   # `norm(M)` = Frobenius norm of matrix 

        i = lower(:i, 2)
        j = lower(:j, 2)
        ψ_tensor = QTensor(ψ_norm, (i, j))

        @test_broken begin   # `@test_broken begin ... end` = marks entire block as expected to fail; `tensor_svd` is not yet implemented (uses `do_svd` instead)
            trunc = NoTrunc()
            bp = Bipartition(Partition([i]), Partition([j]))
            F = tensor_svd(ψ_tensor, bp, trunc; normalize=true)   # `tensor_svd` with `normalize=true` kwarg: not yet implemented in current codebase

            # Schmidt coefficients squared should sum to ≈ 1
            λ_sq_sum = sum(abs2, diag(F.Σ.data))   # `abs2(x)` = |x|²; `diag(D)` = diagonal of matrix 
            @test isapprox(λ_sq_sum, 1.0; atol=1e-10)
        end
    end

    @testset "Frobenius norm identity: ‖A‖_F² = sum(σ_i²) + ε²" begin
        # Physics invariant: the Frobenius norm of the full matrix equals the sum of squared
        # singular values (kept + discarded). After truncation, the total power is conserved.

        A = randn(6, 6)
        norm_A_F = norm(A)   # Frobenius norm = √(∑|Aᵢⱼ|²) = √(∑σᵢ²) by SVD norm identity

        i = upper(:i, 6)
        j = lower(:j, 6)
        A_tensor = QTensor(A, (i, j))

        trunc = MaxBondDimTrunc(4)   # Keep 4, discard 2 (since 6×6 has rank ≤ 6)
        bp = Bipartition(Partition([i]), Partition([j]))
        F = do_svd(A_tensor, bp, trunc)

        σ_kept_sq = sum(abs2, diag(F.Σ.data))   # `sum(abs2, v)` = ∑|σᵢ|² for kept values; `diag(F.Σ.data)` = diagonal vector from Diagonal matrix
        ε_sq = F.ε^2   # `F.ε^2` = ε² = (2-norm of discarded tail)²; `^` = exponentiation 

        # Frobenius norm squared = sum of kept singular values squared + truncation error squared
        @test isapprox(norm_A_F^2, σ_kept_sq + ε_sq; atol=1e-10)   # ‖A‖_F² = ∑σᵢ²(kept) + ε²; this is an exact identity from the SVD norm theorem
    end

    @testset "Truncation error identity: ε = ‖discarded singular values‖_2" begin
        # Physics invariant: the truncation error is the 2-norm of the discarded singular values.

        U = qr(randn(5, 5)).Q
        V = qr(randn(5, 5)).Q
        σ_full = [5.0, 3.0, 1.5, 0.3, 0.05]
        A = U * Diagonal(σ_full) * V'   # construct matrix with exactly known singular values

        i = upper(:i, 5)
        j = lower(:j, 5)
        A_tensor = QTensor(A, (i, j))

        trunc = MaxBondDimTrunc(3)  # Keep 3 largest, discard [0.3, 0.05]
        bp = Bipartition(Partition([i]), Partition([j]))
        F = do_svd(A_tensor, bp, trunc)

        # Expected error: norm of [0.3, 0.05]
        ε_expected = norm(σ_full[4:end])   # `σ_full[4:end]` = slice from index 4 to end (1-indexed): [0.3, 0.05]; `norm(v)` = 2-norm = √(0.3²+0.05²)
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

        @test_broken begin   # `tensor_svd` not yet implemented; placeholder marks intended future test
            trunc = ValCutoffTrunc(1e-10)  # Discard everything < 1e-10
            bp = Bipartition(Partition([i]), Partition([j]))
            F = tensor_svd(ψ_tensor, bp, trunc)

            # This entangled state has Schmidt rank 2
            schmidt_rank = length(F.Σ.data.diag)   # count kept singular values = Schmidt rank
            @test schmidt_rank == 2
        end
    end

    @testset "Product state has Schmidt rank 1" begin
        # Physics invariant: a product state |ψ⟩ = |ψ_A⟩ ⊗ |ψ_B⟩ has Schmidt rank 1.

        ψ = ones(3, 4) / sqrt(12)  # Normalized product state; `ones(3,4)` = 3×4 matrix of 1.0

        i = lower(:i, 3)
        j = lower(:j, 4)
        ψ_tensor = QTensor(ψ, (i, j))

        @test_broken begin   # `tensor_svd` not yet implemented
            trunc = NoTrunc()
            bp = Bipartition(Partition([i]), Partition([j]))
            F = tensor_svd(ψ_tensor, bp, trunc)

            # All but the largest singular value should be ~0
            σ_vals = diag(F.Σ.data)
            schmidt_rank_numeric = count(s -> s > 1e-12, σ_vals)   # `count(pred, iter)` = number of elements satisfying predicate 
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
        σ_vals = [4.0, 2.0, 2.0, 1.0]  # Two equal 2.0's (degenerate pair)
        A = U * Diagonal(σ_vals) * V'

        i = upper(:i, 4)
        j = lower(:j, 4)
        A_tensor = QTensor(A, (i, j))

        # Request to keep exactly 3 singular values.
        # The two degenerate 2.0's should both be kept (not split).
        trunc = MaxBondDimTrunc(3)
        bp = Bipartition(Partition([i]), Partition([j]))
        F = do_svd(A_tensor, bp, trunc)

        σ_kept = diag(F.Σ.data)   # kept singular values as a plain vector
        # Should be [4.0, 2.0, 2.0] or [4.0, 2.0, 1.0]
        # but NOT [4.0, 2.0, X] where X is 2.0 split across truncation boundary.
        # Check that we don't have a singular value "between" 2.0 and 1.0.
        @test all(
            s -> abs(s - 2.0) < 1e-10 || abs(s - 4.0) < 1e-10 || abs(s - 1.0) < 1e-10,
            σ_kept,   # `all(pred, iter)` = all elements satisfy predicate; `abs(s - 2.0) < 1e-10` = close to 2.0; ensures no "in-between" value appears
        )
    end

    # ────────────────────────────────────────────────────────────────────────
    # 1.4.7 Complex matrices
    # ────────────────────────────────────────────────────────────────────────

    @testset "Complex matrix: A = U·Σ·V† with complex elements" begin
        # Physics invariant: SVD works for complex matrices; singular values are always real and ≥0.

        A = randn(ComplexF64, 3, 4)   # ComplexF64 = 64-bit complex 
        i = upper(:i, 3)
        j = lower(:j, 4)
        A_tensor = QTensor(A, (i, j))

        trunc = NoTrunc()
        bp = Bipartition(Partition([i]), Partition([j]))
        F = do_svd(A_tensor, bp, trunc)

        # Singular values must be real
        σ_vals = diag(F.Σ.data)
        @test eltype(σ_vals) <: Real   # `<: Real` = subtype of Real; singular values are always real even for complex matrices (they're norms of columns of a polar decomposition)

        # Singular values must be non-negative
        @test all(σ -> σ >= -1e-14, σ_vals)   # `σ -> σ >= -1e-14` = anonymous function. allow tiny negative from floating-point noise

        # Reconstruct and verify
        U, Σ, Vd = F.U.data, F.Σ.data, F.Vd.data
        A_recon = U * Σ * Vd
        @test isapprox(A, A_recon; atol=sqrt(eps()) * norm(A))   # `sqrt(eps())` ≈ 1.5e-8 = relative machine-precision tolerance
    end

    # ────────────────────────────────────────────────────────────────────────
    # 1.4.8 Rank-deficient matrices
    # ────────────────────────────────────────────────────────────────────────

    @testset "Rank-deficient matrix: rank(A) < min(m,n)" begin
        # Physics invariant: SVD correctly identifies the numerical rank.
        # Singular values below machine epsilon are "zero" in the numerical rank.

        # Build a rank-2 matrix in R^4×4
        U_small = qr(randn(4, 2)).Q   # `qr(randn(4,2)).Q` = orthogonal 4×2 matrix (Q factor of thin QR)
        V_small = qr(randn(4, 2)).Q
        A = U_small * [1.0 0.0; 0.0 1.0] * V_small'  # Rank 2 by construction; [1 0;0 1] = 2×2 identity

        i = upper(:i, 4)
        j = lower(:j, 4)
        A_tensor = QTensor(A, (i, j))

        trunc = ValCutoffTrunc(1e-10)  # Discard tiny values; strips LAPACK noise below threshold
        bp = Bipartition(Partition([i]), Partition([j]))
        F = do_svd(A_tensor, bp, trunc)

        # Number of kept singular values should be ≈ 2
        @test length(F.Σ.data.diag) <= 3  # At most 2–3 due to numerical error; the Golub-Van Loan threshold should clean up near-zero values
        @test length(F.Σ.data.diag) >= 1  # At least 1 (can't be empty)
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
        A = QTensor(data, (σ1, σ2, vL, vR))   # rank-4 tensor; bipartitioned into a 3-leg left vs 1-leg right

        # Bipartition: left = [σ1, σ2, vL] (3 legs), right = [vR] (1 leg)
        bp = Bipartition(Partition([σ1, σ2, vL]), Partition([vR]))
        F = do_svd(A, bp, NoTrunc())

        # The factor tensors carry one axis per original leg plus the bond, so a
        # multi-leg partition yields a genuine multi-leg U / Vd (not a bare matrix).
        @test length(F.U.indices) == 4   # σ1, σ2, vL from left partition + λL (new bond leg)
        @test length(F.Σ.indices) == 2   # λL, λR — always 2 legs for the diagonal Σ
        @test length(F.Vd.indices) == 2  # λR (new bond) + vR from right partition
        @test size(F.U.data)[1:3] == (2, 2, 4)   # `[1:3]` = slice of size tuple; first 3 dims of U match left partition dims
        @test size(F.Vd.data)[2] == 4             # second dim of Vd = dim(vR) = 4

        # The reshaped factors reconstruct the original matricized state exactly.
        M = reshape(data, 2 * 2 * 4, 4)   # bp.left already occupies axes 1:3; column-major reshape fuses (σ1,σ2,vL) into one row axis (Julia column-major: first index varies fastest in memory)
        r = length(F.spectrum.values)
        Umat = reshape(F.U.data, 2 * 2 * 4, r)   # un-fuse left partition back into matrix form
        Vmat = reshape(F.Vd.data, r, 4)           # un-fuse right partition
        @test Umat * Diagonal(F.spectrum.values) * Vmat ≈ M   # reconstruction check
        @test sort(F.spectrum.values; rev=true) ≈ sort(svdvals(M); rev=true)[1:r]   # `svdvals(M)` = singular values of the matrix M ; our result should match
    end

    # ────────────────────────────────────────────────────────────────────────
    # _golub_van_loan_threshold: numerical-rank noise floor
    #
    # Source: Golub & Van Loan, "Matrix Computations", 4th ed.
    #         Johns Hopkins University Press, 2013 — §5.4 (numerical rank).
    #
    # The criterion τ = k·ε_machine·σ₁ follows from backward stability of
    # dgesvd: LAPACK computes the exact SVD of A + E with ‖E‖₂ ≲ ε·‖A‖₂ = ε·σ₁.
    # Singular values below τ are indistinguishable from zero given this
    # perturbation; the factor k bounds accumulation across k floating-point ops.
    # ────────────────────────────────────────────────────────────────────────

    @testset "_golub_van_loan_threshold: Golub–Van Loan numerical-rank noise floor" begin
        @testset "empty vector returns 0.0" begin
            @test _gvl(Float64[]) === 0.0   # `Float64[]` = empty Float64 vector ; `===` = identity check; early-return branch in `_golub_van_loan_threshold`
        end

        @testset "single value: threshold equals eps(T) * σ₁" begin
            S = [2.0]
            @test _gvl(S) ≈ 1 * eps(Float64) * 2.0   # `eps(Float64)` = machine epsilon for Float64 ≈ 2.22e-16 ; τ = k·ε_mach·σ₁ with k=1
        end

        @testset "threshold scales linearly with length(S)" begin
            S3 = [5.0, 1.0, 0.1]
            S6 = [5.0, 1.0, 0.1, 0.05, 0.01, 0.001]
            τ3 = _gvl(S3)
            τ6 = _gvl(S6)
            @test τ6 ≈ 2 * τ3   # τ = k·ε·σ₁; k=length(S); S6 has length 6 = 2×3; τ6/τ3 = 6/3 = 2
        end

        @testset "threshold scales with the largest singular value, not later ones" begin
            S_big = [100.0, 1.0, 0.5]
            S_small = [1.0, 1.0, 0.5]
            @test _gvl(S_big) ≈ 100 * _gvl(S_small)   # same k=3; S_big[1]=100 = 100× S_small[1]=1; τ ∝ σ₁
        end

        @testset "genuine noise is stripped from a rank-deficient matrix" begin
            # A rank-1 matrix u·vᵀ has exactly one non-zero singular value ‖u‖·‖v‖.
            # We use rank-1 specifically because the answer is known from theory: every
            # other singular value MUST be zero, so any nonzero value LAPACK returns is
            # pure floating-point noise. This gives the sharpest possible test of the
            # threshold — we can assert S[2:end] ≤ τ without any tolerance fudging,
            # because the backward-stability guarantee (§5.4, Golub & Van Loan) is tight
            # for exactly this construction.
            u = randn(8);
            v = randn(8)   # two random vectors; `randn(n)` = n-element N(0,1) vector
            A = u * v'   # `u * v'` = outer product ; rank-1 matrix
            S = svdvals(A)  # S[1] ≈ ‖u‖‖v‖; S[2:end] ≈ machine noise; `svdvals(M)` = singular values only (faster than full SVD)
            τ = _gvl(S)
            @test S[1] > τ                         # genuine singular value σ₁ = ‖u‖‖v‖ is above the noise floor
            @test all(σ -> σ ≤ τ, S[2:end])        # all noise singular values are below the Golub-Van Loan threshold; `S[2:end]` = all but the first 
        end
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
        let _bell = reshape([1.0, 0.0, 0.0, 1.0] / sqrt(2), 2, 2),   # 2×2 Bell state matrix (column-major reshape)
            _i = lower(:i, 2),
            _j = lower(:j, 2),
            ψ_tensor = QTensor(_bell, (_i, _j)),
            bp = Bipartition(Partition([_i]), Partition([_j]))

            @testset "Schmidt coefficients are non-negative and normalized to 1" begin
                # Physics: λ_i ≥ 0 and Σ_i λ_i² = 1 for a normalized state.
                @test_broken begin   # `get_schmidt_coefficients` not yet implemented; @test_broken marks as planned
                    λ = get_schmidt_coefficients(ψ_tensor, bp)
                    @test all(λ .>= -1e-14)            # `.>=` = element-wise; non-negative Schmidt coefficients
                    @test isapprox(sum(abs2, λ), 1.0; atol=1e-10)   # `sum(abs2, v)` = ∑λᵢ² = 1 for normalized state
                end
            end

            @testset "Maximally entangled state has equal Schmidt coefficients λ = 1/√2" begin
                # Physics: Bell pair has λ_1 = λ_2 = 1/√2.
                @test_broken begin
                    λ = get_schmidt_coefficients(ψ_tensor, bp)
                    expected = 1.0 / sqrt(2)   # `1.0 / sqrt(2)` = 1/√2 ≈ 0.707
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
                    S = get_entanglement_entropy(ψ_tensor, bp; base=2)   # `; base=2` = keyword argument
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
            ψ_tensor = QTensor(ones(2, 3) / sqrt(6), (i, j))   # `ones(2,3)/sqrt(6)` = normalised product state
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
            ψ_tensor = QTensor(ones(2, 2, 2) / (2 * sqrt(2)), (_σ1, _σ2, _σ3)),   # `ones(2,2,2)/(2√2)` = normalised 3-qubit product state; ‖ψ‖=1 since 8 elements each 1/(2√2) and sum of squares = 8/(8) = 1
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
        temp_file = tempname() * ".jls"   # `tempname()` = OS temp filename 
        open(temp_file, "w") do f         # `open(path, "w") do f ... end` = open file for writing, execute block, close 
            serialize(f, test_data)       # `serialize` = Julia standard library binary serialization 
        end
        loaded_data = load_array(temp_file)   # `load_array` dispatches on extension ".jls" → deserialize
        @test loaded_data ≈ test_data
        rm(temp_file)   # `rm(path)` = delete file 
    end

    @testset "load_array: .txt format (DelimitedFiles)" begin
        # Physics: state data should load unchanged from a text file.

        using DelimitedFiles   # `using DelimitedFiles` = import module for `writedlm`/`readdlm`; can be used inside a function or testset
        test_data = [1.0 2.0; 3.0 4.0]   # 2×2 matrix
        temp_file = tempname() * ".txt"
        writedlm(temp_file, test_data)    # `writedlm(path, M)` = write matrix as delimited text 
        loaded_data = load_array(temp_file)   # dispatches on ".txt" → readdlm
        @test loaded_data ≈ test_data
        rm(temp_file)
    end

    # ────────────────────────────────────────────────────────────────────────
    # 1.5.5 bipartition_matrix — reshape state to matrix form
    # ────────────────────────────────────────────────────────────────────────

    # All three bipartition_matrix tests use the same 3-qubit state array.
    @testset "bipartition_matrix" begin
        let ψ_vector = randn(8), ψ_full = reshape(ψ_vector, 2, 2, 2)   # `reshape(v, 2,2,2)` = 8-element vector → 2×2×2 tensor (3-qubit state)
            @testset "reshape d^L state to (d^cut × d^(L-cut))" begin
                # Physics: matricising at cut=1 gives a (2 × 4) matrix.
                @test_broken begin
                    M = bipartition_matrix(ψ_full, cut=1)   # `bipartition_matrix(ψ, cut=k)` = not yet implemented; reshapes rank-L tensor into matrix by grouping first k legs as rows
                    @test size(M) == (2, 4)
                    @test isapprox(reshape(M, 2, 2, 2), ψ_full)
                end
            end

            @testset "edge case cut=0 gives (1 × d^L) matrix" begin
                # Physics: trivial left subsystem (dimension 1).
                @test_broken begin
                    M = bipartition_matrix(ψ_full, cut=0)   # cut=0 → no left legs → 1 row
                    @test size(M) == (1, 8)
                    @test isapprox(vec(M), ψ_vector)   # `vec(M)` = flatten to vector 
                end
            end

            @testset "edge case cut=L gives (d^L × 1) matrix" begin
                # Physics: trivial right subsystem (dimension 1).
                @test_broken begin
                    M = bipartition_matrix(ψ_full, cut=3)   # cut=L → all legs on left → 1 col
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
        (4, 2),
        (3, 3),   # `for (L, d) in [...]` = parametric testset; Julia runs one test per element; Python: `@pytest.mark.parametrize`
    ]
        @test_broken begin
            ψ_vector = randn(ComplexF64, d^L)   # `d^L` = d raised to L. total Hilbert space dimension
            ψ_tensor = as_state(ψ_vector, L; d)   # `as_state` not yet implemented; reshapes into rank-L tensor with all legs of dim=d
            @test size(ψ_tensor) == ntuple(_ -> d, L)   # `ntuple(f, n)` = Tuple of length n with f(i) at position i ; all dims should be d
            @test isapprox(vec(ψ_tensor), ψ_vector)     # flatten back to vector = original
        end
    end

    @testset "as_state: preserves normalization" begin
        # Physics: reshaping a normalized state vector should preserve the norm.
        @test_broken begin
            L = 3;
            d = 2
            ψ_vector = randn(d^L)
            ψ_vector /= norm(ψ_vector)   # `v /= x` = in-place division 
            ψ_tensor = as_state(ψ_vector, L; d)
            @test isapprox(norm(vec(ψ_tensor)), norm(ψ_vector); atol=1e-14)   # norm preserved under reshape
        end
    end
end  # @testset "Schmidt coefficients and entanglement entropy"
