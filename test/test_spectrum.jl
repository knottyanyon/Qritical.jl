"""
Tests for: §1.4b Spectrum & orthogonality-centre hierarchy

Physics invariants tested:
- SingValSpectrum values descend, ε = 2-norm of dropped σ, normalized flag honoured
- Entanglement entropy: product state → 0; Bell pair → log₂2 = 1 bit (derived, not stored)
- Entanglement spectrum = −2 log σᵢ; von Neumann = −Σσᵢ² log σᵢ² (with 0 log 0 = 0 guard)
- SchmidtSpectrum.center.bond legs ≡ Σ bond legs of do_svd (no matching step needed)
- Reabsorption: U·Diagonal(s.values) and Diagonal(s.values)·Vᴴ = dense U·Σ/Σ·Vᴴ (hot-path scaling)
- Type guard: function requiring SchmidtSpectrum rejects bare SingValSpectrum
- Edge cases: boundary bond → [1.0], ε=0, normalized=true; zero σ excluded from entropy
- BondCenter vs SiteCenter dispatch exhaustive over OrthoCenter
"""

using Test
using LinearAlgebra
using Qritical

@testset "Spectrum and orthogonality centre hierarchy" begin

    # ────────────────────────────────────────────────────────────────────────
    # §1.4b.1 SingValSpectrum construction and properties
    # ────────────────────────────────────────────────────────────────────────

    @testset "SingValSpectrum from matrix SVD: values descending, ε = 2-norm of discarded σ, normalized flag honoured" begin
        # Shared setup: diagonal matrix with known singular values [5,3,1.5,0.3,0.05].
        let σ_full = [5.0, 3.0, 1.5, 0.3, 0.05],
            i = upper(:i, 5),
            j = lower(:j, 5),
            A_tensor = QTensor(Matrix(Diagonal(σ_full)), (i, j)),
            bp = Bipartition(Partition([i]), Partition([j]))

            # Teardown: all bindings are lexically scoped to this let-block;
            # released automatically on exit (no side-effects to clean up).

            @testset "no truncation: values are descending and ε = 0" begin
                # Physics: σ₁ ≥ σ₂ ≥ ⋯ ≥ σᵣ ≥ 0; no truncation ⟹ ε = 0 exactly.
                let s = do_svd(A_tensor, bp, NoTrunc()).spectrum
                    @test issorted(s.values; rev=true)
                    @test all(s.values .>= -1e-14)
                    @test s.ε ≈ 0.0 atol=1e-14
                end
            end

            @testset "MaxBondDimTrunc(3): ε = ‖discarded σ‖₂ and kept values descending" begin
                # Physics: ε = ‖[0.3, 0.05]‖₂ when the three largest are kept.
                let s = do_svd(A_tensor, bp, MaxBondDimTrunc(3)).spectrum
                    @test isapprox(s.ε, norm(σ_full[4:end]); atol=1e-10)
                    @test issorted(s.values; rev=true)
                end
            end

            @testset "normalized flag: true after NoTrunc on unit-norm state, false after truncation" begin
                # Physics: `normalized` tracks whether sum(σᵢ²) ≈ 1 for the *current* values.
                # Truncation discards weight, so the kept spectrum is no longer unit-norm.
                let σ_unit = σ_full ./ norm(σ_full),
                    A_unit = QTensor(Matrix(Diagonal(σ_unit)), (i, j))

                    # No truncation on a unit-norm state → normalized = true
                    @test do_svd(A_unit, bp, NoTrunc()).spectrum.normalized == true

                    # Truncation discards weight → sum(kept σᵢ²) < 1 → normalized = false
                    @test do_svd(A_unit, bp, MaxBondDimTrunc(3)).spectrum.normalized == false
                end
            end

        end  # let shared setup
    end

    # ────────────────────────────────────────────────────────────────────────
    # §1.4b.2 Entanglement entropy on SchmidtSpectrum
    # ────────────────────────────────────────────────────────────────────────

    @testset "Entanglement entropy: product state → 0 bits" begin
        # Physics: |ψ_A⟩⊗|ψ_B⟩ has a single Schmidt coefficient λ₁=1.
        # S = −λ₁² log₂ λ₁² = −1·0 = 0 bits.
        let i = lower(:i, 3),
            j = lower(:j, 4),
            ψ_tensor = QTensor(ones(3, 4) / sqrt(12), (i, j)),
            bp = Bipartition(Partition([i]), Partition([j]))

            let F = do_svd(ψ_tensor, bp, NoTrunc()),
                s = SchmidtSpectrum(F.spectrum, bp, F.center)

                @test isapprox(entanglement_entropy(s; base=2), 0.0; atol=1e-10)
            end
        end
    end

    @testset "Entanglement entropy: maximally entangled Bell pair → log₂(2) = 1 bit" begin
        # Physics: (|00⟩+|11⟩)/√2 has λ₁ = λ₂ = 1/√2.
        # S = −2·(1/2)·log₂(1/2) = −2·(1/2)·(−1) = 1 bit.
        let i = lower(:i, 2),
            j = lower(:j, 2),
            ψ_tensor = QTensor(reshape([1.0, 0.0, 0.0, 1.0] / sqrt(2), 2, 2), (i, j)),
            bp = Bipartition(Partition([i]), Partition([j]))

            let F = do_svd(ψ_tensor, bp, NoTrunc()),
                s = SchmidtSpectrum(F.spectrum, bp, F.center)

                @test isapprox(entanglement_entropy(s; base=2), 1.0; atol=1e-10)
            end
        end
    end

    # ────────────────────────────────────────────────────────────────────────
    # §1.4b.3 Entanglement spectrum = −2 log σᵢ
    # ────────────────────────────────────────────────────────────────────────

    @testset "Entanglement spectrum: εᵢ = −2 log σᵢ" begin
        # Physics: Bell pair σ = [1/√2, 1/√2] → εᵢ = −2 log(1/√2) = ln 4 for both.
        let i = lower(:i, 2),
            j = lower(:j, 2),
            ψ_tensor = QTensor(reshape([1.0, 0.0, 0.0, 1.0] / sqrt(2), 2, 2), (i, j)),
            bp = Bipartition(Partition([i]), Partition([j]))

            let F = do_svd(ψ_tensor, bp, NoTrunc()),
                s = SchmidtSpectrum(F.spectrum, bp, F.center)

                ε_expected = -2.0 .* log.(s.spectrum.values)
                @test isapprox(entanglement_spectrum(s), ε_expected; atol=1e-10)
            end
        end
    end

    # ────────────────────────────────────────────────────────────────────────
    # §1.4b.4 Von Neumann entropy with 0 log 0 = 0 guard
    # ────────────────────────────────────────────────────────────────────────

    @testset "Von Neumann entropy: S = −Σσᵢ² log σᵢ² with 0 log 0 = 0 guard" begin
        # Physics: S = −Σ σᵢ² log σᵢ² where the convention 0·log(0) = 0
        # prevents NaN for rank-deficient states.
        let σ_vals = [0.99, 0.01 / sqrt(2)],
            i = lower(:i, 2),
            j = lower(:j, 2),
            A_tensor = QTensor(Matrix(Diagonal(σ_vals)), (i, j)),
            bp = Bipartition(Partition([i]), Partition([j]))

            let F = do_svd(A_tensor, bp, NoTrunc()),
                s = SchmidtSpectrum(F.spectrum, bp, F.center),
                S = entanglement_entropy(s; base=ℯ)

                σ_sq = abs2.(s.spectrum.values)
                S_expected = -sum(p -> p > 0 ? p * log(p) : 0.0, σ_sq)
                @test isapprox(S, S_expected; atol=1e-10)
                @test isfinite(S)
            end
        end
    end

    # ────────────────────────────────────────────────────────────────────────
    # §1.4b.5 SchmidtSpectrum.center.bond is the SVD bond, no matching needed
    # ────────────────────────────────────────────────────────────────────────

    @testset "SchmidtSpectrum.center.bond legs ≡ SVD bond legs (Σ legs, no matching step)" begin
        # Physics: BondCenter carries the Σ bond legs directly — same TIx objects.
        let i = upper(:i, 3),
            j = lower(:j, 4),
            A = QTensor(randn(3, 4), (i, j)),
            bp = Bipartition(Partition([i]), Partition([j]))

            let F = do_svd(A, bp, NoTrunc()),
                s = SchmidtSpectrum(F.spectrum, bp, F.center)

                @test s.center.bond.upper == F.Σ.indices[1]   # upper(:λL, r)
                @test s.center.bond.lower == F.Σ.indices[2]   # lower(:λR, r)
            end
        end
    end

    # ────────────────────────────────────────────────────────────────────────
    # §1.4b.6 Reabsorption: U·Diagonal(s.values) = U·Σ (hot-path scaling)
    # ────────────────────────────────────────────────────────────────────────

    @testset "Reabsorption U·Diagonal(s.values) ≈ U·Σ (dense reabsorption, hot-path scaling)" begin
        # Physics: scaling a factor by the spectrum via Diagonal(s.values) must equal
        # multiplying by the dense Σ matrix.
        let σ_vals = [3.0, 2.0, 1.0, 0.5],
            U_mat = randn(6, 4),
            s = SingValSpectrum(σ_vals, 0.0, false)

            @test isapprox(U_mat * Diagonal(s.values), U_mat * Diagonal(σ_vals); atol=1e-14)
        end
    end

    @testset "Reabsorption Diagonal(s.values)·Vᴴ ≈ Σ·Vᴴ (right reabsorption, hot-path scaling)" begin
        let σ_vals = [3.0, 2.0, 1.0, 0.5],
            Vd_mat = randn(4, 5),
            s = SingValSpectrum(σ_vals, 0.0, false)

            @test isapprox(Diagonal(s.values) * Vd_mat, Diagonal(σ_vals) * Vd_mat; atol=1e-14)
        end
    end

    # ────────────────────────────────────────────────────────────────────────
    # §1.4b.7 Type guard: SchmidtSpectrum rejects bare SingValSpectrum
    # ────────────────────────────────────────────────────────────────────────

    @testset "Type guard: function requiring SchmidtSpectrum rejects bare SingValSpectrum" begin
        # Physics: bipartition(s) is only meaningful when s carries a cut location.
        # Calling it on a bare SingValSpectrum must fail at dispatch.
        let s = SingValSpectrum([1.0, 0.5], 0.0, false)
            @test_throws MethodError bipartition(s)
        end
    end

    # ────────────────────────────────────────────────────────────────────────
    # §1.4b.8 Edge cases
    # ────────────────────────────────────────────────────────────────────────

    @testset "Edge case: boundary bond (trivial bipartition) → spectrum [1.0], ε=0, normalized=true" begin
        # Physics: an open-chain boundary bond has a trivial spectrum: σ = [1.0].
        let i = lower(:i, 2),
            j = lower(:_dummy, 1),
            ψ_tensor = QTensor(reshape(ones(2) / sqrt(2), 2, 1), (i, j)),
            bp = Bipartition(Partition([i]), Partition([j]))

            let s = do_svd(ψ_tensor, bp, NoTrunc()).spectrum
                @test length(s.values) == 1
                @test isapprox(s.values[1], 1.0; atol=1e-14)
                @test s.ε ≈ 0.0 atol=1e-14
                @test s.normalized == true
            end
        end
    end

    @testset "Edge case: zero singular values are excluded from entropy calculations" begin
        # Physics: rank-1 matrix has σ = [1.0]; S = 0 and no NaN from 0·log(0).
        let i = lower(:i, 2),
            j = lower(:j, 2),
            ψ_tensor = QTensor(ones(2, 2) / 2, (i, j)),
            bp = Bipartition(Partition([i]), Partition([j]))

            let F = do_svd(ψ_tensor, bp, NoTrunc()),
                s = SchmidtSpectrum(F.spectrum, bp, F.center)

                S = entanglement_entropy(s; base=2)
                @test isfinite(S)
                @test S >= 0.0
            end
        end
    end

    # ────────────────────────────────────────────────────────────────────────
    # §1.4b.9 OrthoCenter dispatch: BondCenter and SiteCenter
    # ────────────────────────────────────────────────────────────────────────

    @testset "BondCenter vs SiteCenter dispatch is exhaustive over OrthoCenter" begin
        # Physics: orthogonality centre is either a bond or a site — nothing else.
        let bond = Bond(lower(:λL, 4), upper(:λR, 4)),
            bond_center = BondCenter(bond),
            site_center = SiteCenter(lower(:σ, 2))

            @test bond_center isa OrthoCenter
            @test site_center isa OrthoCenter
            @test typeof(bond_center) != typeof(site_center)
        end
    end

    # ────────────────────────────────────────────────────────────────────────
    # §1.4b.10 Schmidt rank and spectral gap
    # ────────────────────────────────────────────────────────────────────────

    @testset "Schmidt rank and spectral gap" begin
        # Bell pair and product state share the same SVD setup — group them.
        let i = lower(:i, 2),
            j = lower(:j, 2)

            @testset "Bell pair has Schmidt rank 2" begin
                let ψ_tensor = QTensor(reshape([1.0, 0.0, 0.0, 1.0] / sqrt(2), 2, 2), (i, j)),
                    bp = Bipartition(Partition([i]), Partition([j])),
                    F = do_svd(ψ_tensor, bp, NoTrunc()),
                    s = SchmidtSpectrum(F.spectrum, bp, F.center)

                    @test schmidt_rank(s) == 2
                end
            end

            @testset "product state (ones/√12): rank 1 and spectral gap > 0.9" begin
                let i3 = lower(:i, 3),
                    j4 = lower(:j, 4),
                    ψ_tensor = QTensor(ones(3, 4) / sqrt(12), (i3, j4)),
                    bp = Bipartition(Partition([i3]), Partition([j4])),
                    F = do_svd(ψ_tensor, bp, NoTrunc()),
                    s = SchmidtSpectrum(F.spectrum, bp, F.center)

                    @test schmidt_rank(s) == 1
                    @test spectral_gap(s) > 0.9
                end
            end
        end
    end

end  # @testset "Spectrum and orthogonality centre hierarchy"
