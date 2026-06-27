# Tests for §10 — ExactDiagonalization, sparse/dense solve.
# Physics invariants: real eigenvalues, GS energy bound, MPS ↔ ED cross-check.

using SparseArrays

@testset "§10.1 dense Hamiltonian matrix" begin

    @testset "dense_matrix is Hermitian" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0, h=0.1)
        M = dense_matrix(H)
        @test M ≈ M'  atol=1e-12
    end

    @testset "dense_matrix eigenvalues are real" begin
        g = Chain(4)
        H = XXZ(g; J=1.0, Jz=0.5, h=0.2)
        M = dense_matrix(H)
        ev = eigvals(Hermitian(M))
        @test all(abs.(imag.(eigvals(M))) .< 1e-10)
    end

    @testset "sparse(H) and dense_matrix(H) agree" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        M_dense  = dense_matrix(H)
        M_sparse = sparse(H)
        @test M_dense ≈ Matrix(M_sparse)  atol=1e-12
    end

    @testset "sparse(H) is Hermitian" begin
        g = Chain(4)
        H = XXZ(g; J=1.0, Jz=0.5, h=0.0)
        M = sparse(H)
        @test M ≈ M'  atol=1e-12
    end

end

@testset "§10.1 ExactDiagonalization — ground state" begin

    @testset "solve(H, GroundState(), ED(:ground)) returns EDResult" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        result = solve(H, GroundState(), ExactDiagonalization(:ground))
        @test result isa EDResult
        @test result.energy isa Float64
        @test result.state isa Vector{ComplexF64}
    end

    @testset "GS energy ≤ all eigenvalues" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        result = solve(H, GroundState(), ExactDiagonalization(:ground))
        all_ev = real.(eigvals(Hermitian(dense_matrix(H))))
        @test result.energy ≤ minimum(all_ev) + 1e-8
    end

    @testset "ED(:full) returns complete spectrum" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        result = solve(H, GroundState(), ExactDiagonalization(:full))
        d = local_dim(SpinHalf())
        @test length(result.spectrum) == d^3   # 2^3 = 8 eigenvalues
    end

    @testset "ED(:ground) GS energy matches eigmin on L=4 Heisenberg" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        E_ed = minimum(real.(eigvals(Hermitian(dense_matrix(H)))))
        result = solve(H, GroundState(), ExactDiagonalization(:ground))
        @test result.energy ≈ E_ed  atol=1e-8
    end

    @testset "ED GS energy matches power method (L=4)" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        ed_result = solve(H, GroundState(), ExactDiagonalization(:ground))

        ψ_init = let
            psi_vec = normalize(randn(ComplexF64, 2^4))
            to_mps(as_state(psi_vec, [2,2,2,2]); trunc=NoTrunc(), form=:left)
        end
        pm_result = power_method(H, ψ_init; shift=4.0, tol=1e-8, maxiter=200,
                                  trunc=MaxBondDimTrunc(16))
        @test pm_result.energy ≈ ed_result.energy  atol=1e-4
    end

    @testset "GS state is normalized" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        result = solve(H, GroundState(), ExactDiagonalization(:ground))
        @test norm(result.state) ≈ 1.0  atol=1e-10
    end

    @testset "XXZ GS energy at Δ=1 (isotropic) matches exact result L=4" begin
        # For L=4 OBC Heisenberg, E0 = -1.6160254...
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        result = solve(H, GroundState(), ExactDiagonalization(:ground))
        @test result.energy ≈ -1.6160254037844388  atol=1e-8
    end

end

@testset "§10.1 sparse operator size guard" begin
    @testset "large Hilbert space is rejected" begin
        # If local_dim^L > 2^20 (≈1M), sparse should refuse
        g = Chain(25)   # d=2 → 2^25 > 1M
        H = Heisenberg(g; J=1.0)
        @test_throws ArgumentError sparse(H)
    end
end
