# Tests for §6.3 — operator arithmetic, PowerMethod, and identity expect.
# Physics invariants: GS energy convergence, operator algebra.

@testset "§6.3 Operator arithmetic" begin

    @testset "scalar * Operator scales all couplings" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        H2 = 2.0 * H
        for (lt1, lt2) in zip(H.onsite, H2.onsite)
            @test lt2.coupling ≈ 2.0 * lt1.coupling  atol=1e-12
        end
        for (bt1, bt2) in zip(H.bond, H2.bond)
            @test bt2.coupling ≈ 2.0 * bt1.coupling  atol=1e-12
        end
    end

    @testset "Operator + Operator merges term lists" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0, h=0.0)
        M = total_magnetization(g)
        # H + M should have onsite terms from both
        HM = H + M
        @test length(HM.onsite) == length(H.onsite) + length(M.onsite)
        @test length(HM.bond)   == length(H.bond)   + length(M.bond)
    end

    @testset "Operator + Operator — energy additivity on product state" begin
        # ⟨ψ|(H + M)|ψ⟩ = ⟨ψ|H|ψ⟩ + ⟨ψ|M|ψ⟩
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        M = total_magnetization(g)
        mat_H  = dense_matrix(H)
        mat_M  = dense_matrix(M)
        mat_HM = dense_matrix(H + M)
        @test mat_HM ≈ mat_H + mat_M  atol=1e-10
    end

    @testset "expect(ψ, identity_operator) ≈ ‖ψ‖²" begin
        g = Chain(3)
        ψ_vec = normalize(randn(ComplexF64, 2^3))
        ψ_t   = as_state(ψ_vec, [2, 2, 2])
        ψ     = to_mps(ψ_t; trunc=NoTrunc(), form=:left)
        Id_op = identity_operator(g, SpinHalf())
        mpo   = MPO(Id_op)
        @test real(expect(ψ, mpo)) ≈ 1.0  atol=1e-8
    end

end

@testset "§6.3 PowerMethod — GS energy convergence" begin

    @testset "PowerMethod converges to ED GS energy on L=4 Heisenberg" begin
        g   = Chain(4)
        H   = Heisenberg(g; J=1.0)
        mpo = MPO(H)
        # ED reference
        E_ed = minimum(real.(eigvals(dense_matrix(H))))

        # Run power method (imaginary-time evolution by repeated (shift*I - H) application)
        ψ_init = let
            psi_vec = normalize(randn(ComplexF64, 2^4))
            to_mps(as_state(psi_vec, [2,2,2,2]); trunc=NoTrunc(), form=:left)
        end
        result = power_method(H, ψ_init; shift=4.0, tol=1e-8, maxiter=200,
                              trunc=MaxBondDimTrunc(16))
        @test result.energy ≈ E_ed  atol=1e-4
    end

    @testset "PowerMethod energy is real (Hermitian H)" begin
        g   = Chain(3)
        H   = XXZ(g; J=1.0, Jz=0.5, h=0.2)
        mpo = MPO(H)
        ψ_init = let
            psi_vec = normalize(randn(ComplexF64, 2^3))
            to_mps(as_state(psi_vec, [2,2,2]); trunc=NoTrunc(), form=:left)
        end
        result = power_method(H, ψ_init; shift=3.0, tol=1e-6, maxiter=100,
                              trunc=MaxBondDimTrunc(8))
        @test abs(imag(result.energy)) < 1e-8
    end

    @testset "PowerMethod result energy ≤ initial energy" begin
        # The power method can only lower the energy (it filters toward GS)
        g   = Chain(3)
        H   = Heisenberg(g; J=1.0)
        mpo = MPO(H)
        ψ_init = let
            psi_vec = normalize(randn(ComplexF64, 2^3))
            to_mps(as_state(psi_vec, [2,2,2]); trunc=NoTrunc(), form=:left)
        end
        E_init = real(expect(ψ_init, mpo))
        result = power_method(H, ψ_init; shift=3.0, tol=1e-6, maxiter=50,
                              trunc=MaxBondDimTrunc(8))
        @test result.energy ≤ E_init + 1e-8
    end

end
