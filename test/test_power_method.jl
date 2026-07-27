# Tests for §6.3 — operator arithmetic, PowerMethod, and identity expect.
# Physics invariants: GS energy convergence, operator algebra.

@testset "§6.3 LatticeOperator arithmetic" begin
    @testset "scalar * LatticeOperator scales all couplings" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        H2 = 2.0 * H   # multiply operator by scalar; `*` dispatches on (Float64, LatticeOperator) via `Base.:*` extension
        for (lt1, lt2) in zip(H.onsite, H2.onsite)   # `zip(a, b)` pairs elements of two iterables (same as Python); destructure with `(lt1, lt2)`
            @test lt2.coupling ≈ 2.0 * lt1.coupling atol=1e-12   # each on-site coupling should be doubled
        end
        for (bt1, bt2) in zip(H.bond, H2.bond)
            @test bt2.coupling ≈ 2.0 * bt1.coupling atol=1e-12   # each bond coupling should be doubled
        end
    end

    @testset "LatticeOperator + LatticeOperator merges term lists" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0, h=0.0)
        M = total_magnetization(g)   # observable: Σ Sz_i
        # H + M should have onsite terms from both
        HM = H + M   # `+` dispatches on (LatticeOperator, LatticeOperator) via `Base.:+` extension; concatenates term lists
        @test length(HM.onsite) == length(H.onsite) + length(M.onsite)   # total onsite terms = sum of both; physics: H+M adds the Sz fields to the Heisenberg Hamiltonian
        @test length(HM.bond) == length(H.bond) + length(M.bond)   # same for bond terms (M has no bond terms, so this just checks H.bond count is unchanged)
    end

    @testset "LatticeOperator + LatticeOperator — energy additivity on product state" begin
        # ⟨ψ|(H + M)|ψ⟩ = ⟨ψ|H|ψ⟩ + ⟨ψ|M|ψ⟩
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        M = total_magnetization(g)
        mat_H = matrix_repr(H)   # 8×8 dense matrix for H
        mat_M = matrix_repr(M)   # 8×8 dense matrix for M = Σ Sz_i
        mat_HM = matrix_repr(H + M)   # matrix for the combined operator H + M
        @test mat_HM ≈ mat_H + mat_M atol=1e-10   # matrix representation must be linear: matrix(H+M) = matrix(H) + matrix(M); this tests both the `+` operator and `matrix_repr`
    end

    @testset "expect(ψ, identity_operator) ≈ ‖ψ‖²" begin
        g = Chain(3)
        ψ_vec = normalize(randn(ComplexF64, 2^3))   # random unit vector; `normalize(v)` = v/‖v‖
        ψ_t = as_state(ψ_vec, [2, 2, 2])   # wrap as a state tensor with given local dimensions
        ψ = to_mps(ψ_t; trunc=NoTrunc(), form=:left)   # convert to left-canonical MPS
        Id_op = identity_operator(g, SpinHalf())   # identity operator as a LatticeOperator (empty term list)
        mpo = MPO(Id_op)   # build the χ=1 all-identity MPO
        @test real(expect(ψ, mpo)) ≈ 1.0 atol=1e-8   # ⟨ψ|I|ψ⟩ = ‖ψ‖² = 1 for a normalised state; tests that the identity MPO gives the right expectation value and that `expect` computes ⟨ψ|O|ψ⟩ correctly
    end
end

@testset "§6.3 PowerMethod — GS energy convergence" begin
    @testset "PowerMethod converges to ED GS energy on L=4 Heisenberg" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        mpo = MPO(H)   # build MPO for energy measurement during power method
        # ED reference
        E_ed = minimum(real.(eigvals(matrix_repr(H))))   # exact GS energy from full dense diagonalisation; `real.(...)` converts complex eigenvalues to real (imaginary parts should be ~0); `minimum(...)` picks the smallest

        # Run power method (imaginary-time evolution by repeated (shift*I - H) application)
        ψ_init = let   # local scope for the initial state construction
            psi_vec = normalize(randn(ComplexF64, 2^4))   # random initial state; power method works from any initial state with nonzero GS overlap
            to_mps(as_state(psi_vec, [2, 2, 2, 2]); trunc=NoTrunc(), form=:left)   # convert to left-canonical MPS without truncation
        end
        result = power_method(
            H, ψ_init; shift=4.0, tol=1e-8, maxiter=200, trunc=MaxBondDimTrunc(16)
        )   # `MaxBondDimTrunc(16)` limits bond dim to 16; `shift=4.0` must exceed the spectral radius of H (for L=4 Heisenberg this is ~2)
        @test result.energy ≈ E_ed atol=1e-4   # tensor-network power method must agree with exact diagonalisation; `atol=1e-4` is looser than ED-ED comparison because bond truncation introduces an extra approximation error
    end

    @testset "PowerMethod energy is real (Hermitian H)" begin
        g = Chain(3)
        H = XXZ(g; J=1.0, Jz=0.5, h=0.2)
        mpo = MPO(H)
        ψ_init = let
            psi_vec = normalize(randn(ComplexF64, 2^3))
            to_mps(as_state(psi_vec, [2, 2, 2]); trunc=NoTrunc(), form=:left)
        end
        result = power_method(
            H, ψ_init; shift=3.0, tol=1e-6, maxiter=100, trunc=MaxBondDimTrunc(8)
        )
        @test abs(imag(result.energy)) < 1e-8   # Rayleigh quotient ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ should be real for Hermitian H; any imaginary part is numerical noise; `imag(x)` extracts the imaginary part
    end

    @testset "PowerMethod result energy ≤ initial energy" begin
        # The power method can only lower the energy (it filters toward GS)
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        mpo = MPO(H)
        ψ_init = let
            psi_vec = normalize(randn(ComplexF64, 2^3))
            to_mps(as_state(psi_vec, [2, 2, 2]); trunc=NoTrunc(), form=:left)
        end
        E_init = real(expect(ψ_init, mpo))   # energy of the random initial state; no division by norm here because ψ_init is left-canonical (‖ψ‖² = 1)
        result = power_method(
            H, ψ_init; shift=3.0, tol=1e-6, maxiter=50, trunc=MaxBondDimTrunc(8)
        )
        @test result.energy ≤ E_init + 1e-8   # variational principle: any iteration of the power method can only lower (or maintain) the energy; the GS energy is the global minimum of the Rayleigh quotient
    end
end
