# Tests for §10.2 — ED time evolution: exp(-iHt)|ψ⟩
using Random

@testset "§10.2 ED time evolution" begin

    @testset "norm is preserved under RealTime ED propagation" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        psi_vec = normalize(randn(ComplexF64, 2^4))
        p = ConstantProtocol(RealTime(), 0.1, 10, H)
        result = solve(H, as_statevector(psi_vec), ExactDiagonalization(:time), p)
        @test norm(result.state) ≈ 1.0  atol=1e-10
    end

    @testset "energy is conserved under RealTime ED propagation" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        M = matrix_repr(H)
        psi_vec = normalize(randn(ComplexF64, 2^4))
        E_init = real(psi_vec' * M * psi_vec)

        p = ConstantProtocol(RealTime(), 0.05, 20, H)
        result = solve(H, as_statevector(psi_vec), ExactDiagonalization(:time), p)
        E_final = real(result.state' * M * result.state)
        @test E_final ≈ E_init  atol=1e-8
    end

    @testset "ImaginaryTime ED propagation converges to GS" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        M = matrix_repr(H)
        E_gs = minimum(real.(eigvals(Hermitian(M))))

        psi_vec = normalize(randn(ComplexF64, 2^4))
        # Exact ED: total time is dt*nsteps, no Trotter error — use one large step τ=20
        p = ConstantProtocol(ImaginaryTime(), 20.0, 1, H)
        result = solve(H, as_statevector(psi_vec), ExactDiagonalization(:time), p)
        psi_f = normalize(result.state)
        E_final = real(psi_f' * M * psi_f)
        @test E_final ≈ E_gs  atol=1e-6
    end

    @testset "RealTime ED matches fine-dt TEBD on L=4 Heisenberg" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        mpo = MPO(H)

        # Both start from the same state
        psi_vec = normalize(randn(MersenneTwister(7), ComplexF64, 2^4))
        ψ_mps = to_mps(as_state(psi_vec, [2,2,2,2]); trunc=NoTrunc(), form=:left)

        t = 0.5  # total time
        dt_tebd = 0.01  # fine enough for Trotter error to be small

        # ED propagation
        p_ed = ConstantProtocol(RealTime(), t, 1, H)  # single step by exact matrix exp
        result_ed = solve(H, as_statevector(psi_vec), ExactDiagonalization(:time), p_ed)
        E_ed = real(result_ed.state' * matrix_repr(H) * result_ed.state) / real(result_ed.state' * result_ed.state)

        # TEBD (fine dt, 2nd-order Trotter)
        p_tebd = ConstantProtocol(RealTime(), dt_tebd, round(Int, t/dt_tebd), H)
        result_tebd = solve(H, Quench(ψ_mps), TEBD(SuzukiTrotter(2), NoTrunc()), p_tebd)
        ψ_f = result_tebd.state
        E_tebd = real(expect(ψ_f, mpo)) / real(overlap(ψ_f, ψ_f))

        # Energies should match (energy is conserved, so both give E_init)
        E_init = real(psi_vec' * matrix_repr(H) * psi_vec)
        @test E_ed ≈ E_init  atol=1e-8
        @test E_tebd ≈ E_init  atol=1e-4  # TEBD has Trotter error
    end

end
