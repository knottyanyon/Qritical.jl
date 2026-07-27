# Tests for §10.2 — ED time evolution: exp(-iHt)|ψ⟩
using Random   # Julia's standard library for RNGs; `using Random` imports `MersenneTwister`, `randn`, etc.

@testset "§10.2 ED time propagation" begin
    @testset "norm is preserved under RealTime ED propagation" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        psi_vec = normalize(randn(ComplexF64, 2^4))   # random unit vector in 2^4=16 dimensional Hilbert space; `normalize(v)` = v/‖v‖
        p = ConstantProtocol(RealTime(), 0.1, 10, H)   # real-time protocol: dt=0.1, 10 steps → total time T=1.0
        result = solve(H, as_statevector(psi_vec), ExactDiagonalization(:time), p)   # `as_statevector` wraps the vector as a StatevectorState; solve dispatches on (LatticeOperator, StatevectorState, ExactDiagonalization{:time}, ConstantProtocol)
        @test norm(result.state) ≈ 1.0  atol=1e-10   # `norm(v)` = ‖v‖₂; unitary evolution preserves the norm: ‖e^{-iHt}ψ‖ = ‖ψ‖ = 1 because H is Hermitian → U=e^{-iHt} is unitary
    end

    @testset "energy is conserved under RealTime ED propagation" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        M = matrix_repr(H)   # build the full 16×16 Hamiltonian matrix for energy measurement
        psi_vec = normalize(randn(ComplexF64, 2^4))
        E_init = real(psi_vec' * M * psi_vec)   # initial energy: ⟨ψ|H|ψ⟩; `psi_vec'` is the conjugate transpose row vector ` discards imaginary rounding noise

        p = ConstantProtocol(RealTime(), 0.05, 20, H)   # dt=0.05, 20 steps → T=1.0
        result = solve(H, as_statevector(psi_vec), ExactDiagonalization(:time), p)
        E_final = real(result.state' * M * result.state)   # final energy: ⟨ψ(T)|H|ψ(T)⟩
        @test E_final ≈ E_init  atol=1e-8   # energy conservation: for unitary evolution [H, U]=0 → ⟨H⟩ constant in time; this is an exact conservation law (no Trotter error in ED)
    end

    @testset "ImaginaryTime ED propagation converges to GS" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        M = matrix_repr(H)
        E_gs = minimum(real.(eigvals(Hermitian(M))))   # exact GS energy from dense diagonalisation

        psi_vec = normalize(randn(ComplexF64, 2^4))   # random initial state (has nonzero GS overlap with probability 1)
        # Exact ED: total time is dt*nsteps, no Trotter error — use one large step τ=20
        p = ConstantProtocol(ImaginaryTime(), 20.0, 1, H)   # imaginary time: τ=20.0 in a single step; one large step is fine because ED uses the exact matrix exponential (no Trotter splitting); `ImaginaryTime()` selects exp(−τH) instead of exp(−iτH)
        result = solve(H, as_statevector(psi_vec), ExactDiagonalization(:time), p)
        psi_f = normalize(result.state)   # re-normalise after imaginary-time evolution (imaginary time is not unitary → changes norm)
        E_final = real(psi_f' * M * psi_f)   # energy of the final normalised state
        @test E_final ≈ E_gs  atol=1e-6   # after τ=20 the excited state components are suppressed by e^{−(Ek−E0)τ} ≈ e^{−Δ×20} ≈ 0; physics: imaginary-time evolution projects onto the GS
    end

    @testset "RealTime ED matches fine-dt TEBD on L=4 Heisenberg" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        mpo = MPO(H)   # MPO for energy measurement

        # Both start from the same state
        psi_vec = normalize(randn(MersenneTwister(7), ComplexF64, 2^4))   # `MersenneTwister(7)` is a seeded RNG ; using a fixed seed ensures reproducibility
        ψ_mps = to_mps(as_state(psi_vec, [2,2,2,2]); trunc=NoTrunc(), form=:left)

        t = 0.5  # total time
        dt_tebd = 0.01  # fine enough for Trotter error to be small

        # ED propagation
        p_ed = ConstantProtocol(RealTime(), t, 1, H)  # single step by exact matrix exp  # ED: one big step of size t=0.5 using exact matrix exponential exp(−iHt)
        result_ed = solve(H, as_statevector(psi_vec), ExactDiagonalization(:time), p_ed)
        E_ed = real(result_ed.state' * matrix_repr(H) * result_ed.state) / real(result_ed.state' * result_ed.state)   # Rayleigh quotient: ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ (divide by norm² because ED result may not be unit-normalised)

        # TEBD (fine dt, 2nd-order Trotter)
        p_tebd = ConstantProtocol(RealTime(), dt_tebd, round(Int, t/dt_tebd), H)   # 50 steps of size dt=0.01; `round(Int, ...)` = Python `int(round(...))`
        result_tebd = solve(H, Evolution(ψ_mps), TEBD(SuzukiTrotter(2), NoTrunc()), p_tebd)   # TEBD with 2nd-order Trotter; `Evolution(ψ_mps)` is the DynamicsStudy type; `SuzukiTrotter(2)` = Strang splitting
        ψ_f = result_tebd.state
        E_tebd = real(expect(ψ_f, mpo)) / real(overlap(ψ_f, ψ_f))   # TEBD energy via MPO

        # Energies should match (energy is conserved, so both give E_init)
        E_init = real(psi_vec' * matrix_repr(H) * psi_vec)   # initial energy (before any evolution)
        @test E_ed ≈ E_init  atol=1e-8   # ED exactly conserves energy (no Trotter error)
        @test E_tebd ≈ E_init  atol=1e-4  # TEBD has Trotter error  # TEBD approximately conserves energy; error is O(dt²) per step for 2nd-order Trotter = O(dt²×T) total → ~1e-4 for dt=0.01, T=0.5
    end

end
