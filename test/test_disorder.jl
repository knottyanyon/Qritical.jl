# Tests for §9 — disorder realization and imaginary-time GS via TEBD.

using Random

@testset "§9.1 disorder_realization" begin

    @testset "disorder_realization returns correct length" begin
        h = disorder_realization(6, Uniform(-1.0, 1.0), MersenneTwister(42))
        @test length(h) == 6
    end

    @testset "disorder_realization is reproducible with same seed" begin
        rng1 = MersenneTwister(123)
        rng2 = MersenneTwister(123)
        h1 = disorder_realization(8, Uniform(-2.0, 2.0), rng1)
        h2 = disorder_realization(8, Uniform(-2.0, 2.0), rng2)
        @test h1 == h2
    end

    @testset "Uniform distribution samples within bounds" begin
        h = disorder_realization(100, Uniform(-1.0, 1.0), MersenneTwister(7))
        @test all(-1.0 .≤ h .≤ 1.0)
    end

    @testset "XXZ accepts disorder field array" begin
        g = Chain(4)
        rng = MersenneTwister(1)
        h_vec = disorder_realization(4, Uniform(-2.0, 2.0), rng)
        H = XXZ(g; J=1.0, Jz=1.0, h=h_vec)
        @test length(H.onsite) == 4
        # onsite coupling at site i should equal -h_vec[i]
        for lt in H.onsite
            @test lt.coupling ≈ -h_vec[lt.site]  atol=1e-12
        end
    end

    @testset "disorder Hamiltonian is Hermitian" begin
        g = Chain(4)
        rng = MersenneTwister(99)
        h_vec = disorder_realization(4, Uniform(-3.0, 3.0), rng)
        H = XXZ(g; J=1.0, Jz=1.0, h=h_vec)
        M = dense_matrix(H)
        @test M ≈ M'  atol=1e-12
    end

end

@testset "§9.1 imaginary-time GS convergence" begin

    @testset "imaginary-time TEBD converges to ED GS energy" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        E_ed = minimum(real.(eigvals(Hermitian(dense_matrix(H)))))

        # Start from Neel state, evolve in imaginary time
        ψ₀ = canonicalize(neel_state(g), LeftCanonical(NoTrunc()))
        p = ConstantProtocol(ImaginaryTime(), 0.05, 120, H)
        result = solve(H, Quench(ψ₀), TEBD(SuzukiTrotter(2), MaxBondDimTrunc(16)), p)

        ψ_f = result.state
        mpo = MPO(H)
        E_tebd = real(expect(ψ_f, mpo)) / real(overlap(ψ_f, ψ_f))
        @test E_tebd ≈ E_ed  atol=5e-3
    end

    @testset "disordered model: imaginary-time lowers energy below random initial" begin
        g = Chain(4)
        rng = MersenneTwister(42)
        h_vec = disorder_realization(4, Uniform(-2.0, 2.0), rng)
        H = XXZ(g; J=1.0, Jz=1.0, h=h_vec)
        mpo = MPO(H)

        # Random initial state
        psi_vec = normalize(randn(MersenneTwister(1), ComplexF64, 2^4))
        ψ_rand = to_mps(as_state(psi_vec, [2,2,2,2]); trunc=NoTrunc(), form=:left)
        E_rand = real(expect(ψ_rand, mpo)) / real(overlap(ψ_rand, ψ_rand))

        # Imaginary time evolution
        ψ₀ = canonicalize(ψ_rand, LeftCanonical(NoTrunc()))
        p = ConstantProtocol(ImaginaryTime(), 0.05, 30, H)
        result = solve(H, Quench(ψ₀), TEBD(SuzukiTrotter(2), MaxBondDimTrunc(8)), p)

        ψ_f = result.state
        E_final = real(expect(ψ_f, mpo)) / real(overlap(ψ_f, ψ_f))
        @test E_final ≤ E_rand + 1e-6
    end

    @testset "⟨S₁ᶻSᵢᶻ⟩ of TEBD GS matches ED cross-check" begin
        g = Chain(4)
        rng = MersenneTwister(7)
        h_vec = disorder_realization(4, Uniform(-1.0, 1.0), rng)
        H = XXZ(g; J=1.0, Jz=1.0, h=h_vec)
        mpo = MPO(H)

        # Ground state via imaginary-time TEBD
        ψ₀ = canonicalize(neel_state(g), LeftCanonical(NoTrunc()))
        p = ConstantProtocol(ImaginaryTime(), 0.05, 120, H)
        result = solve(H, Quench(ψ₀), TEBD(SuzukiTrotter(2), MaxBondDimTrunc(16)), p)
        ψ_gs = result.state

        # Ground state via exact diagonalization
        ops = operators(SpinHalf())
        Sz = Array(ops.Sz)
        ed_result = solve(H, GroundState(), ExactDiagonalization(:ground))
        psi_ed = ed_result.state

        # Compare ⟨S₁ᶻ Sᵢᶻ⟩ for all i > 1
        for i in 2:4
            # TEBD: two_point(ψ, Sz, Sz, 1, i)
            C_tebd = real(two_point(ψ_gs, Sz, Sz, 1, i)) / real(overlap(ψ_gs, ψ_gs))

            # ED: direct from the state vector
            L = 4; d = 2
            Id = Matrix{Float64}(I, d, d)
            op1 = foldl(kron, [j == 1 ? Sz : (j == i ? Sz : Id) for j in 1:L])
            C_ed = real(psi_ed' * op1 * psi_ed)
            @test C_tebd ≈ C_ed  atol=1e-2
        end
    end

end
