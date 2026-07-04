# Tests for §8 — TEBD quench driver, Tracker, neel_state.
# Physics invariants: Sz conservation, energy conservation, entropy growth.

@testset "§8.1 neel_state" begin

    @testset "neel_state has correct length" begin
        g = Chain(4)
        ψ = neel_state(g)
        @test length(ψ.tensors) == 4
    end

    @testset "neel_state is a product state (bond dim 1)" begin
        g = Chain(6)
        ψ = neel_state(g)
        for i in 1:6
            @test size(ψ.tensors[i].data, 1) == 1 || size(ψ.tensors[i].data, 3) == 1 ||
                  (i > 1 && size(ψ.tensors[i].data, 1) == 1) ||
                  (i < 6 && size(ψ.tensors[i].data, 3) == 1)
        end
        # Bond dimensions between sites should all be 1 (product state)
        for i in 1:5
            @test size(ψ.tensors[i].data, 3) == 1
        end
    end

    @testset "neel_state alternates |↑⟩ |↓⟩" begin
        g = Chain(4)
        ψ = neel_state(g)
        # local_expectation computes <ψ|σᵢ|ψ> directly without building a full MPO
        ops = algebra_generators(SpinHalf())
        sz1 = real(local_expectation(ψ, ComplexF64.(ops.Sz), 1))
        sz2 = real(local_expectation(ψ, ComplexF64.(ops.Sz), 2))
        @test sz1 ≈  0.5  atol=1e-10   # site 1: spin up
        @test sz2 ≈ -0.5  atol=1e-10   # site 2: spin down
    end

    @testset "neel_state has zero total Sz for even L" begin
        g = Chain(6)
        ψ = neel_state(g)
        Mop = total_magnetization(g)
        mpo_m = MPO(Mop)
        sz_tot = real(expect(ψ, mpo_m)) / real(overlap(ψ, ψ))
        @test sz_tot ≈ 0.0  atol=1e-10
    end

    @testset "neel_state entanglement entropy is zero (product state)" begin
        g = Chain(4)
        ψ = neel_state(g)
        # Product state → Schmidt rank 1 at every bond → S=0
        ψ_c = canonicalize(ψ, BondCanonical(2, NoTrunc()))
        sv  = ψ_c.bond_svs[3].values   # singular values at bond between sites 2 and 3
        λ²  = (sv ./ norm(sv)) .^ 2
        S   = -sum(p -> p > 0 ? p * log2(p) : 0.0, λ²)
        @test S ≈ 0.0  atol=1e-10
    end

end

@testset "§8.1 TEBD quench via solve" begin

    @testset "total Sz conserved under XXZ real-time evolution" begin
        g = Chain(4)
        H = XXZ(g; J=1.0, Jz=1.0, h=0.0)
        Mop = total_magnetization(g)
        mpo_m = MPO(Mop)
        mpo_h = MPO(H)

        ψ₀ = neel_state(g)
        # Canonicalize the Neel state
        ψ₀ = canonicalize(ψ₀, LeftCanonical(NoTrunc()))

        Sz_init = real(expect(ψ₀, mpo_m)) / real(overlap(ψ₀, ψ₀))

        p = ConstantProtocol(RealTime(), 0.02, 20, H)
        result = solve(H, Quench(ψ₀), TEBD(SuzukiTrotter(2), MaxBondDimTrunc(16)), p)

        ψ_final = result.state
        Sz_final = real(expect(ψ_final, mpo_m)) / real(overlap(ψ_final, ψ_final))
        @test abs(Sz_final - Sz_init) < 1e-3
    end

    @testset "norm preserved under real-time evolution" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        ψ₀ = canonicalize(neel_state(g), LeftCanonical(NoTrunc()))
        norm_init = real(overlap(ψ₀, ψ₀))

        p = ConstantProtocol(RealTime(), 0.05, 5, H)
        result = solve(H, Quench(ψ₀), TEBD(SuzukiTrotter(1), MaxBondDimTrunc(8)), p)

        norm_final = real(overlap(result.state, result.state))
        @test norm_final ≈ norm_init  atol=1e-6
    end

    @testset "TEBD QuenchResult has correct fields" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        ψ₀ = canonicalize(neel_state(g), LeftCanonical(NoTrunc()))
        p = ConstantProtocol(RealTime(), 0.01, 2, H)
        result = solve(H, Quench(ψ₀), TEBD(SuzukiTrotter(1), NoTrunc()), p)
        @test result isa QuenchResult
        @test result.state isa FiniteMPS
        @test result.steps == 2
    end

end

@testset "§8.2 Tracker" begin

    @testset "NoTracker does not collect data" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        ψ₀ = canonicalize(neel_state(g), LeftCanonical(NoTrunc()))
        p = ConstantProtocol(RealTime(), 0.05, 3, H)
        result = solve(H, Quench(ψ₀), TEBD(SuzukiTrotter(1), NoTrunc()), p;
                       tracker=NoTracker())
        @test isempty(result.measurements)
    end

    @testset "Tracker collects magnetization each step" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        ψ₀ = canonicalize(neel_state(g), LeftCanonical(NoTrunc()))
        p = ConstantProtocol(RealTime(), 0.05, 4, H)
        Mop = total_magnetization(g)
        tracker = Tracker(:mag => Mop; every=1)
        result = solve(H, Quench(ψ₀), TEBD(SuzukiTrotter(1), MaxBondDimTrunc(8)), p;
                       tracker=tracker)
        @test haskey(result.measurements, :mag)
        @test length(result.measurements[:mag]) == 4   # one per step
    end

    @testset "Tracker :mag agrees with direct expect" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        ψ₀ = canonicalize(neel_state(g), LeftCanonical(NoTrunc()))
        p = ConstantProtocol(RealTime(), 0.01, 2, H)
        Mop = total_magnetization(g)
        tracker = Tracker(:mag => Mop; every=1)
        result = solve(H, Quench(ψ₀), TEBD(SuzukiTrotter(1), NoTrunc()), p;
                       tracker=tracker)

        # The magnetization of the Neel state should be ~0 and well-measured
        ψ_final = result.state
        mpo_m = MPO(Mop)
        sz_direct = real(expect(ψ_final, mpo_m)) / real(overlap(ψ_final, ψ_final))
        tracked_last = result.measurements[:mag][end]
        @test tracked_last ≈ sz_direct  atol=1e-8
    end

    @testset "measure_entropy grows from zero after quench" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        ψ₀ = neel_state(g)
        ψ₀ = canonicalize(ψ₀, BondCanonical(2, NoTrunc()))

        sv0 = ψ₀.bond_svs[3].values
        λ0  = (sv0 ./ norm(sv0)) .^ 2
        S_init = -sum(p -> p > 0 ? p * log2(p) : 0.0, λ0)
        @test S_init ≈ 0.0  atol=1e-10

        # After several steps entanglement should grow
        p = ConstantProtocol(RealTime(), 0.05, 20, H)
        result = solve(H, Quench(ψ₀), TEBD(SuzukiTrotter(2), MaxBondDimTrunc(32)), p)
        ψ_f = canonicalize(result.state, BondCanonical(2, NoTrunc()))
        svf = ψ_f.bond_svs[3].values
        λf  = (svf ./ norm(svf)) .^ 2
        S_final = -sum(p -> p > 0 ? p * log2(p) : 0.0, λf)
        @test S_final > 0.01   # entanglement has grown
    end

end
