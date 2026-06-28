# Tests for §7.1–7.2 — gate exponentiation, TEBD time evolution.
# Physics invariants: unitarity, norm conservation (real time), PSD (imaginary time),
# Trotter error scaling, and energy conservation.

@testset "§7.1 TimeAxis + Gate exponentiation" begin

    @testset "RealTime gate is unitary" begin
        ops = algebra_generators(SpinHalf())
        # SzSz bond Hamiltonian (2-site, d²=4)
        h = kron(Array(ops.Sz), Array(ops.I)) + kron(Array(ops.I), Array(ops.Sz))
        G = gate(h, 0.1, RealTime())
        @test G isa Propagator{RealTime}
        @test G.data * G.data' ≈ I(4)  atol=1e-10
        @test G.data' * G.data ≈ I(4)  atol=1e-10
    end

    @testset "ImaginaryTime gate is Hermitian and positive semidefinite" begin
        ops = algebra_generators(SpinHalf())
        h = kron(Array(ops.Sz), Array(ops.Sz))  # 4×4, Hermitian
        G = gate(h, 0.1, ImaginaryTime())
        @test G isa Propagator{ImaginaryTime}
        @test G.data ≈ G.data'  atol=1e-10   # Hermitian
        ev = eigvals(Hermitian(G.data))
        @test all(ev .≥ -1e-10)                # PSD
    end

    @testset "opclass dispatches on time axis type" begin
        ops = algebra_generators(SpinHalf())
        h = kron(Array(ops.Sz), Array(ops.Sz))
        G_real = gate(h, 0.05, RealTime())
        G_imag = gate(h, 0.05, ImaginaryTime())
        @test opclass(G_real) isa Unitary
        @test opclass(G_imag) isa HermitianPSD
    end

    @testset "gate dt stored in Propagator" begin
        ops = algebra_generators(SpinHalf())
        h = kron(Array(ops.Sz), Array(ops.Sz))
        dt = 0.137
        G = gate(h, dt, RealTime())
        @test G.dt ≈ dt
    end

    @testset "gate(h, dt, RealTime) matches matrix exponential exp(-im*dt*h)" begin
        ops = algebra_generators(SpinHalf())
        h4 = kron(Array(ops.Sz), Array(ops.Sz)) + 0.5 * (kron(Array(ops.Sp), Array(ops.Sm)) + kron(Array(ops.Sm), Array(ops.Sp)))
        dt = 0.2
        G = gate(h4, dt, RealTime())
        G_ref = exp(-im * dt * h4)
        @test G.data ≈ G_ref  atol=1e-10
    end

    @testset "gate(h, dt, ImaginaryTime) matches exp(-dt*h)" begin
        ops = algebra_generators(SpinHalf())
        h4 = kron(Array(ops.Sz), Array(ops.Sz))
        dt = 0.3
        G = gate(h4, dt, ImaginaryTime())
        G_ref = exp(-dt * h4)
        @test G.data ≈ G_ref  atol=1e-10
    end

    @testset "ConstantProtocol stores fields" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        p = ConstantProtocol(RealTime(), 0.01, 100, H)
        @test p.dt ≈ 0.01
        @test p.nsteps == 100
        @test total_time(p) ≈ 1.0
    end

    @testset "gate from ConstantProtocol carries axis type" begin
        g = Chain(2)
        H = Heisenberg(g; J=1.0)
        ops = algebra_generators(SpinHalf())
        h4 = kron(Array(ops.Sp), Array(ops.Sm)) * 0.5 +
             kron(Array(ops.Sm), Array(ops.Sp)) * 0.5 +
             kron(Array(ops.Sz), Array(ops.Sz))
        p_real = ConstantProtocol(RealTime(), 0.05, 10, H)
        p_imag = ConstantProtocol(ImaginaryTime(), 0.05, 10, H)
        G_r = gate(h4, p_real)
        G_i = gate(h4, p_imag)
        @test G_r isa Propagator{RealTime}
        @test G_i isa Propagator{ImaginaryTime}
    end

end

@testset "§7.1 bond_hamiltonian extraction" begin

    @testset "bond_hamiltonian has correct size" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        h_bond = bond_hamiltonian(H, 1)   # bond (1,2)
        @test size(h_bond) == (4, 4)      # d=2 → d²=4
    end

    @testset "bond_hamiltonian is Hermitian" begin
        g = Chain(4)
        H = XXZ(g; J=1.0, Jz=0.5, h=0.1)
        for b in 1:3
            h_b = bond_hamiltonian(H, b)
            @test h_b ≈ h_b'  atol=1e-12
        end
    end

    @testset "sum of bond_hamiltonians matches dense_matrix (OBC)" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0, h=0.0)
        H_dense = dense_matrix(H)
        # For OBC Heisenberg, H = h_{12} ⊗ I_3 + I_1 ⊗ h_{23}
        d = 2; L = 3; d2 = d^2; D = d^L
        h12 = bond_hamiltonian(H, 1)
        h23 = bond_hamiltonian(H, 2)
        H_recon = kron(h12, I(d)) + kron(I(d), h23)
        @test H_recon ≈ H_dense  atol=1e-10
    end

end

@testset "§7.1 apply_gate! — two-site update" begin

    @testset "apply_gate! with identity gate leaves MPS unchanged" begin
        g = Chain(4)
        psi_vec = normalize(randn(ComplexF64, 2^4))
        ψ = to_mps(as_state(psi_vec, [2,2,2,2]); trunc=NoTrunc(), form=:left)
        Id4 = Matrix{ComplexF64}(I, 4, 4)
        G = Propagator(Id4, RealTime(), 0.0)
        ψ_new = apply_gate(ψ, G, 1; trunc=NoTrunc())   # apply at bond (1,2)
        @test abs(overlap(ψ, ψ_new)) ≈ 1.0  atol=1e-8
    end

    @testset "apply_gate! preserves norm for RealTime gate" begin
        g = Chain(4)
        psi_vec = normalize(randn(ComplexF64, 2^4))
        ψ = to_mps(as_state(psi_vec, [2,2,2,2]); trunc=NoTrunc(), form=:left)
        ops = algebra_generators(SpinHalf())
        h4 = kron(Array(ops.Sz), Array(ops.Sz))
        G = gate(h4, 0.1, RealTime())
        ψ_new = apply_gate(ψ, G, 2; trunc=NoTrunc())
        norm_sq_before = real(overlap(ψ, ψ))
        norm_sq_after  = real(overlap(ψ_new, ψ_new))
        @test norm_sq_after ≈ norm_sq_before  atol=1e-8
    end

    @testset "apply_gate! then inverse gate returns original state" begin
        g = Chain(4)
        psi_vec = normalize(randn(ComplexF64, 2^4))
        ψ = to_mps(as_state(psi_vec, [2,2,2,2]); trunc=NoTrunc(), form=:left)
        ops = algebra_generators(SpinHalf())
        h4 = kron(Array(ops.Sz), Array(ops.Sz))
        G_fwd = gate(h4, 0.1, RealTime())
        G_bwd = gate(h4, -0.1, RealTime())
        ψ_fwd = apply_gate(ψ, G_fwd, 1; trunc=NoTrunc())
        ψ_back = apply_gate(ψ_fwd, G_bwd, 1; trunc=NoTrunc())
        @test abs(overlap(ψ, ψ_back)) ≈ real(overlap(ψ, ψ)) * real(overlap(ψ_back, ψ_back))  atol=1e-8
    end

end

@testset "§7.2 SuzukiTrotter decomposition" begin

    @testset "SuzukiTrotter(1) has correct number of substeps" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        formula = SuzukiTrotter(1)
        steps = trotter_steps(formula, H, 0.1)
        # 1st order: one pass through all bonds
        @test length(steps) == length(bonds(g))
    end

    @testset "SuzukiTrotter(2) is symmetric (palindrome) in dt" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        formula = SuzukiTrotter(2)
        steps = trotter_steps(formula, H, 0.1)
        dts = [s.dt for s in steps]
        bonds_list = [s.bond for s in steps]
        # Must be palindrome in both bond order and dt
        @test dts ≈ reverse(dts)  atol=1e-12
        @test bonds_list == reverse(bonds_list)
    end

    @testset "trotter_step! conserves energy for short times" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        mpo = MPO(H)
        psi_vec = normalize(randn(ComplexF64, 2^4))
        ψ = to_mps(as_state(psi_vec, [2,2,2,2]); trunc=NoTrunc(), form=:left)
        ψ = canonicalize(ψ, BondCanonical(2, NoTrunc()))
        E_init = real(expect(ψ, mpo)) / real(overlap(ψ, ψ))

        # Very small time step: energy should barely change
        formula = SuzukiTrotter(2)
        ψ_new = trotter_step(ψ, H, 0.001, formula; trunc=MaxBondDimTrunc(16))
        ψ_new = canonicalize(ψ_new, BondCanonical(2, NoTrunc()))
        E_new = real(expect(ψ_new, mpo)) / real(overlap(ψ_new, ψ_new))
        @test abs(E_new - E_init) < 1e-4
    end

    @testset "real-time trotter_step! preserves norm" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        psi_vec = normalize(randn(ComplexF64, 2^4))
        ψ = to_mps(as_state(psi_vec, [2,2,2,2]); trunc=NoTrunc(), form=:left)
        norm_before = real(overlap(ψ, ψ))
        formula = SuzukiTrotter(1)
        ψ_new = trotter_step(ψ, H, 0.05, formula; trunc=NoTrunc())
        norm_after = real(overlap(ψ_new, ψ_new))
        @test norm_after ≈ norm_before  atol=1e-8
    end

end
