# Tests for §7.1–7.2 — gate exponentiation, TEBD time evolution.
# Physics invariants: unitarity, norm conservation (real time), PSD (imaginary time),
# Trotter error scaling, and energy conservation.

@testset "§7.1 TimeAxis + Gate exponentiation" begin
    @testset "RealTime gate is unitary" begin
        ops = algebra_generators(SpinHalf())   # get the spin-1/2 operator matrices as a NamedTuple; `algebra_generators` returns (; Sx, Sy, Sz, Sp, Sm, I)
        # SzSz bond Hamiltonian (2-site, d²=4)
        h = kron(Array(ops.Sz), Array(ops.I)) + kron(Array(ops.I), Array(ops.Sz))   # build a 4×4 two-site Hamiltonian; `kron(A, B)` = Kronecker product
        G = gate(h, 0.1, RealTime())   # exponentiate: G = exp(-i·0.1·h); `RealTime()` is the time axis type that selects the `exp(-i·Δt·h)` formula
        @test G isa Propagator{RealTime}   # `isa` check: G must be a `Propagator{RealTime}` (not `{ImaginaryTime}`); the type parameter encodes which formula was used
        @test G.data * G.data' ≈ I(4) atol=1e-10   # unitarity: U·U† ≈ I; `I(4)` is the 4×4 identity (from LinearAlgebra); `*` is matrix multiplication; `≈` with atol checks floating-point closeness
        @test G.data' * G.data ≈ I(4) atol=1e-10   # also check U†·U ≈ I (both left and right unitarity)
    end

    @testset "ImaginaryTime gate is Hermitian and positive semidefinite" begin
        ops = algebra_generators(SpinHalf())
        h = kron(Array(ops.Sz), Array(ops.Sz))  # 4×4, Hermitian  # Ising-type two-site term Sz⊗Sz; this is a real symmetric matrix (purely diagonal in the Sz basis)
        G = gate(h, 0.1, ImaginaryTime())   # imaginary-time gate: G = exp(-Δτ·h) = e^{-0.1·h}; no `im` factor — this is a real positive semi-definite matrix for positive Δτ
        @test G isa Propagator{ImaginaryTime}   # type check: imaginary-time gates get the `ImaginaryTime` type tag
        @test G.data ≈ G.data' atol=1e-10   # Hermitian: exp(-τ·h) is Hermitian when h is Hermitian (since exp of Hermitian is Hermitian)
        ev = eigvals(Hermitian(G.data))   # eigenvalues of the gate; `Hermitian(M)` wraps M to tell Julia it's Hermitian (faster eigensolver); `eigvals` returns sorted real eigenvalues
        @test all(ev .≥ -1e-10)                # PSD  # all eigenvalues ≥ 0 (positive semi-definite); `.≥` broadcasts element-wise ≥  for positive τ has non-negative eigenvalues
    end

    @testset "opclass dispatches on time axis type" begin
        ops = algebra_generators(SpinHalf())
        h = kron(Array(ops.Sz), Array(ops.Sz))
        G_real = gate(h, 0.05, RealTime())   # real-time gate is unitary
        G_imag = gate(h, 0.05, ImaginaryTime())   # imaginary-time gate is Hermitian PSD
        @test opclass(G_real) isa Unitary   # `opclass` dispatches on the type parameter and returns a type tag; `Unitary` is the tag for unitary gates
        @test opclass(G_imag) isa HermitianPSD   # `HermitianPSD` is the tag for Hermitian positive semi-definite gates; physics: tells downstream whether to renormalize after applying
    end

    @testset "gate dt stored in Propagator" begin
        ops = algebra_generators(SpinHalf())
        h = kron(Array(ops.Sz), Array(ops.Sz))
        dt = 0.137   # arbitrary time step
        G = gate(h, dt, RealTime())   # gate carries dt as a field (useful for reconstructing total time)
        @test G.dt ≈ dt   # check that the stored dt matches the input; `≈` is used instead of `==` because floating-point is involved
    end

    @testset "gate(h, dt, RealTime) matches matrix exponential exp(-im*dt*h)" begin
        ops = algebra_generators(SpinHalf())
        h4 =
            kron(Array(ops.Sz), Array(ops.Sz)) +
            0.5 * (kron(Array(ops.Sp), Array(ops.Sm)) + kron(Array(ops.Sm), Array(ops.Sp)))   # Heisenberg bond Hamiltonian h = Sz⊗Sz + ½(S⁺⊗S⁻ + S⁻⊗S⁺); `*` is scalar×matrix multiplication
        dt = 0.2
        G = gate(h4, dt, RealTime())   # our implementation
        G_ref = exp(-im * dt * h4)   # `exp(A)` is the matrix exponential ; this is the reference "exact" calculation
        @test G.data ≈ G_ref atol=1e-10   # our gate matches the direct matrix exponential
    end

    @testset "gate(h, dt, ImaginaryTime) matches exp(-dt*h)" begin
        ops = algebra_generators(SpinHalf())
        h4 = kron(Array(ops.Sz), Array(ops.Sz))
        dt = 0.3
        G = gate(h4, dt, ImaginaryTime())   # imaginary-time gate
        G_ref = exp(-dt * h4)   # reference: direct matrix exponential (no `im` factor for imaginary time)
        @test G.data ≈ G_ref atol=1e-10
    end

    @testset "ConstantProtocol stores fields" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        p = ConstantProtocol(RealTime(), 0.01, 100, H)   # protocol with dt=0.01, nsteps=100, axis=RealTime
        @test p.dt ≈ 0.01   # check that dt is stored correctly
        @test p.nsteps == 100   # `==` is exact equality (for integers); nsteps is an Int
        @test total_time(p) ≈ 1.0   # `total_time(p) = p.dt * p.nsteps = 0.01 × 100 = 1.0`; total evolution time
    end

    @testset "gate from ConstantProtocol carries axis type" begin
        g = Chain(2)
        H = Heisenberg(g; J=1.0)
        ops = algebra_generators(SpinHalf())
        h4 =
            kron(Array(ops.Sp), Array(ops.Sm)) * 0.5 +   # bond Hamiltonian on 2 sites; built line-by-line for readability
            kron(Array(ops.Sm), Array(ops.Sp)) * 0.5 +
            kron(Array(ops.Sz), Array(ops.Sz))
        p_real = ConstantProtocol(RealTime(), 0.05, 10, H)   # real-time protocol
        p_imag = ConstantProtocol(ImaginaryTime(), 0.05, 10, H)   # imaginary-time protocol
        G_r = gate(h4, p_real)   # `gate(h, protocol)` dispatches on protocol type to extract dt and axis
        G_i = gate(h4, p_imag)
        @test G_r isa Propagator{RealTime}   # the returned gate inherits the time axis from the protocol
        @test G_i isa Propagator{ImaginaryTime}
    end
end

@testset "§7.1 bond_hamiltonian extraction" begin
    @testset "bond_hamiltonian has correct size" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        h_bond = bond_hamiltonian(H, 1)   # bond (1,2)  # extract the 2-site Hamiltonian for bond 1, i.e. h_{12}; shape d²×d² = 4×4 for spin-1/2
        @test size(h_bond) == (4, 4)      # d=2 → d²=4  # `size(M)` returns a tuple (dim1, dim2, ...); `(4, 4)` is a 2-tuple; physics: two-site Hilbert space has dim d² = 4
    end

    @testset "bond_hamiltonian is Hermitian" begin
        g = Chain(4)
        H = XXZ(g; J=1.0, Jz=0.5, h=0.1)
        for b in 1:3   # iterate over all 3 bonds of a 4-site chain; `1:3` is the range [1,2,3]
            h_b = bond_hamiltonian(H, b)   # bond Hamiltonian at bond b
            @test h_b ≈ h_b' atol=1e-12   # `h_b'` = conjugate transpose; Hermitian check; physics: bond terms must be Hermitian for energy to be real
        end
    end

    @testset "sum of bond_hamiltonians matches dense_matrix (OBC)" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0, h=0.0)
        H_dense = matrix_repr(H)   # 8×8 dense matrix
        # For OBC Heisenberg, H = h_{12} ⊗ I_3 + I_1 ⊗ h_{23}
        d = 2;
        L = 3;
        d2 = d^2;
        D = d^L   # local dimension, chain length, bond space dim, Hilbert space dim
        h12 = bond_hamiltonian(H, 1)   # 4×4 bond Hamiltonian for bond (1,2)
        h23 = bond_hamiltonian(H, 2)   # 4×4 bond Hamiltonian for bond (2,3)
        H_recon = kron(h12, I(d)) + kron(I(d), h23)   # embed in full Hilbert space: h₁₂⊗I₃ + I₁⊗h₂₃; `I(d)` = d×d identity; kron products produce the correct 8×8 embedding
        @test H_recon ≈ H_dense atol=1e-10   # reconstructed Hamiltonian must match the directly built matrix
    end
end

@testset "§7.1 apply_gate! — two-site update" begin
    @testset "apply_gate! with identity gate leaves MPS unchanged" begin
        g = Chain(4)
        psi_vec = normalize(randn(ComplexF64, 2^4))   # random unit vector in 2^4=16 dimensional Hilbert space
        ψ = to_mps(as_state(psi_vec, [2, 2, 2, 2]); trunc=NoTrunc(), form=:left)   # convert to left-canonical MPS
        Id4 = Matrix{ComplexF64}(I, 4, 4)   # 4×4 identity matrix 
        G = Propagator(Id4, RealTime(), 0.0)   # identity propagator; `Propagator(data, axis, dt)` constructor
        ψ_new = apply_gate(ψ, G, 1; trunc=NoTrunc())   # apply at bond (1,2)  # apply identity gate at bond 1; should leave the state unchanged
        @test abs(overlap(ψ, ψ_new)) ≈ 1.0 atol=1e-8   # `abs(⟨ψ|ψ_new⟩)` = 1 means the two states are identical (up to global phase); physics: identity gate shouldn't change the state
    end

    @testset "apply_gate! preserves norm for RealTime gate" begin
        g = Chain(4)
        psi_vec = normalize(randn(ComplexF64, 2^4))
        ψ = to_mps(as_state(psi_vec, [2, 2, 2, 2]); trunc=NoTrunc(), form=:left)
        ops = algebra_generators(SpinHalf())
        h4 = kron(Array(ops.Sz), Array(ops.Sz))
        G = gate(h4, 0.1, RealTime())   # unitary gate U = exp(-i·0.1·h)
        ψ_new = apply_gate(ψ, G, 2; trunc=NoTrunc())   # apply at bond (2,3)
        norm_sq_before = real(overlap(ψ, ψ))   # ‖ψ‖² = ⟨ψ|ψ⟩; `real(...)` discards the tiny imaginary rounding error
        norm_sq_after = real(overlap(ψ_new, ψ_new))   # ‖ψ_new‖²
        @test norm_sq_after ≈ norm_sq_before atol=1e-8   # unitary gates preserve norm: ‖U|ψ⟩‖² = ‖|ψ⟩‖²
    end

    @testset "apply_gate! then inverse gate returns original state" begin
        g = Chain(4)
        psi_vec = normalize(randn(ComplexF64, 2^4))
        ψ = to_mps(as_state(psi_vec, [2, 2, 2, 2]); trunc=NoTrunc(), form=:left)
        ops = algebra_generators(SpinHalf())
        h4 = kron(Array(ops.Sz), Array(ops.Sz))
        G_fwd = gate(h4, 0.1, RealTime())   # forward gate: exp(-i·0.1·h)
        G_bwd = gate(h4, -0.1, RealTime())   # backward gate: exp(+i·0.1·h) = inverse of G_fwd
        ψ_fwd = apply_gate(ψ, G_fwd, 1; trunc=NoTrunc())   # apply forward
        ψ_back = apply_gate(ψ_fwd, G_bwd, 1; trunc=NoTrunc())   # apply backward (inverse)
        @test abs(overlap(ψ, ψ_back)) ≈ real(overlap(ψ, ψ)) * real(overlap(ψ_back, ψ_back)) atol=1e-8   # Cauchy-Schwarz equality: |⟨ψ|ψ_back⟩|² = ‖ψ‖²·‖ψ_back‖² iff the states are parallel (same direction up to global phase)
    end
end

@testset "§7.2 SuzukiTrotter decomposition" begin
    @testset "SuzukiTrotter(1) has correct number of substeps" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        formula = SuzukiTrotter(1)   # first-order Trotter: one pass through all bonds
        steps = trotter_steps(formula, H, 0.1)   # generate the list of TrotterSubstep objects for one full Trotter step
        # 1st order: one pass through all bonds
        @test length(steps) == length(bonds(g))   # `bonds(g)` returns all NN bonds; `length(...)` counts them; physics: 1st-order Trotter = one sweep through all bonds
    end

    @testset "SuzukiTrotter(2) is symmetric (palindrome) in dt" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        formula = SuzukiTrotter(2)   # 2nd-order (Strang splitting): even-odd-even or half-steps-forward-half-steps-back
        steps = trotter_steps(formula, H, 0.1)
        dts = [s.dt for s in steps]   # extract the dt from each substep; `[f(x) for x in iter]` = Python list comprehension
        bonds_list = [s.bond for s in steps]   # extract the bond index from each substep
        # Must be palindrome in both bond order and dt
        @test dts ≈ reverse(dts) atol=1e-12   # `reverse(v)` = Python `v[::-1]`; the time steps must be a palindrome for 2nd-order symmetry
        @test bonds_list == reverse(bonds_list)   # bond order must also be a palindrome (symmetric about the midpoint)
    end

    @testset "trotter_step! conserves energy for short times" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        mpo = MPO(H)   # build MPO for energy measurement
        psi_vec = normalize(randn(ComplexF64, 2^4))
        ψ = to_mps(as_state(psi_vec, [2, 2, 2, 2]); trunc=NoTrunc(), form=:left)
        ψ = canonicalize(ψ, BondCanonical(2, NoTrunc()))   # put in bond-canonical form for stable measurement
        E_init = real(expect(ψ, mpo)) / real(overlap(ψ, ψ))   # Rayleigh quotient: E = ⟨H⟩/⟨ψ|ψ⟩; divide by norm² because ψ may not be unit-normalised after canonicalization

        # Very small time step: energy should barely change
        formula = SuzukiTrotter(2)   # use 2nd-order for better accuracy
        ψ_new = trotter_step(ψ, H, 0.001, formula; trunc=MaxBondDimTrunc(16))   # one Trotter step with very small dt=0.001
        ψ_new = canonicalize(ψ_new, BondCanonical(2, NoTrunc()))   # re-canonicalize for measurement
        E_new = real(expect(ψ_new, mpo)) / real(overlap(ψ_new, ψ_new))
        @test abs(E_new - E_init) < 1e-4   # energy change should be tiny for small dt; physics: real-time unitary evolution conserves energy exactly; the small error comes from Trotter discretization
    end

    @testset "real-time trotter_step! preserves norm" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        psi_vec = normalize(randn(ComplexF64, 2^4))
        ψ = to_mps(as_state(psi_vec, [2, 2, 2, 2]); trunc=NoTrunc(), form=:left)
        norm_before = real(overlap(ψ, ψ))   # ‖ψ‖² before evolution
        formula = SuzukiTrotter(1)
        ψ_new = trotter_step(ψ, H, 0.05, formula; trunc=NoTrunc())   # `NoTrunc()` = keep all singular values exactly
        norm_after = real(overlap(ψ_new, ψ_new))   # ‖ψ_new‖² after evolution
        @test norm_after ≈ norm_before atol=1e-8   # unitary gates preserve norm; small deviation due to floating-point arithmetic only
    end
end
