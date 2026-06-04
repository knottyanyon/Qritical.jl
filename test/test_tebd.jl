# v0.5 tests: Vidal form and TEBD — physics behavior, algorithms, edge cases

using LinearAlgebra: norm, I, Diagonal, kron, exp

# ── helpers ───────────────────────────────────────────────────────────────────

# _neel_mps is defined in test_finite_mpo.jl (loaded before this file)

# TODO ── implement this function (≈ 6–8 lines)
#
# Return the 4×4 two-site Heisenberg Hamiltonian:
#
#   h_{i,i+1} = J (Sˣ ⊗ Sˣ + Sʸ ⊗ Sʸ + Sᶻ ⊗ Sᶻ)
#
# Basis convention: |↑⟩ = index 1, |↓⟩ = index 2  (same as heisenberg_mpo in src/).
# Use `kron` from LinearAlgebra for Kronecker products.
# Hint: Sˣ = (S⁺ + S⁻)/2,  Sʸ = -im*(S⁺ - S⁻)/2,  Sᶻ = diag(1/2, -1/2)
#       S⁺ = [0 1; 0 0],     S⁻ = [0 0; 1 0]
#
# The result must be a `Matrix{T}` of size (4, 4).
function _heisenberg_h2(; J::Real=1.0, T::Type=Float64)
    Sz = T[1//2  0; 0  -1//2]
    Sp = T[0  1; 0   0]
    Sm = T[0  0; 1   0]
    Sx = (Sp + Sm) / 2
    Sy = -im * (Sp - Sm) / 2
    return T(J) * real(kron(Sx, Sx) + kron(Sy, Sy) + kron(Sz, Sz))
end

# Uniform H_bonds for an L-site Heisenberg chain.
function _heisenberg_bonds(L::Int; J::Real=1.0, T::Type=Float64)
    return [_heisenberg_h2(; J, T) for _ in 1:L-1]
end

# ── CompositeDoF ──────────────────────────────────────────────────────────────

@testset "CompositeDoF — hilbert_space" begin
    @test hilbert_space(CompositeDoF{Spin{1//2}, Spin{1//2}}()) == 4
    @test hilbert_space(CompositeDoF{Spin{1//2}, Spin{1}  }()) == 6
end

# ── to_vidal — physics behavior ───────────────────────────────────────────────

@testset "to_vidal — form tag is VidalForm()" begin
    mps = FiniteMPS(Spin{1//2}(), 4, 4)
    left_canonical_sweep!(mps)
    @test to_vidal(mps).form == VidalForm()
end

@testset "to_vidal — Vidal normalization ‖Λ_{i-1} Γ_i Λ_i‖ = 1 at every site" begin
    mps   = FiniteMPS(Spin{1//2}(), 5, 4)
    left_canonical_sweep!(mps)
    vidal = to_vidal(mps)
    for i in 1:vidal.L
        λL = vidal.bond_svs[i]             # Λ_{i-1}, length χL
        Γ  = vidal.tensors[i].data         # (χL, d, χR)
        λR = vidal.bond_svs[i + 1]         # Λ_i,     length χR
        θ  = λL .* Γ .* reshape(λR, 1, 1, :)
        @test norm(θ) ≈ 1.0 atol=1e-12
    end
end

@testset "to_vidal / to_canonical — round-trip fidelity ≈ 1" begin
    mps   = FiniteMPS(Spin{1//2}(), 5, 4)
    left_canonical_sweep!(mps)
    vidal = to_vidal(mps)
    mps2  = to_canonical(vidal)
    @test abs2(overlap(mps, mps2)) ≈ 1.0 atol=1e-10
end

# ── apply_gate! — physics behavior ────────────────────────────────────────────

@testset "apply_gate! — identity gate leaves state and bond dims unchanged" begin
    mps   = _neel_mps(4)
    vidal = to_vidal(mps)
    χ_before = [size(vidal.tensors[i].data, 3) for i in 1:3]

    Id4 = Matrix{Float64}(I, 4, 4)
    for bond in 1:3
        apply_gate!(vidal, Id4, (bond, bond + 1))
    end

    mps_back = to_canonical(vidal)
    @test abs2(overlap(mps, mps_back)) ≈ 1.0 atol=1e-10
    @test [size(vidal.tensors[i].data, 3) for i in 1:3] == χ_before
end

@testset "apply_gate! — unitary gate preserves norm" begin
    mps   = FiniteMPS(Spin{1//2}(), 4, 4; T=ComplexF64)
    left_canonical_sweep!(mps)
    vidal = to_vidal(mps)

    h2   = _heisenberg_h2(; T=ComplexF64)
    gate = exp(-im * 0.1 * h2)
    apply_gate!(vidal, gate, (2, 3))

    mps_out = to_canonical(vidal)
    @test real(overlap(mps_out, mps_out)) ≈ 1.0 atol=1e-10
end

# ── trotter_step! — physics behavior ──────────────────────────────────────────

@testset "trotter_step! — energy deviation is O(dt²) (first-order Trotter)" begin
    L       = 6
    mpo     = heisenberg_mpo(L)
    H_bonds = _heisenberg_bonds(L; T=ComplexF64)
    E_neel  = expectation_value(_neel_mps(L), mpo)

    errs = Float64[]
    for dt in [0.1, 0.05, 0.025]
        vidal = to_vidal(_neel_mps(L; T=ComplexF64))
        trotter_step!(vidal, H_bonds, dt)
        E = real(expectation_value(to_canonical(vidal), mpo))
        push!(errs, abs(E - E_neel))
    end
    # halving dt → error drops by ≈ 4 (O(dt²) Trotter error)
    @test errs[2] / errs[1] < 0.4
    @test errs[3] / errs[2] < 0.4
end

# ── time_evolve — physics behavior ────────────────────────────────────────────

@testset "time_evolve — entropy grows at central bond of Néel state" begin
    L       = 6
    H_bonds = _heisenberg_bonds(L; T=ComplexF64)
    mps     = _neel_mps(L; T=ComplexF64)
    center  = L ÷ 2 + 1   # bond index L/2+1 sits at the chain centre

    @test entanglement_entropy(mps, center) ≈ 0.0 atol=1e-12   # product state

    mps_t = time_evolve(mps, H_bonds, 1.0, 0.05)
    @test entanglement_entropy(mps_t, center) > 0.1
end

@testset "time_evolve — imaginary time lowers energy below Néel-state energy" begin
    L       = 4
    mpo     = heisenberg_mpo(L)
    H_bonds = _heisenberg_bonds(L)
    E_neel  = expectation_value(_neel_mps(L), mpo)

    mps_gs = time_evolve(FiniteMPS(Spin{1//2}(), L, 8), H_bonds, 5.0, 0.05; imag=true)
    @test expectation_value(mps_gs, mpo) < E_neel
end

# ── edge cases ────────────────────────────────────────────────────────────────

@testset "apply_gate! — boundary bond (1,2): trivial left Λ = [1.0]" begin
    mps   = FiniteMPS(Spin{1//2}(), 4, 4; T=ComplexF64)
    left_canonical_sweep!(mps)
    vidal = to_vidal(mps)

    h2   = _heisenberg_h2(; T=ComplexF64)
    gate = exp(-im * 0.1 * h2)
    apply_gate!(vidal, gate, (1, 2))

    @test vidal.bond_svs[1] == [1.0]   # left boundary always trivial
end

@testset "apply_gate! — boundary bond (L-1,L): trivial right Λ = [1.0]" begin
    L     = 4
    mps   = FiniteMPS(Spin{1//2}(), L, 4; T=ComplexF64)
    left_canonical_sweep!(mps)
    vidal = to_vidal(mps)

    h2   = _heisenberg_h2(; T=ComplexF64)
    gate = exp(-im * 0.1 * h2)
    apply_gate!(vidal, gate, (L - 1, L))

    @test vidal.bond_svs[L + 1] == [1.0]   # right boundary always trivial
end

@testset "trotter_step! — dt → 0 limit acts as identity" begin
    mps   = _neel_mps(4; T=ComplexF64)
    vidal = to_vidal(mps)

    H_bonds = _heisenberg_bonds(4; T=ComplexF64)
    trotter_step!(vidal, H_bonds, 1e-8)

    mps_out = to_canonical(vidal)
    @test abs2(overlap(mps, mps_out)) ≈ 1.0 atol=1e-6
end

@testset "apply_gate! — bond dimension cap causes ε > 0" begin
    mps   = FiniteMPS(Spin{1//2}(), 6, 8; T=ComplexF64)
    left_canonical_sweep!(mps)
    vidal = to_vidal(mps)

    h2   = _heisenberg_h2(; T=ComplexF64)
    gate = exp(-im * 0.5 * h2)   # large dt → significant entanglement growth
    for bond in 1:5
        apply_gate!(vidal, gate, (bond, bond + 1); trunc=KeepFirst(2))
    end
    # After truncation the state is approximate: fidelity < 1
    mps_in  = FiniteMPS(Spin{1//2}(), 6, 8; T=ComplexF64)
    left_canonical_sweep!(mps_in)
    fidelity = abs2(overlap(mps_in, to_canonical(vidal)))
    @test fidelity < 1.0   # truncation error was non-zero
end

@testset "time_evolve — imaginary time: norm decays then re-normalizes" begin
    L       = 4
    H_bonds = _heisenberg_bonds(L)
    mps     = _neel_mps(L)

    mps_gs = time_evolve(mps, H_bonds, 1.0, 0.1; imag=true)
    # output should be normalized
    @test real(overlap(mps_gs, mps_gs)) ≈ 1.0 atol=1e-10
end
