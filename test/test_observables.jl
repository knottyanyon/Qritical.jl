# v0.6 tests: Observables — local_expectation, entanglement_spectrum, entropy vector

using LinearAlgebra: norm, I, kron, dot

# ── helpers ───────────────────────────────────────────────────────────────────

# Single-site operators for Spin{1//2} (basis: index 1 = |↑⟩, index 2 = |↓⟩)
const _Sz = Float64[1//2 0; 0 -1//2]
const _Sx = Float64[0 1//2; 1//2 0]

# Number operator for Fermionic (basis: index 1 = |0⟩, index 2 = |1⟩)
const _n_op = Float64[0 0; 0 1]

# All-spin-up product state |↑↑...↑⟩ of length L.
function _all_up_mps(L::Int; T::Type=Float64)
    d  = hilbert_space(Spin{1//2}())
    RT = real(T)
    tensors = Vector{IndexedTensor{T, 3}}(undef, L)
    for i in 1:L
        data = zeros(T, 1, d, 1)
        data[1, 1, 1] = one(T)   # always |↑⟩
        vL = upper(bond_label(:χ, i - 1), 1)
        σ  = lower(:σ, d)
        vR = lower(bond_label(:χ, i), 1)
        tensors[i] = IndexedTensor(data, (vL, σ, vR))
    end
    bond_svs = [RT[1] for _ in 1:L+1]
    return FiniteMPS{Spin{1//2}, T, RT}(L, tensors, bond_svs, CanonicalForm(L, L + 1))
end

# Alternating empty/occupied Fermionic product state |01010...⟩.
function _fermionic_neel_mps(L::Int)
    d  = hilbert_space(Fermionic())
    tensors = Vector{IndexedTensor{Float64, 3}}(undef, L)
    for i in 1:L
        data = zeros(1, d, 1)
        data[1, isodd(i) ? 1 : 2, 1] = 1.0   # even sites occupied
        vL = upper(bond_label(:χ, i - 1), 1)
        σ  = lower(:σ, d)
        vR = lower(bond_label(:χ, i), 1)
        tensors[i] = IndexedTensor(data, (vL, σ, vR))
    end
    bond_svs = [Float64[1] for _ in 1:L+1]
    return FiniteMPS{Fermionic, Float64, Float64}(L, tensors, bond_svs, CanonicalForm(L, L + 1))
end

# Maximally entangled two-site Bell state (|↑↓⟩ + |↓↑⟩)/√2.
# Left-canonical: A1 is left-orthogonal; Schmidt values are [1/√2, 1/√2] at bond 2.
function _bell_state_mps()
    d = 2
    # Site 1: shape (1, 2, 2)
    A1 = zeros(1, d, d)
    A1[1, 1, 1] = 1.0   # |↑⟩ → virtual 1
    A1[1, 2, 2] = 1.0   # |↓⟩ → virtual 2
    # Site 2: shape (2, 2, 1)  — absorbs Σ = [1/√2, 1/√2]
    A2 = zeros(d, d, 1)
    A2[1, 2, 1] = 1 / √2   # virtual 1 → |↓⟩
    A2[2, 1, 1] = 1 / √2   # virtual 2 → |↑⟩
    vL1 = upper(bond_label(:χ, 0), 1)
    vR1 = lower(bond_label(:χ, 1), d)
    vL2 = upper(bond_label(:χ, 1), d)
    vR2 = lower(bond_label(:χ, 2), 1)
    σ   = lower(:σ, d)
    tensors = [
        IndexedTensor(A1, (vL1, σ, vR1)),
        IndexedTensor(A2, (vL2, σ, vR2)),
    ]
    bond_svs = [Float64[1], [1/√2, 1/√2], Float64[1]]
    return FiniteMPS{Spin{1//2}, Float64, Float64}(2, tensors, bond_svs, CanonicalForm(2, 3))
end

# Brute-force expectation value via full state vector contraction (reference).
# Only practical for small L (≤ 6).
function _brute_force_expectation(mps::FiniteMPS, op::Matrix, site::Int)
    L = mps.L
    d = size(op, 1)
    # Build state vector by sequential tensor contraction
    A = mps.tensors[1].data   # (1, d, χ)
    psi = reshape(A, d, size(A, 3))   # (d, χ)
    for i in 2:L
        Ai    = mps.tensors[i].data   # (χL, d, χR)
        χL, _, χR = size(Ai)
        dᵢ    = size(psi, 1)
        psi   = reshape(psi * reshape(Ai, χL, d * χR), dᵢ * d, χR)
    end
    psi_vec = vec(psi)   # (d^L,) after χR = 1

    # Build full operator. State vector is column-major: site 1 varies fastest
    # (least significant), so site 1 must be the innermost kron factor.
    # kron(op_i, full_op) prepends site i as the new outermost block each step.
    full_op = Matrix{Float64}(I, 1, 1)
    for i in 1:L
        full_op = kron(i == site ? op : Matrix(I, d, d), full_op)
    end

    norm2 = real(dot(psi_vec, psi_vec))
    return real(dot(psi_vec, full_op * psi_vec)) / norm2
end

# ── local_expectation — physics behavior ──────────────────────────────────────

@testset "local_expectation — ⟨Sz⟩ = +1/2 at every site of |↑↑...↑⟩" begin
    L   = 5
    mps = _all_up_mps(L)
    for i in 1:L
        @test local_expectation(mps, _Sz, i) ≈ 0.5 atol=1e-12
    end
end

@testset "local_expectation — ⟨Sz⟩ of |↑↓⟩ alternates +1/2, -1/2" begin
    mps = _neel_mps(4)
    for i in 1:4
        expected = isodd(i) ? 0.5 : -0.5
        @test local_expectation(mps, _Sz, i) ≈ expected atol=1e-12
    end
end

@testset "local_expectation — agrees with full state-vector contraction" begin
    # Random entangled state; compare single-site Sz against brute-force
    L   = 4
    mps = FiniteMPS(Spin{1//2}(), L, 4)
    left_canonical_sweep!(mps)
    for i in 1:L
        via_mps      = local_expectation(mps, _Sz, i)
        via_brute    = _brute_force_expectation(mps, _Sz, i)
        @test via_mps ≈ via_brute atol=1e-10
    end
end

@testset "local_expectation — ⟨n⟩ ∈ {0,1} for Fermionic product state" begin
    L   = 4
    mps = _fermionic_neel_mps(L)
    for i in 1:L
        ev = local_expectation(mps, _n_op, i)
        @test ev ≈ Float64(iseven(i)) atol=1e-12
    end
end

# ── entanglement_spectrum — physics behavior ───────────────────────────────────

@testset "entanglement_spectrum — Bell state has Schmidt values [1/√2, 1/√2]" begin
    mps = _bell_state_mps()
    spec = entanglement_spectrum(mps, 2)
    @test length(spec) == 2
    @test sort(spec, rev=true) ≈ [1/√2, 1/√2] atol=1e-12
end

@testset "entanglement_spectrum — product state has single Schmidt value 1.0" begin
    mps = _all_up_mps(4)
    for bond in 2:4
        spec = entanglement_spectrum(mps, bond)
        @test spec ≈ [1.0] atol=1e-12
    end
end

@testset "entanglement_spectrum — consistent with entanglement_entropy" begin
    mps = FiniteMPS(Spin{1//2}(), 5, 4)
    left_canonical_sweep!(mps)
    for bond in 2:5
        spec   = entanglement_spectrum(mps, bond)
        nrm    = norm(spec)
        λ      = spec ./ nrm
        S_spec = -sum(x -> x^2 * log(x^2), λ)
        @test S_spec ≈ entanglement_entropy(mps, bond) atol=1e-12
    end
end

# ── entanglement_entropy (vector) — physics behavior ──────────────────────────

@testset "entanglement_entropy(mps) — product state: all zeros, length L-1" begin
    L   = 5
    mps = _all_up_mps(L)
    svec = entanglement_entropy(mps)
    @test length(svec) == L - 1
    @test all(≈(0.0, atol=1e-12), svec)
end

@testset "entanglement_entropy(mps) — Bell state: [log(2)]" begin
    mps  = _bell_state_mps()
    svec = entanglement_entropy(mps)
    @test length(svec) == 1
    @test svec[1] ≈ log(2) atol=1e-12
end

@testset "entanglement_entropy(mps) — vector entries match scalar calls" begin
    L   = 5
    mps = FiniteMPS(Spin{1//2}(), L, 4)
    left_canonical_sweep!(mps)
    svec = entanglement_entropy(mps)
    for b in 1:L-1
        @test svec[b] ≈ entanglement_entropy(mps, b + 1) atol=1e-14
    end
end

# ── edge cases ────────────────────────────────────────────────────────────────

@testset "local_expectation — site i=1 (left boundary, no left virtual bond)" begin
    mps = _neel_mps(4)
    @test local_expectation(mps, _Sz, 1) ≈ 0.5 atol=1e-12
end

@testset "local_expectation — site i=L (right boundary, no right virtual bond)" begin
    L   = 4
    mps = _neel_mps(L)
    @test local_expectation(mps, _Sz, L) ≈ -0.5 atol=1e-12
end

@testset "local_expectation — unnormalized state: result = ⟨ψ|O|ψ⟩/⟨ψ|ψ⟩" begin
    mps   = FiniteMPS(Spin{1//2}(), 4, 4)
    left_canonical_sweep!(mps)
    α     = 3.7
    mps_α = α * mps
    for i in 1:4
        @test local_expectation(mps_α, _Sz, i) ≈ local_expectation(mps, _Sz, i) atol=1e-10
    end
end

@testset "local_expectation — zero operator gives zero" begin
    mps  = _neel_mps(4)
    zero_op = zeros(2, 2)
    for i in 1:4
        @test local_expectation(mps, zero_op, i) ≈ 0.0 atol=1e-14
    end
end
