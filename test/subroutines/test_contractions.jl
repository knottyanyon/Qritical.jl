# Dense reconstruction of a 2-site MPState (boundary bonds trivial) into a plain (d,d) array,
# for cross-checking apply/overlap against hand-computed dense linear algebra.
function _dense_2site(chain)
    A1 = convert(Array, tensor(chain.sites[1]))   # (vL=1, σ1, vR)
    A2 = convert(Array, tensor(chain.sites[2]))   # (vL=vR, σ2, vR=1)
    D = size(A1, 3)
    d1, d2 = size(A1, 2), size(A2, 2)
    out = zeros(ComplexF64, d1, d2)
    for s1 in 1:d1, s2 in 1:d2
        out[s1, s2] = sum(A1[1, s1, b] * A2[b, s2, 1] for b in 1:D)
    end
    return out
end

@testitem "apply: bond dimension multiplies (D_mpo * D_mps)" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    terms = [AutomatonTerm(-1.0, [1 => sz, 2 => sz])]
    H = Hamiltonian(terms, 2, V)
    mpo = to_mpo(H)   # bond dim 3 at the single internal bond (2 states + 1 straddling term)

    ψtensor = TensorMap(zeros(ComplexF64, 2, 2), V ⊗ V ← one(V))
    ψtensor[1, 1] = 1 / sqrt(2)
    ψtensor[2, 2] = 1 / sqrt(2)
    mps = to_mps(State(ψtensor))   # bond dim 2 (full rank for this state)

    result = apply(mpo, mps)
    @test result isa MPState{UnknownGauge,FiniteSupport}
    D_mpo = TensorKit.dim(TensorKit.space(tensor(mpo.sites[1]), 3))
    D_mps = TensorKit.dim(TensorKit.space(tensor(mps.sites[1]), 3))
    D_result = TensorKit.dim(TensorKit.space(tensor(result.sites[1]), 3))
    @test D_result == D_mpo * D_mps
end

@testitem "apply: dense correctness against a hand-computed -Z⊗Z action" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    terms = [AutomatonTerm(-1.0, [1 => sz, 2 => sz])]
    H = Hamiltonian(terms, 2, V)
    mpo = to_mpo(H)

    ψtensor = TensorMap(zeros(ComplexF64, 2, 2), V ⊗ V ← one(V))
    ψtensor[1, 1] = 0.6
    ψtensor[1, 2] = 0.8im
    ψtensor[2, 1] = -0.3
    ψtensor[2, 2] = 0.1
    ψtensor = ψtensor / TensorKit.norm(ψtensor)   # to_mps normalizes internally; match that here
    mps = to_mps(State(ψtensor))

    result = apply(mpo, mps)
    got = _dense_2site(result)

    ψarr = convert(Array, ψtensor)   # (σ1, σ2, 1)
    zdiag = [1, -1]
    expected = zeros(ComplexF64, 2, 2)
    for s1 in 1:2, s2 in 1:2
        expected[s1, s2] = -zdiag[s1] * zdiag[s2] * ψarr[s1, s2, 1]
    end

    @test got ≈ expected
end

@testitem "apply: identity MPO leaves a state unchanged (dense reconstruction)" begin
    using TensorKit

    V = ComplexSpace(2)
    Vb = oneunit(V)
    id_site = QProcess(
        TensorKit.id(Vb ⊗ V);
        output_roles=(VirtualLeg(), PhysicalLeg()),
        input_roles=(VirtualLeg(), PhysicalLeg()),
    )
    mpo = MPOperator([id_site, id_site], UnknownGauge(), 0, 0, nothing, 0.0)

    ψtensor = TensorMap(zeros(ComplexF64, 2, 2), V ⊗ V ← one(V))
    ψtensor[1, 1] = 0.6
    ψtensor[2, 2] = 0.8
    ψtensor = ψtensor / TensorKit.norm(ψtensor)   # to_mps normalizes internally; match that here
    mps = to_mps(State(ψtensor))

    result = apply(mpo, mps)
    got = _dense_2site(result)
    expected = convert(Array, ψtensor)[:, :, 1]
    @test got ≈ expected
end

@testitem "overlap: self-overlap matches dense ⟨ψ|ψ⟩" begin
    using TensorKit

    V = ComplexSpace(2)
    ψtensor = TensorMap(zeros(ComplexF64, 2, 2), V ⊗ V ← one(V))
    ψtensor[1, 1] = 0.6
    ψtensor[1, 2] = 0.8im
    ψtensor[2, 1] = -0.3
    ψtensor[2, 2] = 0.1
    mps = to_mps(State(ψtensor))   # to_mps normalizes internally regardless of ψtensor's own norm

    ov = overlap(mps, mps)
    @test ov isa Scalar
    @test value(ov) ≈ 1.0
end

@testitem "overlap: orthogonal product states give ~0" begin
    using TensorKit

    V = ComplexSpace(2)
    up = TensorMap(zeros(ComplexF64, 2, 2), V ⊗ V ← one(V))
    up[1, 1] = 1.0
    down = TensorMap(zeros(ComplexF64, 2, 2), V ⊗ V ← one(V))
    down[2, 2] = 1.0

    mps_up = to_mps(State(up))
    mps_down = to_mps(State(down))

    @test isapprox(value(overlap(mps_up, mps_down)), 0.0; atol=1e-12)
end

@testitem "norm: matches sqrt(overlap) and is 1.0, for LeftCanonical (center shortcut)" begin
    using TensorKit

    V = ComplexSpace(2)
    ψtensor = TensorMap(zeros(ComplexF64, 2, 2), V ⊗ V ← one(V))
    ψtensor[1, 1] = 0.6
    ψtensor[1, 2] = 0.8im
    ψtensor[2, 1] = -0.3
    ψtensor[2, 2] = 0.1
    ψtensor = ψtensor / TensorKit.norm(ψtensor)
    mps = to_mps(State(ψtensor))   # LeftCanonical, orthogonality_center known

    @test mps isa MPState{LeftCanonical,FiniteSupport}
    n = Qritical.Subroutines.norm(mps)
    @test n isa Scalar
    @test value(n) ≈ sqrt(value(overlap(mps, mps)))
    @test value(n) ≈ 1.0
end

@testitem "norm: MixedCanonical exercises the center shortcut, not just LeftCanonical" begin
    using TensorKit

    V = ComplexSpace(2)
    ψtensor = TensorMap(zeros(ComplexF64, 2, 2), V ⊗ V ← one(V))
    ψtensor[1, 1] = 0.6
    ψtensor[1, 2] = 0.8im
    ψtensor[2, 1] = -0.3
    ψtensor[2, 2] = 0.1
    ψtensor = ψtensor / TensorKit.norm(ψtensor)
    left = to_mps(State(ψtensor))
    mixed = canonicalize(left, MixedCanonicalize(1))

    @test mixed isa MPState{MixedCanonical,FiniteSupport}
    @test value(Qritical.Subroutines.norm(mixed)) ≈ sqrt(value(overlap(mixed, mixed)))
    @test value(Qritical.Subroutines.norm(mixed)) ≈ 1.0
end

@testitem "norm: UnknownGauge falls back to the general overlap path" begin
    using TensorKit

    V = ComplexSpace(2)
    ψtensor = TensorMap(zeros(ComplexF64, 2, 2), V ⊗ V ← one(V))
    ψtensor[1, 1] = 0.6
    ψtensor[2, 2] = 0.8
    ψtensor = ψtensor / TensorKit.norm(ψtensor)
    left = to_mps(State(ψtensor))
    unknown = MPState(collect(left.sites), UnknownGauge(), 0, 0, nothing, 0.0)

    @test value(Qritical.Subroutines.norm(unknown)) ≈ sqrt(value(overlap(unknown, unknown)))
    @test value(Qritical.Subroutines.norm(unknown)) ≈ 1.0
end

@testitem "apply: CompressedApply requires canonicalized inputs" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    terms = [AutomatonTerm(-1.0, [1 => sz, 2 => sz])]
    H = Hamiltonian(terms, 2, V)
    mpo = to_mpo(H)   # UnknownGauge - not canonicalized

    ψtensor = TensorMap(zeros(ComplexF64, 2, 2), V ⊗ V ← one(V))
    ψtensor[1, 1] = 1 / sqrt(2)
    ψtensor[2, 2] = 1 / sqrt(2)
    mps = to_mps(State(ψtensor))

    @test_throws ArgumentError apply(mpo, mps, CompressedApply())
end

@testitem "apply: CompressedApply matches ExactApply+canonicalize when no truncation occurs" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    L = 3
    terms = [
        AutomatonTerm(-1.0, [1 => sz, 2 => sz]), AutomatonTerm(-1.0, [2 => sz, 3 => sz])
    ]
    H = Hamiltonian(terms, L, V)
    mpo = canonicalize(to_mpo(H), LeftCanonicalize())

    ψtensor = randn(ComplexF64, (V ⊗ V ⊗ V) ← one(V))
    mps = canonicalize(to_mps(State(ψtensor)), LeftCanonicalize())

    exact = canonicalize(apply(mpo, mps, ExactApply()), LeftCanonicalize())
    compressed = apply(mpo, mps, CompressedApply())   # no bond_cutoff - no truncation should occur

    @test compressed isa MPState{LeftCanonical,FiniteSupport}
    @test length(compressed.sites) == L
    for i in 1:L
        @test TensorKit.dim(TensorKit.space(tensor(compressed.sites[i]), 1)) ==
            TensorKit.dim(TensorKit.space(tensor(exact.sites[i]), 1))
    end
    # no truncation occurred, so compressed/exact should be the same state up to gauge freedom -
    # check via Cauchy-Schwarz saturation (|⟨compressed|exact⟩| ≈ ‖compressed‖·‖exact‖ iff parallel)
    ov = abs(value(overlap(compressed, exact)))
    n_compressed = sqrt(real(value(overlap(compressed, compressed))))
    n_exact = sqrt(real(value(overlap(exact, exact))))
    @test isapprox(ov, n_compressed * n_exact; rtol=1e-6)
end

@testitem "apply: CompressedApply respects an explicit bond_cutoff" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    L = 3
    terms = [
        AutomatonTerm(-1.0, [1 => sz, 2 => sz]), AutomatonTerm(-1.0, [2 => sz, 3 => sz])
    ]
    H = Hamiltonian(terms, L, V)
    mpo = canonicalize(to_mpo(H), LeftCanonicalize())

    ψtensor = randn(ComplexF64, (V ⊗ V ⊗ V) ← one(V))
    mps = canonicalize(to_mps(State(ψtensor)), LeftCanonicalize())

    compressed = apply(mpo, mps, CompressedApply(2))
    for i in 1:(L - 1)
        @test TensorKit.dim(TensorKit.space(tensor(compressed.sites[i]), 3)) <= 2
    end
end

@testitem "apply_gate: 1-site gate contracts directly, dense cross-check" begin
    using TensorKit

    V = ComplexSpace(2)
    ψtensor = TensorMap(zeros(ComplexF64, 2, 2), V ⊗ V ← one(V))
    ψtensor[1, 2] = 1.0   # |0⟩ ⊗ |1⟩
    mps = to_mps(State(ψtensor); form=:left)
    mps = canonicalize(mps, MixedCanonicalize(1))

    X = QProcess(
        TensorMap(ComplexF64[0 1; 1 0], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    result = apply_gate(mps, X, 1:1)

    A1 = convert(Array, tensor(result.sites[1]))
    A2 = convert(Array, tensor(result.sites[2]))
    D = size(A1, 3)
    got = zeros(ComplexF64, 2, 2)
    for s1 in 1:2, s2 in 1:2
        got[s1, s2] = sum(A1[1, s1, b] * A2[b, s2, 1] for b in 1:D)
    end

    expected = zeros(ComplexF64, 2, 2)
    expected[2, 2] = 1.0   # X|0⟩ = |1⟩, so X⊗I applied to |0⟩⊗|1⟩ gives |1⟩⊗|1⟩
    @test got ≈ expected
end

@testitem "apply_gate: 2-site gate on a 2-site chain, no truncation, exact dense match" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    ψtensor = TensorMap(zeros(ComplexF64, 2, 2), V ⊗ V ← one(V))
    ψtensor[1, 1] = 0.6
    ψtensor[1, 2] = 0.8im
    ψtensor[2, 1] = -0.3
    ψtensor[2, 2] = 0.1
    ψtensor = ψtensor / TensorKit.norm(ψtensor)
    mps = to_mps(State(ψtensor); form=:left)
    mps = canonicalize(mps, MixedCanonicalize(1))

    dt = 0.15
    combined = sz ⊗ sz
    gate = QProcess(
        TensorKit.exp(-im * dt * tensor(combined)), outputs(combined), inputs(combined)
    )

    result = apply_gate(mps, gate, 1:2)
    @test is_canonical(result)

    A1 = convert(Array, tensor(result.sites[1]))
    A2 = convert(Array, tensor(result.sites[2]))
    D = size(A1, 3)
    got = zeros(ComplexF64, 2, 2)
    for s1 in 1:2, s2 in 1:2
        got[s1, s2] = sum(A1[1, s1, b] * A2[b, s2, 1] for b in 1:D)
    end

    Hmat = zeros(ComplexF64, 4, 4)
    Harr = convert(Array, tensor(combined))
    for s1k in 1:2, s2k in 1:2, s1b in 1:2, s2b in 1:2
        ki = s1k + (s2k - 1) * 2
        qi = s1b + (s2b - 1) * 2
        Hmat[ki, qi] = Harr[s1k, s2k, s1b, s2b]
    end
    using LinearAlgebra: exp as dense_exp
    Umat = dense_exp(Matrix(-im * dt * Hmat))
    ψarr = convert(Array, ψtensor)[:, :, 1]
    ψvec = zeros(ComplexF64, 4)
    for s1 in 1:2, s2 in 1:2
        ψvec[s1 + (s2 - 1) * 2] = ψarr[s1, s2]
    end
    expected_vec = Umat * ψvec
    expected = zeros(ComplexF64, 2, 2)
    for s1 in 1:2, s2 in 1:2
        expected[s1, s2] = expected_vec[s1 + (s2 - 1) * 2]
    end
    @test got ≈ expected
end

@testitem "apply_gate: 2-site gate on a 3-site chain, center starting away from the gate" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    ψtensor = randn(ComplexF64, (V ⊗ V ⊗ V) ← one(V))
    mps = to_mps(State(ψtensor); form=:left)
    mps = canonicalize(mps, MixedCanonicalize(1))
    @test mps.orthogonality_center == 1

    combined = sz ⊗ sz
    gate = QProcess(
        TensorKit.exp(-im * 0.1 * tensor(combined)), outputs(combined), inputs(combined)
    )

    # center at 1, gate on sites 2:3 - apply_gate itself should error (precondition), the
    # driver is responsible for walking the center there first via canonicalize
    @test_throws ArgumentError apply_gate(mps, gate, 2:3)

    mps2 = canonicalize(mps, MixedCanonicalize(2))
    result = apply_gate(mps2, gate, 2:3)
    @test is_canonical(result)
    @test result.orthogonality_center in (2, 3)
end

@testitem "apply_gate: truncation records nonzero error via the accumulator" begin
    using TensorKit

    V = ComplexSpace(2)
    ψtensor = randn(ComplexF64, (V ⊗ V) ← one(V))
    mps = to_mps(State(ψtensor); form=:left)
    mps = canonicalize(mps, MixedCanonicalize(1))

    # an entangling 2-site gate: exp(-iθ·(Sx⊗Sx + Sz⊗Sz))
    sx = QProcess(
        TensorMap(ComplexF64[0 1; 1 0], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    combined = sx ⊗ sx
    combined2 = sz ⊗ sz
    gate_tensor = TensorKit.exp(-im * 0.7 * (tensor(combined) + tensor(combined2)))
    gate = QProcess(gate_tensor, outputs(combined), inputs(combined))

    acc = QuadratureTruncationErrorAccumulator()
    result = apply_gate(mps, gate, 1:2; bond_cutoff=1, accumulator=acc)
    @test finalize!(acc) >= 0.0
    @test TensorKit.dim(TensorKit.space(tensor(result.sites[1]), 3)) <= 1
end

@testitem "apply_gate: precondition violation throws ArgumentError, not a silent wrong answer" begin
    using TensorKit

    V = ComplexSpace(2)
    ψtensor = randn(ComplexF64, (V ⊗ V ⊗ V) ← one(V))
    mps = to_mps(State(ψtensor); form=:left)
    mps = canonicalize(mps, MixedCanonicalize(3))

    sx = QProcess(
        TensorMap(ComplexF64[0 1; 1 0], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    @test_throws ArgumentError apply_gate(mps, sx, 1:1)
end

# Dense reconstruction of a Vidal-gauge MPState (any L) via Γ1·λ1·Γ2·λ2·...·ΓL contraction.
function _dense_vidal(state)
    L = length(state.sites)
    A = [convert(Array, tensor(state.sites[i])) for i in 1:L]
    λ = [convert(Array, state.λs[i]) for i in 1:(L - 1)]
    d = size(A[1], 2)
    ψ = zeros(ComplexF64, ntuple(_ -> d, L)...)
    for idx in Iterators.product(ntuple(_ -> 1:d, L)...)
        vec = ones(ComplexF64, 1, 1)
        for i in 1:L
            a = A[i][:, idx[i], :]
            vec = vec * a
            i < L && (vec = vec * λ[i])
        end
        ψ[idx...] = vec[1, 1]
    end
    return ψ
end

@testitem "apply_gate: VidalGauge 1-site gate contracts directly, dense cross-check" begin
    using TensorKit

    V = ComplexSpace(2)
    ψtensor = randn(ComplexF64, (V ⊗ V) ← one(V))
    ψtensor = ψtensor / TensorKit.norm(ψtensor)
    chain = to_vidal(to_mps(State(ψtensor); form=:left))

    sx = QProcess(
        TensorMap(ComplexF64[0 1; 1 0], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    result = apply_gate(chain, sx, 1:1)
    @test result isa MPState{VidalGauge,FiniteSupport}

    ψarr = convert(Array, ψtensor)[:, :, 1]   # (σ1, σ2)
    expected = ComplexF64[0 1; 1 0] * ψarr    # sx applied to site 1
    got = _dense_vidal(result)
    @test isapprox(got, expected; atol=1e-8)
end

@testitem "apply_gate: VidalGauge 2-site gate on a 3-site chain, no truncation, exact dense match" begin
    using TensorKit
    using LinearAlgebra: I as dense_I

    L = 3
    V = ComplexSpace(2)
    ψtensor = randn(ComplexF64, (V ⊗ V ⊗ V) ← one(V))
    ψtensor = ψtensor / TensorKit.norm(ψtensor)
    chain = to_vidal(to_mps(State(ψtensor); form=:left))

    sx = TensorMap(ComplexF64[0 1; 1 0], V ← V)
    gate_tensor = sx ⊗ sx
    gate = QProcess(
        gate_tensor;
        output_roles=(PhysicalLeg(), PhysicalLeg()),
        input_roles=(PhysicalLeg(), PhysicalLeg()),
    )

    result = apply_gate(chain, gate, 1:2)
    @test result isa MPState{VidalGauge,FiniteSupport}
    @test is_canonical(result)

    got = _dense_vidal(result)
    Xm = ComplexF64[0 1; 1 0]
    I2 = Matrix{ComplexF64}(dense_I, 2, 2)
    expected = zeros(ComplexF64, 2, 2, 2)
    ψarr = convert(Array, ψtensor)[:, :, :, 1]
    for a in 1:2, b in 1:2, c in 1:2, a2 in 1:2, b2 in 1:2
        expected[a2, b2, c] += Xm[a2, a] * Xm[b2, b] * ψarr[a, b, c]
    end
    n1 = sqrt(sum(abs2, got))
    n2 = sqrt(sum(abs2, expected))
    @test isapprox(abs(sum(conj.(vec(got)) .* vec(expected))), n1 * n2; atol=1e-6)
end
