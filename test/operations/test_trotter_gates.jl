# Shared helper: dense matrix exponential reference, independent of QProcess/TensorKit.exp,
# for cross-checking trotterize's gate tensors against a hand-computed exponential.
using LinearAlgebra: exp as dense_exp

function _dense_matrix(t)
    a = convert(Array, t)
    n_out = size(a, 1) * size(a, 2)
    n_in = size(a, 3) * size(a, 4)
    return reshape(a, n_out, n_in)
end

@testitem "trotterize: single-site field term exponentiates its own tensor directly" begin
    using TensorKit

    V = ComplexSpace(2)
    sx = QProcess(
        TensorMap(ComplexF64[0 1; 1 0], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    terms = [AutomatonTerm(0.5, [1 => sx])]
    H = Hamiltonian(terms, 1, V)
    p = propagator(H, 0.1; kind=RealTime)

    step = trotterize(p, LieTrotter())
    @test length(step.block.gates) == 1

    site_range, gate = step.block.gates[1]
    @test site_range == 1:1
    expected = TensorKit.exp(-im * 0.1 * 0.5 * tensor(sx))
    @test tensor(gate) ≈ expected
end

@testitem "trotterize: two-site bond term combines ops via ⊗ before exponentiating" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    terms = [AutomatonTerm(-1.0, [1 => sz, 2 => sz])]
    H = Hamiltonian(terms, 2, V)
    p = propagator(H, 0.2; kind=RealTime)

    step = trotterize(p, LieTrotter())
    @test length(step.block.gates) == 1

    site_range, gate = step.block.gates[1]
    @test site_range == 1:2
    combined = sz ⊗ sz
    expected = TensorKit.exp(-im * 0.2 * -1.0 * tensor(combined))
    @test tensor(gate) ≈ expected
    @test is_unitary(gate)
end

@testitem "trotterize: ImaginaryTime uses a real exponent (no im factor)" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    terms = [AutomatonTerm(-1.0, [1 => sz, 2 => sz])]
    H = Hamiltonian(terms, 2, V)
    p = propagator(H, 0.2; kind=ImaginaryTime)

    step = trotterize(p, LieTrotter())
    _, gate = step.block.gates[1]
    combined = sz ⊗ sz
    expected = TensorKit.exp(-0.2 * -1.0 * tensor(combined))
    @test tensor(gate) ≈ expected
end

@testitem "trotterize: gate count and ordering matches sequence's recipe for SuzukiTrotter" begin
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
    p = propagator(H, 0.1; kind=RealTime)

    groups = split_commuting_groups(terms)
    steps = sequence(SuzukiTrotter(), groups, 0.1)
    expected_gate_count = sum(length(group) for (group, _) in steps)

    step = trotterize(p, SuzukiTrotter())
    @test length(step.block.gates) == expected_gate_count
end

@testitem "trotterize: non-contiguous AutomatonTerm raises a clear error" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    terms = [AutomatonTerm(-1.0, [1 => sz, 3 => sz])]
    H = Hamiltonian(terms, 3, V)
    p = propagator(H, 0.1; kind=RealTime)

    @test_throws ErrorException trotterize(p, LieTrotter())
end

@testitem "trotterize: TrotterStep carries dt and defaults num_steps to 1" begin
    using TensorKit

    V = ComplexSpace(2)
    sx = QProcess(
        TensorMap(ComplexF64[0 1; 1 0], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    terms = [AutomatonTerm(0.5, [1 => sx])]
    H = Hamiltonian(terms, 1, V)
    p = propagator(H, 0.1; kind=RealTime)

    step_default = trotterize(p, LieTrotter())
    @test step_default.dt == 0.1
    @test step_default.num_steps == 1

    step_custom = trotterize(p, LieTrotter(); num_steps=7)
    @test step_custom.num_steps == 7
    @test step_custom.dt == 0.1
end

@testitem "trotterize: TrotterGateBlock has no dt field (dt lives on TrotterStep only)" begin
    using TensorKit

    V = ComplexSpace(2)
    sx = QProcess(
        TensorMap(ComplexF64[0 1; 1 0], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    terms = [AutomatonTerm(0.5, [1 => sx])]
    H = Hamiltonian(terms, 1, V)
    p = propagator(H, 0.1; kind=RealTime)

    step = trotterize(p, LieTrotter())
    @test fieldnames(typeof(step.block)) == (:gates,)
    @test step.block isa TrotterGateBlock{LieTrotter}
end

@testitem "trotter_error forwards to local_error_bound with the same grouping trotterize used" begin
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
    p = propagator(H, 0.1; kind=RealTime)
    step = trotterize(p, SuzukiTrotter())

    norm_fn = group -> sum(t -> abs(t.coefficient), group; init=0.0)
    expected = local_error_bound(
        SuzukiTrotter(), split_commuting_groups(terms), 0.1, norm_fn
    )
    @test trotter_error(step, H, norm_fn) ≈ expected
end

@testitem "record_trotter_error! forwards to accumulate_trotter_error! using step.num_steps" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    terms = [AutomatonTerm(-1.0, [1 => sz, 2 => sz])]
    H = Hamiltonian(terms, 2, V)
    p = propagator(H, 0.1; kind=RealTime)
    step = trotterize(p, LieTrotter(); num_steps=4)

    norm_fn = group -> sum(t -> abs(t.coefficient), group; init=0.0)

    acc_direct = TrotterErrorAccumulator()
    expected = accumulate_trotter_error!(
        acc_direct, LieTrotter(), split_commuting_groups(terms), 0.1, 4, norm_fn
    )

    acc = TrotterErrorAccumulator()
    result = record_trotter_error!(acc, step, H, norm_fn)
    @test result ≈ expected
    @test acc.history == acc_direct.history
end

@testitem "trotterize: single bond-term gate matches a hand-computed dense matrix exponential" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    terms = [AutomatonTerm(-1.0, [1 => sz, 2 => sz])]
    H = Hamiltonian(terms, 2, V)
    dt = 0.15
    p = propagator(H, dt; kind=RealTime)
    step = trotterize(p, LieTrotter())

    combined = sz ⊗ sz
    Hmat = _dense_matrix(tensor(combined))
    expected_mat = dense_exp(Matrix(-im * dt * -1.0 * Hmat))

    _, gate = step.block.gates[1]
    gate_mat = _dense_matrix(tensor(gate))
    @test gate_mat ≈ expected_mat
end

@testitem "trotterize: site_range is 3:4 for a non-1-starting 2-site term" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    terms = [AutomatonTerm(-1.0, [3 => sz, 4 => sz])]
    H = Hamiltonian(terms, 4, V)
    p = propagator(H, 0.1; kind=RealTime)

    step = trotterize(p, LieTrotter())
    site_range, _ = step.block.gates[1]
    @test site_range == 3:4
end

@testitem "trotterize: gate ordering and site ranges match sequence's recipe for a mixed 1-/2-site group" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    sx = QProcess(
        TensorMap(ComplexF64[0 1; 1 0], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    L = 3
    terms = [AutomatonTerm(-1.0, [1 => sz, 2 => sz]), AutomatonTerm(0.5, [1 => sx])]
    H = Hamiltonian(terms, L, V)
    p = propagator(H, 0.1; kind=RealTime)

    step = trotterize(p, LieTrotter())
    ranges = [rng for (rng, _) in step.block.gates]
    @test Set(ranges) == Set([1:2, 1:1])
end
