# Shared dense-XXZ helper for cross-checking evolve! - builds the same 2-site XXZ generator
# xxz_hamiltonian uses, as a dense d^L x d^L matrix, via TensorKit.exp / matrix exponentiation.
using LinearAlgebra: exp as dense_exp, I as dense_I

function _dense_xxz(L, Jxy, Jz)
    Xm = ComplexF64[0 1; 1 0]
    Ym = ComplexF64[0 -im; im 0]
    Zm = ComplexF64[1 0; 0 -1]
    I2 = Matrix{ComplexF64}(dense_I, 2, 2)
    function _kronN(mats...)
        r = mats[1]
        for m in mats[2:end]
            r = kron(r, m)
        end
        return r
    end
    d = 2
    Hmat = zeros(ComplexF64, d^L, d^L)
    for i in 1:(L - 1)
        for (J, M) in ((Jxy, Xm), (Jxy, Ym), (Jz, Zm))
            ops = [j == i || j == i + 1 ? M : I2 for j in 1:L]
            Hmat += J * _kronN(ops...)
        end
    end
    return Hmat
end

function _dense_state_vector(mps)
    L = length(mps.sites)
    A = [convert(Array, tensor(mps.sites[i])) for i in 1:L]
    d = size(A[1], 2)
    ψ = zeros(ComplexF64, d^L)
    for idx in Iterators.product(ntuple(_ -> 1:d, L)...)
        vec = [1.0 + 0.0im]
        for i in 1:L
            a = A[i]
            newvec = zeros(ComplexF64, size(a, 3))
            for bin in eachindex(vec), bout in axes(a, 3)
                newvec[bout] += vec[bin] * a[bin, idx[i], bout, 1]
            end
            vec = newvec
        end
        lin = sum((idx[i] - 1) * 2^(i - 1) for i in 1:L) + 1
        ψ[lin] = vec[1]
    end
    return ψ
end

@testitem "evolve!: TFIM-like XXZ end-to-end, matches repeated-dense-Trotter-step reference" begin
    using TensorKit

    L = 2
    V = ComplexSpace(2)
    H = xxz_hamiltonian(L, V; Jxy=1.0, Jz=0.0, h=0.0)

    ψtensor = TensorMap(zeros(ComplexF64, 2, 2), V ⊗ V ← one(V))
    ψtensor[1, 1] = 0.6
    ψtensor[2, 2] = 0.8
    ψtensor = ψtensor / TensorKit.norm(ψtensor)
    ψ0 = to_mps(State(ψtensor))

    dt = 0.05
    num_steps = 5
    algorithm = TEBDAlgorithm(LieTrotter(), nothing, num_steps)
    result = evolve!(algorithm, H, ψ0, dt)

    @test is_canonical(result)
    ψ_tebd = _dense_state_vector(result)

    Hmat = _dense_xxz(L, 1.0, 0.0)
    U = dense_exp(Matrix(-im * dt * Hmat))
    ψvec = _dense_state_vector(ψ0)
    for _ in 1:num_steps
        ψvec = U * ψvec
    end

    @test isapprox(abs(ψ_tebd' * ψvec), 1.0; atol=1e-6)
end

@testitem "evolve!: norm-preservation with renormalize=true and no truncation" begin
    using TensorKit

    L = 3
    V = ComplexSpace(2)
    H = xxz_hamiltonian(L, V; Jxy=1.0, Jz=0.5, h=0.2)

    ψtensor = randn(ComplexF64, (V ⊗ V ⊗ V) ← one(V))
    ψtensor = ψtensor / TensorKit.norm(ψtensor)
    ψ0 = to_mps(State(ψtensor))

    algorithm = TEBDAlgorithm(SuzukiTrotter(), nothing, 4; renormalize=true)
    result = evolve!(algorithm, H, ψ0, 0.05)

    n = Qritical.Subroutines.norm(result)
    @test value(n) ≈ 1.0
end

@testitem "evolve!: unified collector receives TEBDStepSnapshot with observables, entropy, error" begin
    using TensorKit

    struct SnapshotCollector <: Qritical.AbstractCollector
        seen::Vector{Any}
    end
    SnapshotCollector() = SnapshotCollector(Any[])
    Qritical.step!(::Qritical.Active, c::SnapshotCollector, ctx::NamedTuple) =
        push!(c.seen, ctx.snapshot)
    Qritical.finalize!(::Qritical.Active, c::SnapshotCollector) = c.seen

    L = 3
    V = ComplexSpace(2)
    H = xxz_hamiltonian(L, V; Jxy=1.0, Jz=0.5, h=0.0)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    ψtensor = randn(ComplexF64, (V ⊗ V ⊗ V) ← one(V))
    ψtensor = ψtensor / TensorKit.norm(ψtensor)
    ψ0 = to_mps(State(ψtensor))

    collector = SnapshotCollector()
    num_steps = 4
    algorithm = TEBDAlgorithm(LieTrotter(), nothing, num_steps; snapshot_every=2)
    observables = Dict(:mag => LocalObservable(1, sz))
    evolve!(algorithm, H, ψ0, 0.05; observables=observables, collector=collector)

    @test length(collector.seen) == num_steps
    for (i, snap) in enumerate(collector.seen)
        @test snap.step == i
        @test snap.entanglement_entropy >= 0.0
        @test snap.truncation_error >= 0.0
        if i % 2 == 0
            @test snap.observables !== nothing
            @test haskey(snap.observables, :mag)
        else
            @test snap.observables === nothing
        end
        @test snap.trotter_error_bound === nothing
    end
end

@testitem "evolve!: trotter_norm callback populates trotter_error_bound" begin
    using TensorKit

    L = 2
    V = ComplexSpace(2)
    H = xxz_hamiltonian(L, V; Jxy=1.0, Jz=0.0, h=0.0)

    struct EchoCollector2 <: Qritical.AbstractCollector
        seen::Vector{Any}
    end
    EchoCollector2() = EchoCollector2(Any[])
    Qritical.step!(::Qritical.Active, c::EchoCollector2, ctx::NamedTuple) =
        push!(c.seen, ctx.snapshot)
    Qritical.finalize!(::Qritical.Active, c::EchoCollector2) = c.seen

    ψtensor = randn(ComplexF64, (V ⊗ V) ← one(V))
    ψtensor = ψtensor / TensorKit.norm(ψtensor)
    ψ0 = to_mps(State(ψtensor))

    collector = EchoCollector2()
    algorithm = TEBDAlgorithm(LieTrotter(), nothing, 2)
    norm_fn = group -> sum(t -> abs(t.coefficient), group; init=0.0)
    evolve!(algorithm, H, ψ0, 0.05; trotter_norm=norm_fn, collector=collector)

    @test all(s.trotter_error_bound !== nothing for s in collector.seen)
end

function _dense_state_vector_vidal(state)
    L = length(state.sites)
    A = [convert(Array, tensor(state.sites[i])) for i in 1:L]
    λ = [convert(Array, state.λs[i]) for i in 1:(L - 1)]
    d = size(A[1], 2)
    ψ = zeros(ComplexF64, d^L)
    for idx in Iterators.product(ntuple(_ -> 1:d, L)...)
        vec = ones(ComplexF64, 1, 1)
        for i in 1:L
            vec = vec * A[i][:, idx[i], :]
            i < L && (vec = vec * λ[i])
        end
        lin = sum((idx[i] - 1) * 2^(i - 1) for i in 1:L) + 1
        ψ[lin] = vec[1, 1]
    end
    return ψ
end

@testitem "evolve!: VidalGauge XXZ Néel-state quench end-to-end, matches dense reference" begin
    using TensorKit

    L = 3
    V = ComplexSpace(2)
    H = xxz_hamiltonian(L, V; Jxy=1.0, Jz=1.0, h=0.0)

    ψ0 = to_vidal(neel_state(L, V))

    dt = 0.05
    num_steps = 5
    algorithm = TEBDAlgorithm(LieTrotter(), nothing, num_steps)
    result = evolve!(algorithm, H, ψ0, dt)

    @test result isa MPState{VidalGauge,FiniteSupport}
    ψ_tebd = _dense_state_vector_vidal(result)

    Hmat = _dense_xxz(L, 1.0, 1.0)
    U = dense_exp(Matrix(-im * dt * Hmat))
    ψvec = _dense_state_vector(neel_state(L, V))
    for _ in 1:num_steps
        ψvec = U * ψvec
    end

    # Against the *exact* exponential (not a repeated-dense-Trotter-step reference), a genuine
    # O(dt) LieTrotter error is expected here - atol matches that order of magnitude, not exact
    # numerical agreement.
    @test isapprox(abs(ψ_tebd' * ψvec), 1.0; atol=1e-2)
end

@testitem "evolve!: VidalGauge Néel-state quench shows entanglement entropy growth" begin
    using TensorKit

    L = 4
    V = ComplexSpace(2)
    H = xxz_hamiltonian(L, V; Jxy=1.0, Jz=1.0, h=0.0)
    ψ0 = to_vidal(neel_state(L, V))

    struct EntropyCollector <: Qritical.AbstractCollector
        seen::Vector{Float64}
    end
    EntropyCollector() = EntropyCollector(Float64[])
    Qritical.step!(::Qritical.Active, c::EntropyCollector, ctx::NamedTuple) =
        push!(c.seen, ctx.snapshot.entanglement_entropy)
    Qritical.finalize!(::Qritical.Active, c::EntropyCollector) = c.seen

    collector = EntropyCollector()
    algorithm = TEBDAlgorithm(LieTrotter(), nothing, 10)
    evolve!(algorithm, H, ψ0, 0.05; collector=collector)

    @test collector.seen[end] > collector.seen[1]
    @test all(s -> s >= -1e-8, collector.seen)
end
