# Shared helper for dense-reconstructing a materialized MPOperator against a hand/independent
# reference - lives at file top-level (outside any @testitem) since TestItems.jl treats top-level
# code preceding @testitem blocks in the same file as shared setup, the same convention already
# used by test_decompositions.jl's SpectrumEcho / test_expectation_values.jl's SnapshotEcho.
using LinearAlgebra: kron

function _boundary_channel(automaton, bond::Int, label)
    idx = findfirst(==(label), automaton.states[bond + 1])
    v = zeros(ComplexF64, length(automaton.states[bond + 1]))
    v[idx] = 1
    return v
end

function _dense_reconstruct(mpo, automaton)
    L = length(mpo.sites)
    site_arrays = [convert(Array, Qritical.tensor(mpo.sites[i])) for i in 1:L]
    d = size(site_arrays[1], 2)
    start_lbl = automaton.states[1][1]
    accept_lbl = automaton.states[1][2]
    lvec = _boundary_channel(automaton, 0, start_lbl)
    rvec = _boundary_channel(automaton, L, accept_lbl)

    kets = collect(Iterators.product(ntuple(_ -> 1:d, L)...))
    bras = collect(Iterators.product(ntuple(_ -> 1:d, L)...))
    Hmat = zeros(ComplexF64, d^L, d^L)
    for (ki, kets_) in enumerate(kets), (qi, bras_) in enumerate(bras)
        vec = lvec
        for i in 1:L
            a = site_arrays[i]
            newvec = zeros(ComplexF64, size(a, 3))
            for bin in eachindex(vec), bout in axes(a, 3)
                newvec[bout] += vec[bin] * a[bin, kets_[i], bout, bras_[i]]
            end
            vec = newvec
        end
        Hmat[ki, qi] = sum(vec .* rvec)
    end
    return Hmat
end

function _kronN(mats...)
    r = mats[1]
    for m in mats[2:end]
        r = kron(r, m)
    end
    return r
end

@testitem "build_automaton: nearest-neighbor bond dimension" begin
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
    automaton = build_automaton(terms, L, V)

    # 2 + |{terms straddling b}|: bond0=2 (none started), bond1=3 (term1 live), bond2=3 (term2
    # live), bond3=2 (both completed) - nearest-neighbor terms never widen more than 1 bond.
    @test length.(automaton.states) == [2, 3, 3, 2]
end

@testitem "build_automaton: long-range term widens exactly its span" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    L = 5
    terms = [AutomatonTerm(-1.0, [1 => sz, 5 => sz])]
    automaton = build_automaton(terms, L, V)

    # term spans sites 1..5: live at bonds 1,2,3,4 (2 + |{term}| = 3), baseline 2 at bonds 0,5.
    @test length.(automaton.states) == [2, 3, 3, 3, 3, 2]
end

@testitem "materialize: nearest-neighbor TFIM-style dense reconstruction" begin
    using TensorKit
    using LinearAlgebra

    V = ComplexSpace(2)
    Sz = ComplexF64[1 0; 0 -1]
    sz = QProcess(
        TensorMap(Sz, V ← V); output_roles=PhysicalLeg(), input_roles=PhysicalLeg()
    )
    L = 3
    terms = [
        AutomatonTerm(-1.0, [1 => sz, 2 => sz]), AutomatonTerm(-1.0, [2 => sz, 3 => sz])
    ]
    automaton = build_automaton(terms, L, V)
    mpo = materialize(automaton)
    @test mpo isa MPOperator{UnknownGauge,FiniteSupport}

    Hmat = _dense_reconstruct(mpo, automaton)
    I2 = Matrix{ComplexF64}(I, 2, 2)
    Href = -_kronN(Sz, Sz, I2) - _kronN(I2, Sz, Sz)
    @test Hmat ≈ Href atol = 1e-10
end

@testitem "materialize: long-range term dense reconstruction" begin
    using TensorKit
    using LinearAlgebra

    V = ComplexSpace(2)
    Sz = ComplexF64[1 0; 0 -1]
    sz = QProcess(
        TensorMap(Sz, V ← V); output_roles=PhysicalLeg(), input_roles=PhysicalLeg()
    )
    L = 5
    terms = [AutomatonTerm(-1.0, [1 => sz, 5 => sz])]
    automaton = build_automaton(terms, L, V)
    mpo = materialize(automaton)

    Hmat = _dense_reconstruct(mpo, automaton)
    I2 = Matrix{ComplexF64}(I, 2, 2)
    Href = -_kronN(Sz, I2, I2, I2, Sz)
    @test Hmat ≈ Href atol = 1e-10
end

@testitem "Schollwöck eq. 182-185: XXZ+field bond dimension D_W=5" begin
    using TensorKit

    V = ComplexSpace(2)
    Sp = QProcess(
        TensorMap(ComplexF64[0 1; 0 0], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    Sm = QProcess(
        TensorMap(ComplexF64[0 0; 1 0], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    sz = QProcess(
        TensorMap(ComplexF64[0.5 0; 0 -0.5], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )

    L = 4
    J, Jz, h = 1.0, 1.0, 0.5
    terms = AutomatonTerm[]
    for i in 1:(L - 1)
        push!(terms, AutomatonTerm(J / 2, [i => Sp, i + 1 => Sm]))
        push!(terms, AutomatonTerm(J / 2, [i => Sm, i + 1 => Sp]))
        push!(terms, AutomatonTerm(Jz, [i => sz, i + 1 => sz]))
    end
    for i in 1:L
        push!(terms, AutomatonTerm(-h, [i => sz]))
    end

    automaton = build_automaton(terms, L, V)
    # every bulk bond carries start, accept, and exactly the 3 in-flight NN interaction states
    # (S+S-, S-S+, SzSz) live at that specific bond - D_W = 5, matching Schollwöck eq. 184 exactly.
    for b in 1:(L - 1)
        @test length(automaton.states[b + 1]) == 5
    end

    mpo = materialize(automaton)
    @test mpo isa MPOperator{UnknownGauge,FiniteSupport}
end

@testitem "Schollwöck eq. 186-187: NN vs NNN bond-dimension cost (isolated terms)" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )

    L = 4

    # NN term alone: spans exactly 1 bond, no intermediate "bookkeeping" state needed - matches
    # Schollwöck's claim that a range-1 term costs nothing beyond the start/accept baseline.
    nn_terms = [AutomatonTerm(1.0, [2 => sz, 3 => sz])]
    nn_automaton = build_automaton(nn_terms, L, V)
    @test length.(nn_automaton.states) == [2, 2, 3, 2, 2]

    # NNN term alone: spans 2 bonds, needs exactly 1 intermediate "bookkeeping" state at the one
    # bond strictly between its endpoints - Schollwöck's eq. 187 "intermediate state 3", inserted
    # "merely as a book-keeping device". Deliberately tested in isolation from a full
    # translation-invariant NN+NNN sum: this naive per-term-lane construction gives each literal
    # AutomatonTerm its own dedicated channel rather than sharing one state per interaction *type*
    # across all sites the way Schollwöck's translation-invariant eq. 187 matrix does - so a full
    # lattice sum of overlapping same-type terms costs more here than Schollwöck's D_W=4 (that
    # state-sharing is a Phase 2 optimization, not part of this construction's correctness claim).
    nnn_terms = [AutomatonTerm(1.0, [1 => sz, 3 => sz])]
    nnn_automaton = build_automaton(nnn_terms, L, V)
    @test length.(nnn_automaton.states) == [2, 3, 3, 2, 2]

    mpo = materialize(nnn_automaton)
    @test mpo isa MPOperator{UnknownGauge,FiniteSupport}
end

@testitem "materialize: TFIM (Sz⊗Sz + Sx field) dense reconstruction" begin
    using TensorKit
    using LinearAlgebra

    L = 4
    J, g = 1.0, 0.5
    V = ComplexSpace(2)
    Sz = ComplexF64[1 0; 0 -1]
    Sx = ComplexF64[0 1; 1 0]
    sz = QProcess(
        TensorMap(Sz, V ← V); output_roles=PhysicalLeg(), input_roles=PhysicalLeg()
    )
    sx = QProcess(
        TensorMap(Sx, V ← V); output_roles=PhysicalLeg(), input_roles=PhysicalLeg()
    )

    terms = AutomatonTerm[]
    for i in 1:(L - 1)
        push!(terms, AutomatonTerm(-J, [i => sz, i + 1 => sz]))
    end
    for i in 1:L
        push!(terms, AutomatonTerm(-g * J, [i => sx]))
    end
    automaton = build_automaton(terms, L, V)
    mpo = materialize(automaton)
    Hmat = _dense_reconstruct(mpo, automaton)

    I2 = Matrix{ComplexF64}(I, 2, 2)
    function embed(op, site)
        mats = [i == site ? op : I2 for i in 1:L]
        r = mats[1]
        for m in mats[2:end]
            r = kron(r, m)
        end
        return r
    end
    H_ref = zeros(ComplexF64, 2^L, 2^L)
    for i in 1:(L - 1)
        mats = [k == i || k == i + 1 ? Sz : I2 for k in 1:L]
        H_ref .+= -J .* _kronN(mats...)
    end
    for i in 1:L
        H_ref .+= (-g * J) .* embed(Sx, i)
    end

    @test Hmat ≈ H_ref atol = 1e-10
end

@testitem "build_automaton == fill_topology(build_topology(...), ...)" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    L = 4
    terms = [AutomatonTerm(-1.0, [1 => sz, 3 => sz]), AutomatonTerm(0.5, [2 => sz])]

    direct = build_automaton(terms, L, V)
    topology = build_topology(terms, L)
    via_topology = fill_topology(topology, terms, V)

    @test direct.states == via_topology.states
    @test topology.states == direct.states
    for i in 1:L
        @test keys(direct.transitions[i]) == keys(via_topology.transitions[i])
        for key in keys(direct.transitions[i])
            c1, op1 = direct.transitions[i][key]
            c2, op2 = via_topology.transitions[i][key]
            @test c1 * tensor(op1) ≈ c2 * tensor(op2)
        end
    end
end

@testitem "fill_topology: reused topology across coefficient realizations" begin
    using TensorKit
    using LinearAlgebra

    V = ComplexSpace(2)
    Sz = ComplexF64[1 0; 0 -1]
    sz = QProcess(
        TensorMap(Sz, V ← V); output_roles=PhysicalLeg(), input_roles=PhysicalLeg()
    )
    L = 3

    site_pattern(J1, J2) =
        [AutomatonTerm(J1, [1 => sz, 2 => sz]), AutomatonTerm(J2, [2 => sz, 3 => sz])]

    topology = build_topology(site_pattern(0.0, 0.0), L)   # coefficients irrelevant to topology

    for (J1, J2) in [(1.0, -0.5), (2.3, 0.1), (-1.7, 4.2)]
        terms = site_pattern(J1, J2)
        automaton_via_topology = fill_topology(topology, terms, V)
        automaton_fresh = build_automaton(terms, L, V)

        mpo1 = materialize(automaton_via_topology)
        mpo2 = materialize(automaton_fresh)
        H1 = _dense_reconstruct(mpo1, automaton_via_topology)
        H2 = _dense_reconstruct(mpo2, automaton_fresh)
        @test H1 ≈ H2 atol = 1e-10
    end
end

@testitem "fill_topology: term-count mismatch throws ArgumentError" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    L = 3
    topology = build_topology([AutomatonTerm(1.0, [1 => sz, 2 => sz])], L)

    mismatched = [AutomatonTerm(1.0, [1 => sz, 2 => sz]), AutomatonTerm(0.5, [3 => sz])]
    @test_throws ArgumentError fill_topology(topology, mismatched, V)
end
