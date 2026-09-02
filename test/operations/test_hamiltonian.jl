@testitem "Hamiltonian subtype relationships and stub accessors" begin
    using TensorKit

    V = ComplexSpace(2)
    h = Hamiltonian(AutomatonTerm[], 3, V)
    @test h isa Observable
    @test h isa AbstractProcess

    @test_throws ErrorException tensor(h)
    @test_throws ErrorException outputs(h)
    @test_throws ErrorException inputs(h)
end

@testitem "to_mpo(::Hamiltonian) materializes via build_automaton/materialize" begin
    using TensorKit

    V = ComplexSpace(2)
    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    L = 4
    terms = [AutomatonTerm(-1.0, [i => sz, i + 1 => sz]) for i in 1:(L - 1)]
    H = Hamiltonian(terms, L, V)

    mpo_via_H = to_mpo(H)
    mpo_direct = materialize(build_automaton(terms, L, V))

    @test mpo_via_H isa MPOperator{UnknownGauge,FiniteSupport}
    @test length(mpo_via_H.sites) == length(mpo_direct.sites)
    @test all(
        tensor(mpo_via_H.sites[i]) ≈ tensor(mpo_direct.sites[i]) for
        i in 1:length(mpo_direct.sites)
    )
end

@testitem "symmetry_group reads sectortype off physical_spaces" begin
    using TensorKit

    V = ComplexSpace(2)
    H_single = Hamiltonian(AutomatonTerm[], 3, V)
    @test symmetry_group(H_single) === Trivial

    H_vector = Hamiltonian(AutomatonTerm[], 2, [V, V])
    @test symmetry_group(H_vector) === Trivial
end
