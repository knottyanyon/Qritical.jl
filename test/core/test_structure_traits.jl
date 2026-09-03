@testitem "StructureInfo: HasStructure trait family" begin
    using TensorKit

    @testitem "symmetry_structure on TIx" begin
        @test symmetry_structure(typeof(TIx(4))) isa NoSymmetryInfo
        @test !carries_symmetry_info(TIx(4))

        g = TIx(GradedSpace(Z2Irrep(0) => 2, Z2Irrep(1) => 3))
        @test symmetry_structure(typeof(g)) isa CarriesSymmetryInfo
        @test carries_symmetry_info(g)
    end

    @testitem "entanglement_structure on TIx" begin
        @test entanglement_structure(typeof(TIx(4))) isa NoEntanglementInfo   # PhysicalLeg default
        @test !carries_entanglement_info(TIx(4))

        χ = TIx(8, VirtualLeg())
        @test entanglement_structure(typeof(χ)) isa CarriesEntanglementInfo
        @test carries_entanglement_info(χ)
    end

    @testitem "leg participates in TIx equality" begin
        @test TIx(4) == TIx(4, PhysicalLeg())
        @test TIx(4) != TIx(4, VirtualLeg())
    end

    @testitem "symmetry_structure and entanglement_structure on MulTIx" begin
        α = TIx(3)
        σ = TIx(2)

        g = MulTIx(:ασ, (α, σ))
        @test symmetry_structure(g) isa NoSymmetryInfo
        @test entanglement_structure(g) isa CarriesEntanglementInfo   # default leg = VirtualLeg()
        @test carries_entanglement_info(g)

        h = MulTIx(:ασ, (α, σ), PhysicalLeg())
        @test entanglement_structure(h) isa NoEntanglementInfo
        @test !carries_entanglement_info(h)

        a = TIx(GradedSpace(Z2Irrep(0) => 1, Z2Irrep(1) => 1))
        b = TIx(GradedSpace(Z2Irrep(0) => 1, Z2Irrep(1) => 1))
        gg = MulTIx(:ab, (a, b))
        @test symmetry_structure(gg) isa CarriesSymmetryInfo
        @test carries_symmetry_info(gg)
    end
end
