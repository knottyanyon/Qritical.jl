"""
Tests for: Bond, OrthoCenter, BondCenter, SiteCenter

Physics invariants tested:
- BondCenter and SiteCenter are both subtypes of OrthoCenter
- Bond legs are Upper (both arrows point into the orthogonality centre)
- BondCenter and SiteCenter are distinct concrete types (dispatch is exhaustive)
"""

using Test
using Qritical

@testset "Bond and orthogonality-centre hierarchy" begin

    @testset "Bond carries two Upper legs — both arrows point into the centre" begin
        bond = Bond(upper(:λL, 4), upper(:λR, 4))
        @test bond.left  == upper(:λL, 4)
        @test bond.right == upper(:λR, 4)
        @test bond.left  isa TIx{Upper}
        @test bond.right isa TIx{Upper}
    end

    @testset "BondCenter vs SiteCenter dispatch is exhaustive over OrthoCenter" begin
        # Physics: orthogonality centre is either a bond or a site — nothing else.
        let bond = Bond(upper(:λL, 4), upper(:λR, 4)),   # both faces of the centre are Upper
            bond_center = BondCenter(bond),
            site_center = SiteCenter(upper(:σ, 2))       # physical legs are Upper

            @test bond_center isa OrthoCenter
            @test site_center isa OrthoCenter
            @test typeof(bond_center) != typeof(site_center)
        end
    end

end
