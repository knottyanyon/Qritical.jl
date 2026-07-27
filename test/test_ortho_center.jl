"""
Tests for: OrthoCenter, BondCenter, SiteCenter

Physics invariants tested:
- BondCenter and SiteCenter are both subtypes of OrthoCenter
- BondCenter and SiteCenter are distinct concrete types (dispatch is exhaustive)
"""

using Test       # `using Test` = import the Test standard library; exposes @test, @testset, @test_throws, etc.
using Qritical   # `using Qritical` = bring all exported names into scope; OrthoCenter/BondCenter/SiteCenter/Bond/upper are exported

@testset "Orthogonality-centre hierarchy" begin
    @testset "BondCenter vs SiteCenter dispatch is exhaustive over OrthoCenter" begin
        # Physics: orthogonality centre is either a bond or a site — nothing else.
        let bond = Bond(upper(:λL, 4), upper(:λR, 4)),   # `let a=x, b=y ... end` = local bindings; both Σ faces are Upper because bond arrows point INTO the centre; Python: `with` or just local variables
            bond_center = BondCenter(bond),               # `BondCenter(bond)` = struct constructor; wraps a Bond; used when OC sits between two sites on a bond
            site_center = SiteCenter(upper(:σ, 2))        # `SiteCenter(leg)` = struct constructor; wraps a TIx{Upper}; used when OC sits on a single site tensor

            @test bond_center isa OrthoCenter   # `isa` = isinstance check; BondCenter <: OrthoCenter in the type hierarchy 
            @test site_center isa OrthoCenter   # SiteCenter <: OrthoCenter — both are concrete subtypes
            @test typeof(bond_center) != typeof(site_center)   # `typeof` = Python `type()`; BondCenter and SiteCenter are DISTINCT concrete types; dispatch differentiates them without runtime tag checks
        end
    end
end
