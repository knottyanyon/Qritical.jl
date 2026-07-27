"""
Tests for: LegOrientation, orientation, leg_orientation (src/topograph/orientation.jl).
"""

@testset "Topograph: leg orientation" begin
    @testset "orientation follows the TIx variance convention" begin
        # Upper is a contravariant, incoming index (the ψ in |ψ⟩ = A^σ|σ⟩), Lower is the covariant, outgoing basis-ket index, matching the rest of Qritical's index layer.
        @test orientation(TIx{Upper}(:σ, 2)) === Incoming()
        @test orientation(TIx{Lower}(:σ, 2)) === Outgoing()
    end

    @testset "leg_orientation matches orientation on the leg's own index" begin
        upper_leg = Leg(TIx{Upper}(:σ, 2), WireId(1), NodeId(1), 1)
        lower_leg = Leg(TIx{Lower}(:σ, 2), WireId(1), NodeId(1), 1)
        @test leg_orientation(upper_leg) === Incoming()
        @test leg_orientation(lower_leg) === Outgoing()
    end

    @testset "leg_orientation stays type-stable through a leg-table lookup" begin
        # A Dict{LegId,Leg} lookup returns a value whose static type is the unparameterized Leg (the variance parameter V is not known until runtime), this is exactly the second inference boundary named in the topograph design. leg_orientation pays the one dynamic dispatch inside _leg_orientation, @inferred checks nothing leaks past it.
        table = Dict{LegId,Leg}(LegId(1) => Leg(TIx{Upper}(:σ, 2), WireId(1), NodeId(1), 1))
        fetched = table[LegId(1)]
        @test (@inferred leg_orientation(fetched)) === Incoming()
    end
end
