"""
Tests for: WireId/LegId/NodeId (src/topograph/ids.jl).
"""

@testset "Topograph: identifiers" begin
    @testset "WireId/LegId/NodeId are distinct types" begin # avoid dispatch confusion
        @test NodeId(1) isa NodeId # a node is point in Joyal's graph (G, G₀): NodeId indexes G₀, the 0-cells where wires attach
        @test WireId(1) isa WireId # a wire is a 1-manifold. indexes the 1-manifold components of G − G₀ (the wires themselves)
        @test LegId(1) isa LegId # indexes attachment points (wire, end) pairs (not a Joyal primitive but for our bookkeeping purposes)

        #a LegId can never be passed where a WireId is expected
        @test !(WireId(1) isa LegId)
    end

    @testset "identifiers support == and hash" begin
        @test WireId(3) == WireId(3)   # `==` must compare by wrapped value, not by object identity (auto-derived structural equality would need this too. we define it explicitly to be safe)
        @test WireId(3) != WireId(4)
        @test hash(WireId(3)) == hash(WireId(3))   # `hash` must agree with `==`
        d = Dict(WireId(1) => "first")   # `Dict(k => v)` = Julia dict literal
        @test d[WireId(1)] == "first"   # lookup by a FRESH WireId(1) object (not the one used to build `d`) which proves equality/hash are value-based, not identity-based
    end
end
