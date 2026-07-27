"""
Tests for: attachment (src/topograph/attachment.jl).
"""

@testset "Topograph: attachment is derived, never stored" begin   # attachment(w) is a PURE FUNCTION of (start, finish, closed) never a stored field, so there is no way for it to drift out of sync with the ends
    @testset "loose means neither end attached" begin
        w = Wire(WireId(1), 2; label=:free)   # default start=finish=nothing, closed=false
        @test attachment(w) === Loose()   # neither end attached ⟹ Loose() bare identity wire / δ tensor
    end

    @testset "half-loose means exactly one end attached" begin
        w = Wire(WireId(1), 2; label=:open, start=LegId(1))   # only `start` set; `finish` stays nothing (keyword default)
        @test attachment(w) === HalfLoose()   # exactly one end attached ⟹ HalfLoose() - a free/open physical leg
    end

    @testset "pinned means both ends attached, different nodes" begin
        w = Wire(WireId(1), 2; label=:bond, start=LegId(1), finish=LegId(2))   # both ends set to DISTINCT LegIds (belonging to different owning nodes in a real network)
        @test attachment(w) === Pinned()   # both ends attached ⟹ Pinned() - a contracted bond
    end

    @testset "pinned (self-loop / trace) means both ends attached, same node" begin
        # Attachment cannot distinguish "different node" from "same node" from the Wire alone that distinction is a property of the OWNING NODES (via Leg.owner), not of the wire so both collapse to Pinned() here; a trace is a Pinned wire whose two legs share owner.
        w = Wire(WireId(1), 2; label=:trace, start=LegId(1), finish=LegId(2))   # same shape as the bond case above. attachment() genuinely cannot tell them apart, by design
        @test attachment(w) === Pinned()   # still Pinned().  the node-owner check belongs to `traces(net)` vs `bonds(net)`, not to `attachment`
    end

    @testset "circle means closed flag wins regardless of ends" begin
        w = Wire(WireId(1), 2; label=:loop, closed=true)   # `closed=true` overrides everything else. even though start/finish are both nothing here
        @test attachment(w) === Circle()   # closed ⟹ Circle(), checked FIRST in `attachment` (`w.closed && return Circle()`) - a bare scalar factor `dim(V)`, invisible to `@tensor`
    end
end
