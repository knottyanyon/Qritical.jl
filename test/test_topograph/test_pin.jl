"""
Tests for: pin!/cut! (src/topograph/pin.jl).
"""

# Shared setup: two nodes joined by one wire, one leg per node already added to it (but not yet pinned). `mk_pair(ix_a, ix_b)` lets each test choose which side is Upper/Lower.
function mk_pair(ix_a, ix_b)
    g = TensorNetwork()
    nA = add_node!(g)
    nB = add_node!(g)
    w = add_wire!(g, dim(ix_a); label=:bond)
    A = QTensor(randn(dim(ix_a), 2), (ix_a, upper(:σ, 2)))
    B = QTensor(randn(dim(ix_b), 2), (ix_b, lower(:σ, 2)))
    ℓA = add_leg!(g, A, nA, w)
    ℓB = add_leg!(g, B, nB, w)
    return g, w, ℓA, ℓB
end

@testset "Topograph: attach!" begin
    @testset "attach! sets start for an Outgoing leg, finish for an Incoming leg" begin
        g = TensorNetwork()
        n = add_node!(g)
        w_out = add_wire!(g, 2; label=:out)
        w_in = add_wire!(g, 2; label=:in)
        A = QTensor(randn(2, 2), (lower(:σ, 2), upper(:τ, 2)))
        ℓ_out = add_leg!(g, A, n, w_out)   # A.indices[1] = lower(:σ,2) = Outgoing
        ℓ_in = add_leg!(g, A, n, w_in)     # A.indices[2] = upper(:τ,2) = Incoming
        attach!(g, ℓ_out)
        attach!(g, ℓ_in)
        @test g.wires[w_out].start == ℓ_out
        @test g.wires[w_out].finish === nothing
        @test g.wires[w_in].finish == ℓ_in
        @test g.wires[w_in].start === nothing
    end

    @testset "attach! turns a loose wire half-loose, not pinned" begin
        g = TensorNetwork()
        n = add_node!(g)
        w = add_wire!(g, 2; label=:σ)
        A = QTensor(randn(2, 2), (upper(:σ, 2), lower(:τ, 2)))
        ℓ = add_leg!(g, A, n, w)
        @test attachment(g.wires[w]) === Loose()   # add_leg! alone does not attach anything
        attach!(g, ℓ)
        @test attachment(g.wires[w]) === HalfLoose()
    end

    @testset "attach! rejects re-attaching an already-attached end" begin
        g = TensorNetwork()
        n = add_node!(g)
        w = add_wire!(g, 2; label=:σ)
        A = QTensor(randn(2, 2), (upper(:σ, 2), lower(:τ, 2)))
        ℓ = add_leg!(g, A, n, w)
        attach!(g, ℓ)
        @test_throws ArgumentError attach!(g, ℓ)
    end
end

@testset "Topograph: pin! and cut!" begin
    @testset "pin! sets start to the Outgoing (Lower) leg and finish to the Incoming (Upper) leg" begin
        g, w, ℓA, ℓB = mk_pair(lower(:vL, 4), upper(:vL, 4))
        pin!(g, ℓA, ℓB)
        wire = g.wires[w]
        @test wire.start == ℓA
        @test wire.finish == ℓB
        @test attachment(wire) === Pinned()
    end

    @testset "pin! works regardless of argument order" begin
        g, w, ℓA, ℓB = mk_pair(upper(:vL, 4), lower(:vL, 4))   # ℓA is Incoming, ℓB is Outgoing this time
        pin!(g, ℓA, ℓB)   # called (Incoming, Outgoing), still resolves to the same physical pinning
        wire = g.wires[w]
        @test wire.start == ℓB    # the Outgoing one, whichever argument position it came in on
        @test wire.finish == ℓA
    end

    @testset "pinning two Upper (Incoming) legs together is rejected" begin
        g, w, ℓA, ℓB = mk_pair(upper(:vL, 4), upper(:vL, 4))
        @test_throws ArgumentError pin!(g, ℓA, ℓB)
    end

    @testset "pinning two Lower (Outgoing) legs together is rejected" begin
        g, w, ℓA, ℓB = mk_pair(lower(:vL, 4), lower(:vL, 4))
        @test_throws ArgumentError pin!(g, ℓA, ℓB)
    end

    @testset "pinning two legs that do not share a wire is rejected" begin
        g = TensorNetwork()
        nA = add_node!(g)
        nB = add_node!(g)
        w1 = add_wire!(g, 4; label=:w1)
        w2 = add_wire!(g, 4; label=:w2)   # a second, distinct wire
        A = QTensor(randn(4, 2), (lower(:vL, 4), upper(:σ, 2)))
        B = QTensor(randn(4, 2), (upper(:vL, 4), lower(:σ, 2)))
        ℓA = add_leg!(g, A, nA, w1)
        ℓB = add_leg!(g, B, nB, w2)
        @test_throws ArgumentError pin!(g, ℓA, ℓB)
    end

    @testset "pinning an already-pinned wire is rejected" begin
        g, w, ℓA, ℓB = mk_pair(lower(:vL, 4), upper(:vL, 4))
        pin!(g, ℓA, ℓB)
        @test_throws ArgumentError pin!(g, ℓA, ℓB)
    end

    @testset "the Wire.start/Leg.wire round-trip invariant holds after pin!" begin
        # a LegId stored in Wire.start/finish must itself point back at that same wire. This is trivially true here since add_leg! already fixed each leg's .wire field, pin! only ever assigns an existing leg's own id into its own wire's start/finish, but the property is worth asserting directly rather than trusting it by construction alone.
        g, w, ℓA, ℓB = mk_pair(lower(:vL, 4), upper(:vL, 4))
        pin!(g, ℓA, ℓB)
        wire = g.wires[w]
        @test g.legs[wire.start].wire == wire.id
        @test g.legs[wire.finish].wire == wire.id
    end

    @testset "cut! reverts a pinned wire to Loose" begin
        g, w, ℓA, ℓB = mk_pair(lower(:vL, 4), upper(:vL, 4))
        pin!(g, ℓA, ℓB)
        cut!(g, w)
        wire = g.wires[w]
        @test wire.start === nothing
        @test wire.finish === nothing
        @test attachment(wire) === Loose()
    end

    @testset "cut! is idempotent on an already-loose wire" begin
        g = TensorNetwork()
        w = add_wire!(g, 4; label=:free)
        cut!(g, w)
        @test attachment(g.wires[w]) === Loose()
    end

    @testset "pin! then cut! round-trips back to the original unattached state" begin
        g, w, ℓA, ℓB = mk_pair(lower(:vL, 4), upper(:vL, 4))
        before = attachment(g.wires[w])
        pin!(g, ℓA, ℓB)
        @test attachment(g.wires[w]) === Pinned()
        cut!(g, w)
        @test attachment(g.wires[w]) === before   # both Loose(), the round trip returns to where it started
    end
end
