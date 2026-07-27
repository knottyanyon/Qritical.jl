"""
Tests for: is_progressive, has_circuit (src/topograph/progressive.jl).

Physical legs are left out of these networks entirely: compactify only ever adds a fresh, degree-1 outer node for each open leg, and a degree-1 node can never sit on a cycle, so they cannot affect has_circuit and only add noise to what these tests are checking.
"""

@testset "Topograph: is_progressive and has_circuit" begin
    @testset "an open (OBC-MPS-like) chain is progressive" begin
        # site1 --vR/vL-- site2 --vR/vL-- site3, no bond closes back on itself.
        g = TensorNetwork()
        s1, s2, s3 = add_node!(g), add_node!(g), add_node!(g)
        A1 = QTensor(randn(2), (upper(:vR, 2),))
        A2 = QTensor(randn(2, 2), (lower(:vL, 2), upper(:vR, 2)))
        A3 = QTensor(randn(2), (lower(:vL, 2),))
        w12 = add_wire!(g, 2; label=:bond12)
        w23 = add_wire!(g, 2; label=:bond23)
        ℓ1R = add_leg!(g, A1, s1, w12)
        ℓ2L = add_leg!(g, A2, s2, w12)
        ℓ2R = add_leg!(g, A2, s2, w23)
        ℓ3L = add_leg!(g, A3, s3, w23)
        pin!(g, ℓ1R, ℓ2L)
        pin!(g, ℓ2R, ℓ3L)
        @test is_ordinary(g)
        @test !has_circuit(compactify(g))
        @test is_progressive(g)
    end

    @testset "a closed ring (PBC-MPS-like) is not progressive" begin
        # Same three sites, but bond31 closes site3.vR back onto site1.vL: a genuine 3-cycle.
        g = TensorNetwork()
        s1, s2, s3 = add_node!(g), add_node!(g), add_node!(g)
        A1 = QTensor(randn(2, 2), (lower(:vL, 2), upper(:vR, 2)))
        A2 = QTensor(randn(2, 2), (lower(:vL, 2), upper(:vR, 2)))
        A3 = QTensor(randn(2, 2), (lower(:vL, 2), upper(:vR, 2)))
        w12 = add_wire!(g, 2; label=:bond12)
        w23 = add_wire!(g, 2; label=:bond23)
        w31 = add_wire!(g, 2; label=:bond31)
        ℓ1L = add_leg!(g, A1, s1, w31)
        ℓ1R = add_leg!(g, A1, s1, w12)
        ℓ2L = add_leg!(g, A2, s2, w12)
        ℓ2R = add_leg!(g, A2, s2, w23)
        ℓ3L = add_leg!(g, A3, s3, w23)
        ℓ3R = add_leg!(g, A3, s3, w31)
        pin!(g, ℓ1R, ℓ2L)
        pin!(g, ℓ2R, ℓ3L)
        pin!(g, ℓ3R, ℓ1L)
        @test is_ordinary(g)   # no circles, this is a genuine cycle among pinned bonds, not a Circle() wire
        @test has_circuit(compactify(g))
        @test !is_progressive(g)
    end

    @testset "a trace (both legs of one wire on the same node) is not progressive" begin
        g = TensorNetwork()
        n = add_node!(g)
        A = QTensor(randn(2, 2), (lower(:vL, 2), upper(:vR, 2)))
        w = add_wire!(g, 2; label=:trace)
        ℓL = add_leg!(g, A, n, w)
        ℓR = add_leg!(g, A, n, w)
        pin!(g, ℓR, ℓL)
        @test attachment(g.wires[w]) === Pinned()
        @test has_circuit(compactify(g))
        @test !is_progressive(g)
    end
end
