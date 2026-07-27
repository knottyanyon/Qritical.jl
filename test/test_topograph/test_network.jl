"""
Tests for: TensorNetwork (src/topograph/network.jl).
"""

@testset "Topograph: TensorNetwork" begin
    @testset "add_node!/add_wire!/add_leg! populate the tables" begin
        g = TensorNetwork()
        n = add_node!(g)
        w = add_wire!(g, 3; label=:vL)
        A = QTensor(randn(3, 4), (upper(:i, 3), lower(:j, 4)))
        ℓ = add_leg!(g, A, n, w)
        @test nodes(g) == [n]
        @test wires(g) == [w]
        @test legs(g, n) == [ℓ]
        @test g.legs[ℓ].owner == n
        @test g.legs[ℓ].wire == w
        @test g.legs[ℓ].axis == 1   # this node's first leg
    end

    @testset "add_leg! assigns axes in the order legs are added" begin
        g = TensorNetwork()
        n = add_node!(g)
        w1 = add_wire!(g, 3; label=:i)
        w2 = add_wire!(g, 4; label=:j)
        A = QTensor(randn(3, 4), (upper(:i, 3), lower(:j, 4)))
        ℓ1 = add_leg!(g, A, n, w1)
        ℓ2 = add_leg!(g, A, n, w2)
        @test g.legs[ℓ1].axis == 1
        @test g.legs[ℓ2].axis == 2
        @test legs(g, n) == [ℓ1, ℓ2]
    end

    @testset "add_leg! rejects a leg whose dimension does not match its wire's Int space" begin
        g = TensorNetwork()
        n = add_node!(g)
        w = add_wire!(g, 99; label=:mismatched)   # wire declares dim 99, tensor's first leg has dim 3
        A = QTensor(randn(3, 4), (upper(:i, 3), lower(:j, 4)))
        @test_throws ArgumentError add_leg!(g, A, n, w)
    end

    @testset "ends: a wire touching two different nodes returns both" begin
        g = TensorNetwork()
        nA = add_node!(g)
        nB = add_node!(g)
        w = add_wire!(g, 4; label=:bond)
        A = QTensor(randn(4, 2), (lower(:vL, 4), upper(:σ, 2)))
        B = QTensor(randn(4, 2), (upper(:vL, 4), lower(:σ, 2)))
        ℓA = add_leg!(g, A, nA, w)
        ℓB = add_leg!(g, B, nB, w)
        pin!(g, ℓA, ℓB)
        @test ends(g, w) == Set([nA, nB])
    end

    @testset "ends: an unattached wire touches no nodes" begin
        g = TensorNetwork()
        w = add_wire!(g, 4; label=:free)
        @test ends(g, w) == Set{NodeId}()
    end

    @testset "incident and degree: a node's valence is how many wires touch it" begin
        g = TensorNetwork()
        n = add_node!(g)
        w1 = add_wire!(g, 3; label=:i)
        w2 = add_wire!(g, 4; label=:j)
        A = QTensor(randn(3, 4), (upper(:i, 3), lower(:j, 4)))
        add_leg!(g, A, n, w1)
        add_leg!(g, A, n, w2)
        @test Set(incident(g, n)) == Set([w1, w2])
        @test degree(g, n) == 2
    end
end
