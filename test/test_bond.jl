"""
Tests for: Bond — pure link geometry in a tensor network.

Physics invariants tested:
- Bond carries exactly two Upper legs (both arrows point into the centre)
- Left and right faces are distinct TIx objects
"""

using Test
using Qritical

@testset "Bond: pure link geometry" begin

    @testset "Bond carries two Upper legs — both arrows point into the centre" begin
        bond = Bond(upper(:λL, 4), upper(:λR, 4))
        @test bond.left  == upper(:λL, 4)
        @test bond.right == upper(:λR, 4)
        @test bond.left  isa TIx{Upper}
        @test bond.right isa TIx{Upper}
    end

    @testset "left and right faces are distinct legs" begin
        bond = Bond(upper(:λL, 3), upper(:λR, 5))
        @test bond.left != bond.right
        @test dim(bond.left)  == 3
        @test dim(bond.right) == 5
    end

    @testset "Bond legs match the Σ factor legs from do_svd" begin
        i = upper(:i, 3)
        j = lower(:j, 4)
        A = QTensor(randn(3, 4), (i, j))
        bp = Bipartition(Partition([i]), Partition([j]))
        F = do_svd(A, bp, NoTrunc())
        bond = F.center.bond
        @test bond.left  === F.Σ.indices[1]   # same TIx object, no label matching
        @test bond.right === F.Σ.indices[2]
    end

end
