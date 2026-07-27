"""
Tests for: Bond — pure link geometry in a tensor network.

Physics invariants tested:
- Bond carries exactly two Upper legs (both arrows point into the centre)
- Left and right faces are distinct TIx objects
"""

using Test       # `using Test` = import Test module; exposes @test, @testset, @test_throws, etc.
using Qritical   # `using Qritical` = bring all exported names into scope 

@testset "Bond: pure link geometry" begin   # top-level testset groups all Bond tests; failures reported per testset
    @testset "Bond carries two Upper legs — both arrows point into the centre" begin
        bond = Bond(upper(:λL, 4), upper(:λR, 4))   # `Bond(left, right)` = construct a Bond between two Upper-variance legs; both Upper because bond arrows point INTO the orthogonality centre Σ from both sides
        @test bond.left == upper(:λL, 4)   # `bond.left` = field access; `==` uses our extended TIx equality (label + dim + variance)
        @test bond.right == upper(:λR, 4)   # right face: distinct label :λR but same dim=4
        @test bond.left isa TIx{Upper}     # `isa TIx{Upper}` = Python: `isinstance(bond.left, TIx[Upper])`; confirms variance is Upper
        @test bond.right isa TIx{Upper}     # same: both faces must be Upper (arrows pointing in)
    end

    @testset "left and right faces are distinct legs" begin
        bond = Bond(upper(:λL, 3), upper(:λR, 5))   # bond with asymmetric dimensions (left=3, right=5); physically possible if truncation changes rank
        @test bond.left != bond.right    # `!=` = not equal; even though both are Upper, they have different labels (:λL vs :λR) and different dims (3 vs 5)
        @test dim(bond.left) == 3       # `dim(ix)` = read the stored dimension; left face has dim 3
        @test dim(bond.right) == 5       # right face has dim 5
    end

    @testset "Bond legs match the Σ factor legs from do_svd" begin
        i = upper(:i, 3)   # input index for the tensor; Upper = domain (incoming)
        j = lower(:j, 4)   # output index; Lower = codomain (outgoing)
        A = QTensor(randn(3, 4), (i, j))   # `randn(3, 4)` = 3×4 random matrix from N(0,1) 
        bp = Bipartition(Partition([i]), Partition([j]))   # bipartition: left={i} (rows), right={j} (cols)
        F = do_svd(A, bp, NoTrunc())   # `do_svd(A, bp, NoTrunc())` = full SVD of A matricised by bp; returns FullSVD with fields .U, .Σ, .Vd, .spectrum, .center
        bond = F.center.bond   # `.center` = BondCenter; `.bond` = the Bond object linking U and Vd through Σ
        @test bond.left === F.Σ.indices[1]   # `===` = identity equality (same object in memory, not just equal value); bond.left is THE SAME TIx object as Σ's first leg (λL); no label matching needed
        @test bond.right === F.Σ.indices[2]   # bond.right IS the same object as Σ's second leg (λR); this is the key design invariant of Bond
    end
end
