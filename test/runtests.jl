using Qritical
using Test
using Aqua

@testset "Qritical.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Qritical)
    end

    include("test_tensor_index.jl")

    @testset "Bisection" begin
        @testset "construction" begin
            b = Bisection([1, 2], 5)
            @test b.left == [1, 2]
            @test b.right == [3, 4, 5]

            b2 = Bisection([1, 2], [3, 4, 5])
            @test b2.left == [1, 2]
            @test b2.right == [3, 4, 5]

            b3 = Bisection([1], 10)
            @test b3.left == [1]
            @test b3.right == collect(2:10)
        end

        @testset "validation" begin
            @test_throws ArgumentError Bisection([1], [1, 2])       # overlap
            @test_throws ArgumentError Bisection([1, 1], 5)         # duplicate left
            @test_throws ArgumentError Bisection([1], [2, 2])       # duplicate right
            @test_throws ArgumentError Bisection([0], 5)            # non-positive index
        end

        @testset "reshape consistency with tuple API" begin
            T = reshape(1.0:8.0, 2, 2, 2)
            result_bisection = reshape_tensor_for_bipartition(T, Bisection([1], 3))
            result_tuple     = reshape_tensor_for_bipartition(T, [(1,)])
            @test result_bisection == result_tuple
        end

        @testset "construction from IndexedTensor" begin
            σ  = PhysicalIndex(:σ, 2, 1, Contravariant)
            αL = BondIndex(:αL, 4, Covariant)
            αR = BondIndex(:αR, 4, Contravariant)
            A  = IndexedTensor(rand(2, 4, 4), (σ, αL, αR))

            # by index object: put σ on the left
            b1 = Bisection(A, [σ])
            @test b1.left  == [1]
            @test b1.right == [2, 3]

            # by index type: all PhysicalIndex legs on the left
            b2 = Bisection(A, PhysicalIndex)
            @test b2.left  == [1]
            @test b2.right == [2, 3]

            # by index type: all BondIndex legs on the left
            b3 = Bisection(A, BondIndex)
            @test b3.left  == [2, 3]
            @test b3.right == [1]

            # index not in tensor
            other = BondIndex(:β, 4, Contravariant)
            @test_throws ArgumentError Bisection(A, [other])

            # type not present in tensor
            @test_throws ArgumentError Bisection(IndexedTensor(rand(4, 4), (αL, αR)), PhysicalIndex)
        end
    end
end
