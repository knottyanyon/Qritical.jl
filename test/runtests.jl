using Qritical
using Test
using Aqua

@testset "Qritical.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Qritical)
    end

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
    end
end
