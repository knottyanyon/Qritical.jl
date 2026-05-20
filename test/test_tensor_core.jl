using LinearAlgebra
using TensorOperations
using HalfIntegers: half

@testset "TensorCore" begin

    # ── IndexedTensor construction ────────────────────────────────────────────

    @testset "IndexedTensor construction" begin
        σ = PhysicalIndex(:σ, SpinSite(half(1), 1), UpIndex)
        α = BondIndex(:α, 1, 2, 4, DownIndex)   # bond 1→2, outgoing from site 1
        β = BondIndex(:β, 2, 3, 4, UpIndex)     # bond 2→3, incoming to site 2
        A = IndexedTensor(rand(2, 4, 4), (σ, α, β))

        @test size(A) == (2, 4, 4)
        @test ndims(A) == 3
        @test TensorOperations.tensorstructure(A, 1, false) === σ
        @test TensorOperations.tensorstructure(A, 2, false) === α
        @test TensorOperations.tensorstructure(A, 3, false) === β
    end

    @testset "IndexedTensor construction (explicit backend)" begin
        σ = PhysicalIndex(:σ, SpinSite(half(1), 1), UpIndex)
        α = BondIndex(:α, 1, 2, 4, DownIndex)
        A = IndexedTensor(rand(2, 4), (σ, α); backend=:native)
        @test A.data isa Array
    end

    # ── TensorOperations interface ────────────────────────────────────────────

    @testset "checkcontractible" begin
        α = BondIndex(:α, 2, 3, 4, UpIndex)
        @test_nowarn TensorOperations.checkcontractible(α, dual(α), false, false, :α)

        # same direction
        @test_throws ArgumentError TensorOperations.checkcontractible(
            α, α, false, false, :α
        )

        # different kinds
        σ = PhysicalIndex(:σ, SpinSite(half(3), 1), DownIndex)   # spin-3/2: dim 4
        @test_throws ArgumentError TensorOperations.checkcontractible(
            α, σ, false, false, :x
        )

        # label mismatch (same endpoints, same dim, opposite dir)
        @test_throws ArgumentError TensorOperations.checkcontractible(
            α, BondIndex(:β, 2, 3, 4, DownIndex), false, false, :x
        )

        # dimension mismatch (same label, same endpoints, opposite dir)
        @test_throws DimensionMismatch TensorOperations.checkcontractible(
            α, BondIndex(:α, 2, 3, 3, DownIndex), false, false, :x
        )
    end

    @testset "@tensor contraction" begin
        α = BondIndex(:α, 2, 3, 4, UpIndex)
        β = BondIndex(:β, 1, 2, 3, UpIndex)
        σ = PhysicalIndex(:σ, SpinSite(half(1), 1), UpIndex)
        A = IndexedTensor(rand(2, 4, 3), (σ, dual(α), β))
        B = IndexedTensor(rand(4, 5), (α, BondIndex(:γ, 3, 4, 5, UpIndex)))
        @tensor C[s, b, g] := A[s, a, b] * B[a, g]
        @test size(C) == (2, 3, 5)
    end

    # ── kronecker_delta ───────────────────────────────────────────────────────

    @testset "kronecker_delta" begin
        α = BondIndex(:α, 2, 3, 4, UpIndex)
        δ = kronecker_delta(α, dual(α))
        @test size(δ.data) == (4, 4)
        @test δ.data ≈ Matrix(I, 4, 4)
        @test δ.indices == (α, dual(α))

        # not a dual pair
        @test_throws ArgumentError kronecker_delta(α, α)

        # different kinds
        σ = PhysicalIndex(:σ, SpinSite(half(3), 1), DownIndex)   # spin-3/2: dim 4
        @test_throws ArgumentError kronecker_delta(α, σ)
    end

    # ── Bisection ─────────────────────────────────────────────────────────────

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
            @test_throws ArgumentError Bisection([1], [1, 2])    # overlap
            @test_throws ArgumentError Bisection([1, 1], 5)      # duplicate left
            @test_throws ArgumentError Bisection([1], [2, 2])    # duplicate right
            @test_throws ArgumentError Bisection([0], 5)         # non-positive index
        end

        @testset "reshape consistency with tuple API" begin
            T = reshape(1.0:8.0, 2, 2, 2)
            result_bisection = reshape_tensor_for_bipartition(T, Bisection([1], 3))
            result_tuple = reshape_tensor_for_bipartition(T, [(1,)])
            @test result_bisection == result_tuple
        end

        @testset "construction from IndexedTensor" begin
            σ = PhysicalIndex(:σ, SpinSite(half(1), 1), UpIndex)
            αL = BondIndex(:αL, 1,2, 4, DownIndex)   # bond 1→2, outgoing from site 1
            αR = BondIndex(:αR, 2, 3, 4, UpIndex)     # bond 2→3, incoming to site 2
            A = IndexedTensor(rand(2, 4, 4), (σ, αL, αR))

            b1 = Bisection(A, [σ])
            @test b1.left == [1]
            @test b1.right == [2, 3]

            b2 = Bisection(A, PhysicalIndex)
            @test b2.left == [1]
            @test b2.right == [2, 3]

            b3 = Bisection(A, BondIndex)
            @test b3.left == [2, 3]
            @test b3.right == [1]

            other = BondIndex(:β, 1, 2, 4, UpIndex)
            @test_throws ArgumentError Bisection(A, [other])

            @test_throws ArgumentError Bisection(
                IndexedTensor(rand(4, 4), (αL, αR)), PhysicalIndex
            )
        end
    end

end
