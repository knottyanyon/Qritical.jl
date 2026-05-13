using LinearAlgebra
using TensorOperations
using TensorKit: TensorKit
using HalfIntegers: half

@testset "TensorIndex" begin

    # ── Direction enum: UpIndex / DownIndex ────────────────────────────

    @testset "IndexDirection enum" begin
        @test UpIndex != DownIndex
        @test flip(UpIndex) == DownIndex
        @test flip(DownIndex) == UpIndex
    end

    # ── Index structs: PhysicalIndex, BondIndex ──────────────────────────────

    @testset "BondIndex" begin
        α = BondIndex(:α, 4, UpIndex)
        @test α.label == :α
        @test α.dim == 4
        @test α.dir == UpIndex
        @test is_bond(α)
        @test !is_physical(α)
    end

    # ── Step 4: PhysicalIndex with AbstractSite + local_hilbert_dim ─────────

    @testset "PhysicalIndex with site" begin
        σ = PhysicalIndex(:σ, SpinSite(half(1), 1), UpIndex)
        @test σ.label == :σ
        @test σ.site === SpinSite(half(1), 1)
        @test σ.site.lattice_ordinal == 1
        @test σ.dir == UpIndex
        @test is_physical(σ)
        @test !is_bond(σ)
    end

    @testset "local_hilbert_dim on indices" begin
        @test local_hilbert_dim(PhysicalIndex(:σ, SpinSite(half(1), 1), UpIndex)) == 2
        @test local_hilbert_dim(PhysicalIndex(:σ, SpinSite(half(2), 1), DownIndex)) == 3
        @test local_hilbert_dim(PhysicalIndex(:n, SpinlessBosonicSite(1; n_max_occ=3), DownIndex)) == 4
        @test local_hilbert_dim(BondIndex(:α, 16, DownIndex)) == 16
    end

    @testset "dual preserves site" begin
        σ = PhysicalIndex(:σ, SpinSite(half(1), 3), UpIndex)
        @test dual(σ).dir == DownIndex
        @test dual(σ).site === σ.site
        @test dual(dual(σ)) == σ
    end

    # ── Index algebra: dual, adjoint, direction helpers ──────────────────────

    @testset "dual and adjoint" begin
        α = BondIndex(:α, 4, UpIndex)
        @test dual(α).dir == DownIndex
        @test dual(α).label == α.label
        @test dual(α).dim == α.dim
        @test α' == dual(α)
        @test dual(dual(α)) == α

        σ = PhysicalIndex(:σ, SpinSite(half(1), 1), DownIndex)
        @test dual(σ).dir == UpIndex
        @test dual(σ).site === σ.site
        @test local_hilbert_dim(dual(σ)) == local_hilbert_dim(σ)
    end

    @testset "isdual" begin
        α = BondIndex(:α, 4, UpIndex)
        @test isdual(α, dual(α))
        @test !isdual(α, α)
        @test !isdual(α, BondIndex(:α, 4, UpIndex))  # same direction
        @test !isdual(α, BondIndex(:α, 3, DownIndex))      # different dim
        σ = PhysicalIndex(:σ, SpinSite(half(3), 1), DownIndex)   # spin-3/2: dim 4
        @test !isdual(α, σ)                                 # different kind
    end

    @testset "as_up and as_down" begin
        α = BondIndex(:α, 4, UpIndex)
        @test as_down(α).dir == DownIndex
        @test as_down(dual(α)) == dual(α)   # idempotent
        @test as_up(α) == α                 # idempotent
        @test as_up(dual(α)).dir == UpIndex
    end

    # ── Tensor layer: IndexedTensor and TensorOperations ─────────────────────

    @testset "IndexedTensor construction" begin
        σ = PhysicalIndex(:σ, SpinSite(half(1), 1), UpIndex)
        α = BondIndex(:α, 4, DownIndex)
        β = BondIndex(:β, 4, UpIndex)
        A = IndexedTensor(rand(2, 4, 4), (σ, α, β))

        @test size(A) == (2, 4, 4)
        @test ndims(A) == 3
        @test TensorOperations.tensorstructure(A, 1, false) === σ
        @test TensorOperations.tensorstructure(A, 2, false) === α
        @test TensorOperations.tensorstructure(A, 3, false) === β
    end

    @testset "checkcontractible" begin
        α = BondIndex(:α, 4, UpIndex)
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

        # dimension mismatch
        γ = BondIndex(:γ, 3, DownIndex)
        @test_throws DimensionMismatch TensorOperations.checkcontractible(
            α, γ, false, false, :x
        )
    end

    @testset "@tensor contraction" begin
        α = BondIndex(:α, 4, UpIndex)
        β = BondIndex(:β, 3, UpIndex)
        σ = PhysicalIndex(:σ, SpinSite(half(1), 1), UpIndex)
        A = IndexedTensor(rand(2, 4, 3), (σ, dual(α), β))
        B = IndexedTensor(rand(4, 5), (α, BondIndex(:γ, 5, UpIndex)))
        @tensor C[s, b, g] := A[s, a, b] * B[a, g]
        @test size(C) == (2, 3, 5)
    end

    # ── Tensor operations: Kronecker delta ───────────────────────────────────

    @testset "kronecker_delta" begin
        α = BondIndex(:α, 4, UpIndex)
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
            αL = BondIndex(:αL, 4, DownIndex)
            αR = BondIndex(:αR, 4, UpIndex)
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

            other = BondIndex(:β, 4, UpIndex)
            @test_throws ArgumentError Bisection(A, [other])

            @test_throws ArgumentError Bisection(
                IndexedTensor(rand(4, 4), (αL, αR)), PhysicalIndex
            )
        end
    end
end
