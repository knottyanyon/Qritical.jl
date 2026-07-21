@testset "QTensor — wrapping an array with named legs" begin

    # ── The numbers inside a QTensor stay exactly the same ───────────────────────

    @testset "The numbers stored inside don't change" begin
        @testset "2D matrix: values come back out unchanged" begin
            data = [1.0 2.0; 3.0 4.0]
            i = upper(:i, 2)
            j = lower(:j, 2)
            t = QTensor(data, (i, j))
            @test t.data == data
            @test t.data[1, 1] == 1.0
            @test t.data[2, 2] == 4.0
        end

        @testset "3D tensor: data round-trips exactly" begin
            data = rand(Float64, 2, 3, 4)
            data_copy = copy(data)
            indices = (upper(:i, 2), lower(:j, 3), upper(:k, 4))
            t = QTensor(data, indices)
            @test t.data == data_copy
            @test t.data ≈ data_copy
        end

        @testset "complex element type: data round-trips exactly" begin
            data = [1.0+2.0im 3.0-4.0im; 5.0im 6.0]
            i = upper(:α, 2)
            j = lower(:β, 2)
            t = QTensor(data, (i, j))
            @test t.data == data
            @test eltype(t.data) == ComplexF64
        end

        @testset "mutating .data after construction reflects in tensor" begin
            data = [1.0 2.0; 3.0 4.0]
            i = upper(:i, 2)
            j = lower(:j, 2)
            t = QTensor(data, (i, j))
            data[1, 1] = 99.0
            @test t.data[1, 1] == 99.0
        end
    end

    # ── Number of legs matches number of array dimensions ────────────────────────

    @testset "Number of legs always equals number of array dimensions" begin
        @testset "1D array gets exactly 1 leg" begin
            data = [1.0, 2.0, 3.0]
            i = lower(:i, 3)
            t = QTensor(data, (i,))
            @test length(t.indices) == 1
            @test ndims(t.data) == 1
            @test length(t.indices) == ndims(t.data)
        end

        @testset "2D matrix gets exactly 2 legs" begin
            data = rand(3, 2)
            i = upper(:i, 3)
            j = lower(:j, 2)
            t = QTensor(data, (i, j))
            @test length(t.indices) == 2
            @test ndims(t.data) == 2
        end

        @testset "3D array gets exactly 3 legs" begin
            data = rand(2, 3, 4)
            indices = (upper(:i, 2), lower(:j, 3), upper(:k, 4))
            t = QTensor(data, indices)
            @test length(t.indices) == 3
            @test ndims(t.data) == 3
        end

        @testset "4D array gets exactly 4 legs" begin
            data = rand(2, 3, 4, 5)
            indices = (upper(:i, 2), lower(:j, 3), upper(:k, 4), lower(:ℓ, 5))
            t = QTensor(data, indices)
            @test length(t.indices) == 4
            @test ndims(t.data) == 4
        end

        @testset "a single number (0D scalar) gets zero legs" begin
            data = fill(42.0)
            t = QTensor(data, ())
            @test length(t.indices) == 0
            @test ndims(t.data) == 0
        end
    end

    # ── Array behaviour inherited for free by subtyping AbstractArray ────────────
    # These tests cover Base.size / Base.getindex / Base.setindex! / Base.IndexStyle
    # and functions derived from them (ndims, length).  QTensor does not implement
    # any of this logic itself — it simply delegates to the backing array, and Julia
    # fills in everything else automatically.

    @testset "Inherited array behaviour (Base.size, getindex, setindex!, ndims, length)" begin
        @testset "Base.size — reports the correct shape" begin
            data = rand(Float64, 3, 2)
            i = upper(:i, 3)
            j = lower(:j, 2)
            t = QTensor(data, (i, j))
            @test size(t) == (3, 2)
            @test size(t, 1) == 3
            @test size(t, 2) == 2
        end

        @testset "Base.getindex — t[i, j] reads the correct element" begin
            data = [1.0 2.0 3.0; 4.0 5.0 6.0]
            i = upper(:i, 2)
            j = lower(:j, 3)
            t = QTensor(data, (i, j))
            @test t[1, 1] == 1.0
            @test t[1, 2] == 2.0
            @test t[2, 3] == 6.0
        end

        @testset "Base.setindex! — t[i, j] = v writes through to the backing array" begin
            data = zeros(2, 2)
            i = upper(:i, 2)
            j = lower(:j, 2)
            t = QTensor(data, (i, j))
            t[1, 1] = 99.0
            @test t[1, 1] == 99.0
            @test data[1, 1] == 99.0
        end

        @testset "ndims() — derived from Base.size, counts the number of legs" begin
            @test ndims(QTensor(rand(2), (lower(:i, 2),))) == 1
            @test ndims(QTensor(rand(2, 3), (upper(:i, 2), lower(:j, 3)))) == 2
            @test ndims(
                QTensor(rand(2, 3, 4), (upper(:i, 2), lower(:j, 3), upper(:k, 4)))
            ) == 3
        end

        @testset "length() — derived from Base.size, counts the total number of elements" begin
            t2d = QTensor(rand(2, 3), (upper(:i, 2), lower(:j, 3)))
            @test length(t2d) == 6

            t3d = QTensor(rand(2, 3, 4), (upper(:i, 2), lower(:j, 3), upper(:k, 4)))
            @test length(t3d) == 24
        end
    end

    # ── Each leg's name, size, and direction are stored and retrievable ───────────

    @testset "Each leg keeps its name, size, and direction (upper/lower)" begin
        @testset "upper and lower directions are remembered correctly" begin
            i_upper = upper(:i, 2)
            j_lower = lower(:j, 3)
            t = QTensor(rand(2, 3), (i_upper, j_lower))

            @test t.indices[1] == i_upper
            @test t.indices[2] == j_lower
            @test which_space(t.indices[1]) == :domain
            @test which_space(t.indices[2]) == :codomain
        end

        @testset "leg names (labels) are remembered even when directions are mixed" begin
            indices = (upper(:vL, 2), upper(:σ, 3), lower(:vR, 4))
            t = QTensor(rand(2, 3, 4), indices)

            @test label(t.indices[1]) == :vL
            @test label(t.indices[2]) == :σ
            @test label(t.indices[3]) == :vR
        end

        @testset "leg sizes (dimensions) are remembered correctly" begin
            indices = (upper(:i, 2), lower(:j, 3), upper(:k, 4))
            t = QTensor(rand(2, 3, 4), indices)

            @test dim(t.indices[1]) == 2
            @test dim(t.indices[2]) == 3
            @test dim(t.indices[3]) == 4
        end
    end

    # ── Boundary situations: very small tensors and invalid inputs ────────────────

    @testset "Unusual but valid sizes, and invalid inputs that should error" begin
        @testset "a single scalar value with no legs at all" begin
            data = fill(42.0)
            t = QTensor(data, ())
            @test ndims(t.data) == 0
            @test length(t.indices) == 0
            @test t.data[] == 42.0
        end

        @testset "a complex scalar value with no legs keeps its element type" begin
            data = fill(1.0 + 2.0im)
            t = QTensor(data, ())
            @test eltype(t.data) == ComplexF64
            @test t.data[] == 1.0 + 2.0im
        end

        @testset "a 1D tensor with only one element" begin
            data = [42.0]
            i = lower(:i, 1)
            t = QTensor(data, (i,))
            @test size(t) == (1,)
            @test length(t) == 1
            @test t[1] == 42.0
        end

        @testset "a large tensor still preserves all data and leg info" begin
            data = rand(100, 50, 20)
            i = upper(:i, 100)
            j = lower(:j, 50)
            k = upper(:k, 20)
            t = QTensor(data, (i, j, k))
            @test size(t) == (100, 50, 20)
            @test length(t.indices) == 3
            @test dim(t.indices[1]) == 100
        end
    end

    # ── Julia's type system tracks the element type and shape at compile time ──────

    @testset "Julia's type system records element type, number of legs, and backing array" begin
        @testset "Float64 elements in a 2D tensor are tracked in the type" begin
            t = QTensor(rand(Float64, 2, 3), (upper(:i, 2), lower(:j, 3)))
            @test typeof(t) <: QTensor{Float64,2,Array{Float64,2}}
        end

        @testset "ComplexF64 elements in a 3D tensor are tracked in the type" begin
            t = QTensor(
                rand(ComplexF64, 2, 3, 4), (upper(:i, 2), lower(:j, 3), upper(:k, 4))
            )
            @test typeof(t) <: QTensor{ComplexF64,3,Array{ComplexF64,3}}
        end

        @testset "integer element type is also tracked correctly" begin
            t = QTensor(rand(Int, 2, 2), (upper(:i, 2), lower(:j, 2)))
            @test eltype(t) == Int
        end
    end

    # ── The legs are stored in a fixed-length tuple and can be accessed by position ─

    @testset "Legs are stored in an ordered tuple and can be accessed by position" begin
        @testset "t.indices is a plain Julia Tuple" begin
            t = QTensor(rand(2, 3), (upper(:i, 2), lower(:j, 3)))
            @test isa(t.indices, Tuple)
            @test typeof(t.indices) <: Tuple
        end

        @testset "individual legs can be retrieved by index" begin
            i, j = upper(:i, 2), lower(:j, 3)
            t = QTensor(rand(2, 3), (i, j))
            @test t.indices[1] == i
            @test t.indices[2] == j
        end

        @testset "a fused (grouped) leg can also sit in the tuple" begin
            i = upper(:i, 2)
            σ = upper(:σ, 3)
            grouped = MulTIx(:iσ, (i, σ))
            k = upper(:k, 4)
            t = QTensor(rand(6, 4), (grouped, k))

            @test t.indices[1] == grouped
            @test dim(t.indices[1]) == 6
            @test t.indices[2] == k
        end
    end

    # ── QTensor and the original array share the same memory ─────────────────────

    @testset "QTensor shares memory with the array it was built from" begin
        @testset "writing to the tensor also changes the original array" begin
            data = [1.0 2.0; 3.0 4.0]
            i = upper(:i, 2)
            j = lower(:j, 2)
            t = QTensor(data, (i, j))

            t[1, 1] = 99.0
            @test data[1, 1] == 99.0
        end

        @testset "writing to the original array also changes the tensor" begin
            data = [1.0 2.0; 3.0 4.0]
            i = upper(:i, 2)
            j = lower(:j, 2)
            t = QTensor(data, (i, j))

            data[2, 2] = 88.0
            @test t[2, 2] == 88.0
        end

        @testset "two tensors built from separate arrays don't interfere" begin
            data1 = [1.0 2.0; 3.0 4.0]
            data2 = copy(data1)
            i = upper(:i, 2)
            j = lower(:j, 2)
            t1 = QTensor(data1, (i, j))
            t2 = QTensor(data2, (i, j))

            t1[1, 1] = 99.0
            @test t2[1, 1] == 1.0
        end
    end

    @testset "mismatched leg size and array size throws an error" begin
        @test_throws ArgumentError QTensor(rand(2, 3), (upper(:i, 5), lower(:j, 3)))
        @test_throws ArgumentError QTensor(rand(2, 3), (upper(:i, 2), lower(:j, 9)))
    end

    # ── Deferred: backend switch (TensorKit) ──────────────────────────────────────
    # The tests below are placeholders for the deferred backend-switch feature (1.2, second physics test).
    # Once `with_backend(:tensorkit) do … end` and `tensorkit_space(::TIx)` are implemented,
    # these tests will verify that an QTensor with TensorMap backing store:
    #   - carries the same index metadata
    #   - behaves identically under contractions
    #   - respects symmetry-graded spaces once enabled

    @testset "Not yet implemented: switching to a TensorKit-backed tensor [placeholder]" begin
        @test_broken false  # with_backend(:tensorkit) do … end produces TensorMap-backed tensor
        @test_broken false  # round-trip: TensorMap-backed tensor preserves metadata
        @test_broken false  # indices carry symmetry-graded ElementarySpace when enabled
    end
end


# ── group_legs ────────────────────────────────────────────────────────────────
# group_legs(A, bp) permutes and reshapes A into a rank-2 QTensor whose first
# axis collects the left-partition legs and whose second axis collects the right.
# This is the tensor-layer bridge to Bipartition (defined in the index layer).

@testset "group_legs: reshaping a tensor into a matrix along a bipartition" begin

    @testset "rank-3 tensor {σ, vL | vR}: output shape is (dim(σ)*dim(vL)) × dim(vR)" begin
        σ  = upper(:σ,  2)
        vL = upper(:vL, 3)
        vR = lower(:vR, 4)
        A  = QTensor(rand(2, 3, 4), (σ, vL, vR))
        bp = Bipartition(Partition([σ, vL]), Partition([vR]))
        M  = group_legs(A, bp)
        @test size(M) == (6, 4)
    end

    @testset "result is a rank-2 QTensor with a MulTIx on each leg" begin
        σ  = upper(:σ,  2)
        vL = upper(:vL, 3)
        vR = lower(:vR, 4)
        A  = QTensor(rand(2, 3, 4), (σ, vL, vR))
        bp = Bipartition(Partition([σ, vL]), Partition([vR]))
        M  = group_legs(A, bp)
        @test M isa QTensor
        @test ndims(M) == 2
        @test M.indices[1] isa MulTIx
        @test M.indices[2] isa MulTIx
    end

    @testset "left partition legs become rows, right partition legs become columns" begin
        σ  = upper(:σ,  2)
        vL = upper(:vL, 3)
        vR = lower(:vR, 4)
        A  = QTensor(rand(2, 3, 4), (σ, vL, vR))
        bp = Bipartition(Partition([σ, vL]), Partition([vR]))
        M  = group_legs(A, bp)
        @test size(M, 1) == dim(σ) * dim(vL)   # rows = left partition
        @test size(M, 2) == dim(vR)             # cols = right partition
    end

    @testset "fused leg metadata records the constituent legs" begin
        σ  = upper(:σ,  2)
        vL = upper(:vL, 3)
        vR = lower(:vR, 4)
        A  = QTensor(rand(2, 3, 4), (σ, vL, vR))
        bp = Bipartition(Partition([σ, vL]), Partition([vR]))
        M  = group_legs(A, bp)
        @test dim(M.indices[1]) == dim(σ) * dim(vL)
        @test dim(M.indices[2]) == dim(vR)
        @test M.indices[1].indices == (σ, vL)
        @test M.indices[2].indices == (vR,)
    end

    @testset "data round-trip: reshape back recovers the original values" begin
        σ  = upper(:σ,  2)
        vL = upper(:vL, 3)
        vR = lower(:vR, 4)
        data = rand(2, 3, 4)
        A  = QTensor(copy(data), (σ, vL, vR))
        bp = Bipartition(Partition([σ, vL]), Partition([vR]))
        M  = group_legs(A, bp)
        # legs were already in (σ, vL, vR) order so no permutation needed
        recovered = reshape(M.data, dim(σ), dim(vL), dim(vR))
        @test recovered ≈ data
    end

    @testset "legs in non-natural order: tensor axes are permuted before reshape" begin
        σ  = upper(:σ,  2)
        vL = upper(:vL, 3)
        vR = lower(:vR, 4)
        data = rand(2, 3, 4)   # tensor natural order: (σ=2, vL=3, vR=4)
        A  = QTensor(copy(data), (σ, vL, vR))
        # put vR first in left, σ second — forces a permutation
        bp = Bipartition(Partition([vR, σ]), Partition([vL]))
        M  = group_legs(A, bp)
        @test size(M) == (dim(vR) * dim(σ), dim(vL))   # (4*2) × 3
        # vR is at axis 3, σ at axis 1, vL at axis 2  →  perm [3,1,2]
        expected = reshape(permutedims(data, [3, 1, 2]), dim(vR)*dim(σ), dim(vL))
        @test M.data ≈ expected
    end

    @testset "single-leg partitions on a 2D tensor: matrix is unchanged" begin
        α = upper(:α, 5)
        β = lower(:β, 3)
        data = rand(5, 3)
        A  = QTensor(copy(data), (α, β))
        bp = Bipartition(Partition([α]), Partition([β]))
        M  = group_legs(A, bp)
        @test size(M) == (5, 3)
        @test M.data ≈ data
    end

    @testset "empty left partition: result has exactly 1 row" begin
        α = upper(:α, 5)
        β = lower(:β, 3)
        A  = QTensor(rand(5, 3), (α, β))
        bp = Bipartition(Partition([]), Partition([α, β]))
        M  = group_legs(A, bp)
        @test size(M) == (1, 15)
    end

    @testset "empty right partition: result has exactly 1 column" begin
        α = upper(:α, 5)
        β = lower(:β, 3)
        A  = QTensor(rand(5, 3), (α, β))
        bp = Bipartition(Partition([α, β]), Partition([]))
        M  = group_legs(A, bp)
        @test size(M) == (15, 1)
    end

    @testset "a leg in the bipartition that is not in the tensor raises ArgumentError" begin
        σ     = upper(:σ,  2)
        vR    = lower(:vR, 4)
        ghost = lower(:ghost, 7)   # not a leg of A
        A  = QTensor(rand(2, 4), (σ, vR))
        bp = Bipartition(Partition([σ, ghost]), Partition([vR]))
        @test_throws ArgumentError group_legs(A, bp)
    end

    @testset "a leg of the tensor not covered by the bipartition raises ArgumentError" begin
        σ  = upper(:σ,  2)
        vL = upper(:vL, 3)
        vR = lower(:vR, 4)
        A  = QTensor(rand(2, 3, 4), (σ, vL, vR))
        # vR is not in either partition
        bp = Bipartition(Partition([σ]), Partition([vL]))
        @test_throws ArgumentError group_legs(A, bp)
    end

end

@testset "TensorOperations.tensorstructure — exposes leg metadata to @tensor contractions" begin
    i, j = upper(:i, 2), lower(:j, 3)
    t = QTensor(rand(2, 3), (i, j))
    @test TensorOperations.tensorstructure(t, 1, false) == i
    @test TensorOperations.tensorstructure(t, 2, false) == j
    @test TensorOperations.tensorstructure(t, 1, true) == i   # conjA=true makes no difference
end

@testset "@tensor contraction — two QTensors contract correctly via shared index label" begin
    A = QTensor([1.0 2.0; 3.0 4.0], (upper(:i, 2), lower(:j, 2)))
    B = QTensor([1.0; 0.0;;], (upper(:j, 2), lower(:k, 1)))  # 2×1
    @tensor C[i, k] := A[i, j] * B[j, k]
    @test C ≈ [1.0; 3.0;;]
end