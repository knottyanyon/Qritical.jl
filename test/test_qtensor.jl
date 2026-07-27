@testset "QTensor — wrapping an array with named legs" begin

    # ── The numbers inside a QTensor stay exactly the same ───────────────────────

    @testset "The numbers stored inside don't change" begin
        @testset "2D matrix: values come back out unchanged" begin
            data = [1.0 2.0; 3.0 4.0]   # `[r1c1 r1c2; r2c1 r2c2]` = 2×2 matrix literal; Julia is column-major so this is stored as [1,3,2,4] in memory (Python/NumPy: row-major [1,2,3,4])
            i = upper(:i, 2)             # leg i: Upper variance, dim=2 (rows)
            j = lower(:j, 2)             # leg j: Lower variance, dim=2 (cols)
            t = QTensor(data, (i, j))    # `QTensor(data, indices)` = wrap array with named indices; `(i, j)` = 2-Tuple of AbstractIx
            @test t.data == data         # `.data` = field access to the backing array; `==` = element-wise equality
            @test t.data[1, 1] == 1.0   # 1-indexed (Julia); Python: `t.data[0, 0]`
            @test t.data[2, 2] == 4.0   # (2,2) = bottom-right element
        end

        @testset "3D tensor: data round-trips exactly" begin
            data = rand(Float64, 2, 3, 4)   # `rand(Float64, 2,3,4)` = 3D array of random Float64
            data_copy = copy(data)   # `copy(data)` = shallow copy of array; needed because QTensor shares memory with data
            indices = (upper(:i, 2), lower(:j, 3), upper(:k, 4))   # 3 legs, mixed variance
            t = QTensor(data, indices)
            @test t.data == data_copy   # data unchanged after wrapping
            @test t.data ≈ data_copy    # `≈` = `isapprox`. numerically close (same here since no operations performed)
        end

        @testset "complex element type: data round-trips exactly" begin
            data = [1.0+2.0im 3.0-4.0im; 5.0im 6.0]   # `im` = imaginary unit. ComplexF64 array
            i = upper(:α, 2)
            j = lower(:β, 2)
            t = QTensor(data, (i, j))
            @test t.data == data                  # exact equality for complex values
            @test eltype(t.data) == ComplexF64    # `eltype(arr)` = element type. should be ComplexF64
        end

        @testset "mutating .data after construction reflects in tensor" begin
            data = [1.0 2.0; 3.0 4.0]
            i = upper(:i, 2)
            j = lower(:j, 2)
            t = QTensor(data, (i, j))
            data[1, 1] = 99.0   # mutate the original array AFTER wrapping
            @test t.data[1, 1] == 99.0   # QTensor stores a reference, not a copy → mutation visible through t; Python: same behavior for NumPy arrays passed to a wrapping class
        end
    end

    # ── Number of legs matches number of array dimensions ────────────────────────

    @testset "Number of legs always equals number of array dimensions" begin
        @testset "1D array gets exactly 1 leg" begin
            data = [1.0, 2.0, 3.0]   # 1D vector
            i = lower(:i, 3)
            t = QTensor(data, (i,))   # `(i,)` = 1-element Tuple 
            @test length(t.indices) == 1   # `length(t.indices)` counts the legs; a 1D array has exactly 1
            @test ndims(t.data) == 1   # `ndims(arr)` = number of dimensions
            @test length(t.indices) == ndims(t.data)   # invariant: #legs == #dims always holds
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
            indices = (upper(:i, 2), lower(:j, 3), upper(:k, 4), lower(:ℓ, 5))   # `:ℓ` is a Unicode Symbol (Julia supports Unicode identifiers; Python does too but `ℓ` is unusual)
            t = QTensor(data, indices)
            @test length(t.indices) == 4
            @test ndims(t.data) == 4
        end

        @testset "a single number (0D scalar) gets zero legs" begin
            data = fill(42.0)   # `fill(x)` = 0-dimensional array containing x ; ndims=0
            t = QTensor(data, ())   # `()` = empty Tuple; zero legs for a 0D tensor (scalar)
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
            @test size(t) == (3, 2)    # `size(t)` = Python `t.shape`; returns (nrows, ncols) tuple
            @test size(t, 1) == 3      # `size(t, 1)` = Python `t.shape[0]`; size along dimension 1 (rows); 1-indexed!
            @test size(t, 2) == 2      # size along dimension 2 (cols)
        end

        @testset "Base.getindex — t[i, j] reads the correct element" begin
            data = [1.0 2.0 3.0; 4.0 5.0 6.0]   # 2×3 matrix
            i = upper(:i, 2)
            j = lower(:j, 3)
            t = QTensor(data, (i, j))
            @test t[1, 1] == 1.0   # `t[1,1]` = top-left (Julia 1-indexed); Python: `t[0,0]`; implemented via `Base.getindex` which delegates to `t.data[1,1]`
            @test t[1, 2] == 2.0   # row 1, col 2
            @test t[2, 3] == 6.0   # bottom-right: row 2, col 3
        end

        @testset "Base.setindex! — t[i, j] = v writes through to the backing array" begin
            data = zeros(2, 2)   # `zeros(2, 2)` = 2×2 matrix of 0.0 
            i = upper(:i, 2)
            j = lower(:j, 2)
            t = QTensor(data, (i, j))
            t[1, 1] = 99.0   # `t[1,1] = 99.0` calls `Base.setindex!(t, 99.0, 1, 1)` which writes to `t.data[1,1]`; Python: `t[0,0] = 99.0`
            @test t[1, 1] == 99.0     # value updated in the QTensor view
            @test data[1, 1] == 99.0  # ALSO updated in the original array (shared memory)
        end

        @testset "ndims() — derived from Base.size, counts the number of legs" begin
            @test ndims(QTensor(rand(2), (lower(:i, 2),))) == 1                                     # 1D
            @test ndims(QTensor(rand(2, 3), (upper(:i, 2), lower(:j, 3)))) == 2                    # 2D
            @test ndims(
                QTensor(rand(2, 3, 4), (upper(:i, 2), lower(:j, 3), upper(:k, 4)))
            ) == 3   # 3D; `ndims` is derived from `Base.size` (Julia provides it automatically for AbstractArray subtypes)
        end

        @testset "length() — derived from Base.size, counts the total number of elements" begin
            t2d = QTensor(rand(2, 3), (upper(:i, 2), lower(:j, 3)))
            @test length(t2d) == 6   # `length(t)` = total number of elements = product of sizes = 2*3=6 

            t3d = QTensor(rand(2, 3, 4), (upper(:i, 2), lower(:j, 3), upper(:k, 4)))
            @test length(t3d) == 24  # 2*3*4 = 24
        end
    end

    # ── Each leg's name, size, and direction are stored and retrievable ───────────

    @testset "Each leg keeps its name, size, and direction (upper/lower)" begin
        @testset "upper and lower directions are remembered correctly" begin
            i_upper = upper(:i, 2)   # Upper = domain = contravariant (incoming arrow)
            j_lower = lower(:j, 3)   # Lower = codomain = covariant (outgoing arrow)
            t = QTensor(rand(2, 3), (i_upper, j_lower))

            @test t.indices[1] == i_upper   # `t.indices` = Tuple of AbstractIx; [1] = first leg; should be i_upper
            @test t.indices[2] == j_lower   # second leg should be j_lower
            @test which_space(t.indices[1]) == :domain    # Upper index is in the domain (ket space, incoming)
            @test which_space(t.indices[2]) == :codomain  # Lower index is in the codomain (bra space, outgoing)
        end

        @testset "leg names (labels) are remembered even when directions are mixed" begin
            indices = (upper(:vL, 2), upper(:σ, 3), lower(:vR, 4))   # MPS tensor: vL (left bond), σ (physical), vR (right bond)
            t = QTensor(rand(2, 3, 4), indices)

            @test label(t.indices[1]) == :vL   # `label(ix)` = accessor returning Symbol label
            @test label(t.indices[2]) == :σ    # physical leg
            @test label(t.indices[3]) == :vR   # right bond leg
        end

        @testset "leg sizes (dimensions) are remembered correctly" begin
            indices = (upper(:i, 2), lower(:j, 3), upper(:k, 4))
            t = QTensor(rand(2, 3, 4), indices)

            @test dim(t.indices[1]) == 2   # `dim(ix)` = stored dimension field; must match array size
            @test dim(t.indices[2]) == 3
            @test dim(t.indices[3]) == 4
        end
    end

    # ── Boundary situations: very small tensors and invalid inputs ────────────────

    @testset "Unusual but valid sizes, and invalid inputs that should error" begin
        @testset "a single scalar value with no legs at all" begin
            data = fill(42.0)   # 0D array: `fill(x)` returns a 0-dimensional Array{Float64,0}
            t = QTensor(data, ())
            @test ndims(t.data) == 0    # 0 dimensions
            @test length(t.indices) == 0  # 0 legs
            @test t.data[] == 42.0      # `t.data[]` = index a 0D array with NO indices 
        end

        @testset "a complex scalar value with no legs keeps its element type" begin
            data = fill(1.0 + 2.0im)   # 0D complex array
            t = QTensor(data, ())
            @test eltype(t.data) == ComplexF64  # element type preserved
            @test t.data[] == 1.0 + 2.0im       # value preserved
        end

        @testset "a 1D tensor with only one element" begin
            data = [42.0]   # 1-element vector
            i = lower(:i, 1)   # dim=1 leg; boundary bond in OBC MPS has χ=1
            t = QTensor(data, (i,))
            @test size(t) == (1,)   # 1-element, 1D
            @test length(t) == 1
            @test t[1] == 42.0    # access the single element
        end

        @testset "a large tensor still preserves all data and leg info" begin
            data = rand(100, 50, 20)
            i = upper(:i, 100)
            j = lower(:j, 50)
            k = upper(:k, 20)
            t = QTensor(data, (i, j, k))
            @test size(t) == (100, 50, 20)
            @test length(t.indices) == 3
            @test dim(t.indices[1]) == 100   # large bond dimensions typical in MPS for high-entanglement states
        end
    end

    # ── Julia's type system tracks the element type and shape at compile time ──────

    @testset "Julia's type system records element type, number of legs, and backing array" begin
        @testset "Float64 elements in a 2D tensor are tracked in the type" begin
            t = QTensor(rand(Float64, 2, 3), (upper(:i, 2), lower(:j, 3)))
            @test typeof(t) <: QTensor{Float64,2,Array{Float64,2}}   # `typeof(t)` = Python `type(t)`; `<:` = subtype check; QTensor{Element,Valence,Data} — all type params known at compile time
        end

        @testset "ComplexF64 elements in a 3D tensor are tracked in the type" begin
            t = QTensor(
                rand(ComplexF64, 2, 3, 4), (upper(:i, 2), lower(:j, 3), upper(:k, 4))
            )
            @test typeof(t) <: QTensor{ComplexF64,3,Array{ComplexF64,3}}   # element=ComplexF64, valence=3, backing=Array{ComplexF64,3}
        end

        @testset "integer element type is also tracked correctly" begin
            t = QTensor(rand(Int, 2, 2), (upper(:i, 2), lower(:j, 2)))
            @test eltype(t) == Int   # `eltype(t)` = Julia builtin from AbstractArray; returns element type 
        end
    end

    # ── The legs are stored in a fixed-length tuple and can be accessed by position ─

    @testset "Legs are stored in an ordered tuple and can be accessed by position" begin
        @testset "t.indices is a plain Julia Tuple" begin
            t = QTensor(rand(2, 3), (upper(:i, 2), lower(:j, 3)))
            @test isa(t.indices, Tuple)            # `isa(x, T)` = Python `isinstance(x, T)`; indices stored as a Tuple (immutable, fixed-length)
            @test typeof(t.indices) <: Tuple       # Tuple is a parametric type; `<:` confirms it's some Tuple type
        end

        @testset "individual legs can be retrieved by index" begin
            i, j = upper(:i, 2), lower(:j, 3)   # `a, b = expr, expr` = parallel assignment 
            t = QTensor(rand(2, 3), (i, j))
            @test t.indices[1] == i   # 1-indexed Tuple access
            @test t.indices[2] == j
        end

        @testset "a fused (grouped) leg can also sit in the tuple" begin
            i = upper(:i, 2)
            σ = upper(:σ, 3)
            grouped = MulTIx(:iσ, (i, σ))   # `MulTIx(:label, (ix1, ix2))` = fused index with dim = dim(i)*dim(σ) = 2*3 = 6
            k = upper(:k, 4)
            t = QTensor(rand(6, 4), (grouped, k))   # first axis has dim=6 (matches grouped), second has dim=4 (matches k)

            @test t.indices[1] == grouped   # first leg is the fused index
            @test dim(t.indices[1]) == 6    # dim = product of constituent dims = 2*3 = 6
            @test t.indices[2] == k         # second leg is normal
        end
    end

    # ── QTensor and the original array share the same memory ─────────────────────

    @testset "QTensor shares memory with the array it was built from" begin
        @testset "writing to the tensor also changes the original array" begin
            data = [1.0 2.0; 3.0 4.0]
            i = upper(:i, 2)
            j = lower(:j, 2)
            t = QTensor(data, (i, j))

            t[1, 1] = 99.0   # write via QTensor setindex!
            @test data[1, 1] == 99.0   # reflected in original array; QTensor stores a reference (not a copy); Python: same behavior with NumPy views
        end

        @testset "writing to the original array also changes the tensor" begin
            data = [1.0 2.0; 3.0 4.0]
            i = upper(:i, 2)
            j = lower(:j, 2)
            t = QTensor(data, (i, j))

            data[2, 2] = 88.0   # write via original array
            @test t[2, 2] == 88.0   # reflected via tensor; same shared memory
        end

        @testset "two tensors built from separate arrays don't interfere" begin
            data1 = [1.0 2.0; 3.0 4.0]
            data2 = copy(data1)   # `copy` = independent copy; different memory address
            i = upper(:i, 2)
            j = lower(:j, 2)
            t1 = QTensor(data1, (i, j))
            t2 = QTensor(data2, (i, j))

            t1[1, 1] = 99.0           # modify t1's backing data
            @test t2[1, 1] == 1.0     # t2 is unaffected because data2 is a separate copy
        end
    end

    @testset "mismatched leg size and array size throws an error" begin
        @test_throws ArgumentError QTensor(rand(2, 3), (upper(:i, 5), lower(:j, 3)))   # leg i claims dim=5 but array axis 1 has size 2 → mismatch → ArgumentError
        @test_throws ArgumentError QTensor(rand(2, 3), (upper(:i, 2), lower(:j, 9)))   # leg j claims dim=9 but axis 2 has size 3
    end

    # ── Deferred: backend switch (TensorKit) ──────────────────────────────────────
    # The tests below are placeholders for the deferred backend-switch feature (1.2, second physics test).
    # Once `with_backend(:tensorkit) do … end` and `tensorkit_space(::TIx)` are implemented,
    # these tests will verify that an QTensor with TensorMap backing store:
    #   - carries the same index metadata
    #   - behaves identically under contractions
    #   - respects symmetry-graded spaces once enabled

    @testset "Not yet implemented: switching to a TensorKit-backed tensor [placeholder]" begin
        @test_broken false  # `@test_broken expr` = marks test as expected to fail; passes in CI as "broken" (green); becomes a real failure once the feature is accidentally implemented; Python: no exact equivalent — closest is `pytest.xfail`
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
        σ = upper(:σ, 2)   # physical leg, dim=2 (qubit)
        vL = upper(:vL, 3)   # left virtual bond, dim=3 (bond dimension χ)
        vR = lower(:vR, 4)   # right virtual bond, dim=4
        A = QTensor(rand(2, 3, 4), (σ, vL, vR))   # rank-3 MPS site tensor
        bp = Bipartition(Partition([σ, vL]), Partition([vR]))   # left={σ,vL} → rows; right={vR} → cols
        M = group_legs(A, bp)   # `group_legs(A, bp)` = fuse left legs and right legs; result is rank-2 QTensor
        @test size(M) == (6, 4)   # rows = dim(σ)*dim(vL) = 2*3 = 6; cols = dim(vR) = 4
    end

    @testset "result is a rank-2 QTensor with a MulTIx on each leg" begin
        σ = upper(:σ, 2)
        vL = upper(:vL, 3)
        vR = lower(:vR, 4)
        A = QTensor(rand(2, 3, 4), (σ, vL, vR))
        bp = Bipartition(Partition([σ, vL]), Partition([vR]))
        M = group_legs(A, bp)
        @test M isa QTensor             # `isa` = isinstance; result is still a QTensor
        @test ndims(M) == 2             # rank-2 (matrix)
        @test M.indices[1] isa MulTIx  # first leg is a fused MulTIx encapsulating σ and vL
        @test M.indices[2] isa MulTIx  # second leg is a fused MulTIx encapsulating vR (single-leg fuse)
    end

    @testset "left partition legs become rows, right partition legs become columns" begin
        σ = upper(:σ, 2)
        vL = upper(:vL, 3)
        vR = lower(:vR, 4)
        A = QTensor(rand(2, 3, 4), (σ, vL, vR))
        bp = Bipartition(Partition([σ, vL]), Partition([vR]))
        M = group_legs(A, bp)
        @test size(M, 1) == dim(σ) * dim(vL)   # rows = product of left partition dims; physics: these will become row indices of the Schmidt decomposition matrix
        @test size(M, 2) == dim(vR)             # cols = product of right partition dims
    end

    @testset "fused leg metadata records the constituent legs" begin
        σ = upper(:σ, 2)
        vL = upper(:vL, 3)
        vR = lower(:vR, 4)
        A = QTensor(rand(2, 3, 4), (σ, vL, vR))
        bp = Bipartition(Partition([σ, vL]), Partition([vR]))
        M = group_legs(A, bp)
        @test dim(M.indices[1]) == dim(σ) * dim(vL)     # fused left leg has combined dimension
        @test dim(M.indices[2]) == dim(vR)               # fused right leg retains its dimension
        @test M.indices[1].indices == (σ, vL)            # `.indices` on a MulTIx = Tuple of constituent legs; preserved for un-grouping
        @test M.indices[2].indices == (vR,)              # right partition wrapped in a 1-tuple
    end

    @testset "data round-trip: reshape back recovers the original values" begin
        σ = upper(:σ, 2)
        vL = upper(:vL, 3)
        vR = lower(:vR, 4)
        data = rand(2, 3, 4)
        A = QTensor(copy(data), (σ, vL, vR))
        bp = Bipartition(Partition([σ, vL]), Partition([vR]))
        M = group_legs(A, bp)
        # legs were already in (σ, vL, vR) order so no permutation needed
        recovered = reshape(M.data, dim(σ), dim(vL), dim(vR))   # `reshape(arr, d1, d2, d3)` = undo the fusion; Julia is column-major so reshape fills col-by-col; Python: `np.reshape(M.data, (dim_σ, dim_vL, dim_vR), order='F')` (Fortran order) to get column-major
        @test recovered ≈ data   # numerical comparison with tolerance; `≈` = `isapprox`
    end

    @testset "legs in non-natural order: tensor axes are permuted before reshape" begin
        σ = upper(:σ, 2)
        vL = upper(:vL, 3)
        vR = lower(:vR, 4)
        data = rand(2, 3, 4)   # tensor natural order: (σ=2, vL=3, vR=4)
        A = QTensor(copy(data), (σ, vL, vR))
        # put vR first in left, σ second — forces a permutation before reshape
        bp = Bipartition(Partition([vR, σ]), Partition([vL]))   # left={vR,σ}; vR is at axis 3 in A, σ at axis 1
        M = group_legs(A, bp)
        @test size(M) == (dim(vR) * dim(σ), dim(vL))   # (4*2) × 3 = 8 × 3; rows = product of left partition
        # vR is at axis 3, σ at axis 1, vL at axis 2  →  perm [3,1,2]
        expected = reshape(permutedims(data, [3, 1, 2]), dim(vR)*dim(σ), dim(vL))   # `permutedims(arr, perm)` = transpose generalization ; then reshape to 2D
        @test M.data ≈ expected
    end

    @testset "single-leg partitions on a 2D tensor: matrix is unchanged" begin
        α = upper(:α, 5)
        β = lower(:β, 3)
        data = rand(5, 3)
        A = QTensor(copy(data), (α, β))
        bp = Bipartition(Partition([α]), Partition([β]))   # both partitions have a single leg — no real fusion needed
        M = group_legs(A, bp)
        @test size(M) == (5, 3)   # shape unchanged
        @test M.data ≈ data       # values unchanged
    end

    @testset "empty left partition: result has exactly 1 row" begin
        α = upper(:α, 5)
        β = lower(:β, 3)
        A = QTensor(rand(5, 3), (α, β))
        bp = Bipartition(Partition([]), Partition([α, β]))   # empty left → dim(left fused) = 1 (empty product convention)
        M = group_legs(A, bp)
        @test size(M) == (1, 15)   # 1 row; 5*3=15 cols
    end

    @testset "empty right partition: result has exactly 1 column" begin
        α = upper(:α, 5)
        β = lower(:β, 3)
        A = QTensor(rand(5, 3), (α, β))
        bp = Bipartition(Partition([α, β]), Partition([]))   # empty right → dim(right fused) = 1
        M = group_legs(A, bp)
        @test size(M) == (15, 1)   # 5*3=15 rows; 1 col
    end

    @testset "a leg in the bipartition that is not in the tensor raises ArgumentError" begin
        σ = upper(:σ, 2)
        vR = lower(:vR, 4)
        ghost = lower(:ghost, 7)   # not a leg of A; "ghost" leg not part of this tensor
        A = QTensor(rand(2, 4), (σ, vR))
        bp = Bipartition(Partition([σ, ghost]), Partition([vR]))   # ghost is in bp.left but not in A
        @test_throws ArgumentError group_legs(A, bp)   # must throw: ghost not found in A.indices
    end

    @testset "a leg of the tensor not covered by the bipartition raises ArgumentError" begin
        σ = upper(:σ, 2)
        vL = upper(:vL, 3)
        vR = lower(:vR, 4)
        A = QTensor(rand(2, 3, 4), (σ, vL, vR))
        # vR is not in either partition — uncovered leg
        bp = Bipartition(Partition([σ]), Partition([vL]))   # vR is missing from both left and right
        @test_throws ArgumentError group_legs(A, bp)   # must throw: vR is an uncovered leg
    end
end

@testset "TensorOperations.tensorstructure — exposes leg metadata to @tensor contractions" begin
    i, j = upper(:i, 2), lower(:j, 3)   # parallel assignment; `upper/lower` return TIx{Upper/Lower}
    t = QTensor(rand(2, 3), (i, j))
    @test TensorOperations.tensorstructure(t, 1, false) == i   # `TensorOperations.tensorstructure(t, pos, conjA)` = hook called by @tensor to get the index at position `pos`; `false` = not conjugated; returns i
    @test TensorOperations.tensorstructure(t, 2, false) == j   # returns j for position 2
    @test TensorOperations.tensorstructure(t, 1, true) == i    # `conjA=true` means the tensor is conjugated; for a free tensor the structure is unchanged (adjoint affects data not metadata)
end

@testset "@tensor contraction — two QTensors contract correctly via shared index label" begin
    A = QTensor([1.0 2.0; 3.0 4.0], (upper(:i, 2), lower(:j, 2)))   # 2×2 matrix with named legs i (rows) and j (cols)
    B = QTensor([1.0; 0.0;;], (upper(:j, 2), lower(:k, 1)))          # `[1.0; 0.0;;]` = 2×1 column vector (trailing `;;` forces 2D); column vector with upper j and lower k
    @tensor C[i, k] := A[i, j] * B[j, k]   # `@tensor C[i,k] := A[i,j] * B[j,k]` = Einstein contraction: sum over shared index j ; `:=` creates a new array C
    @test C ≈ [1.0; 3.0;;]   # A[:,1] = [1;3]; B = [1;0]; C = A*B = [1*1+2*0; 3*1+4*0] = [1;3]; `;;` makes it 2D (2×1 matrix, not a vector)
end
