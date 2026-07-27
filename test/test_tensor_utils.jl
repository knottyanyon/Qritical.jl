const _autolabel = Qritical._autolabel   # `const` = module-level constant binding. `Qritical._autolabel` = access unexported private function by qualifying with module name

@testset "_autolabel: fusing multiple index labels" begin
    @testset "two constituents: label order matters" begin
        α = upper(:α, 3)   # Upper index with label :α
        σ = lower(:σ, 2)   # Lower index with label :σ
        @test Qritical._autolabel((α, σ)) == :ασ   # `(α, σ)` = 2-Tuple; `_autolabel` joins label strings: "α"*"σ" = "ασ" → Symbol :ασ
        @test Qritical._autolabel((σ, α)) == :σα   # order matters: reversed tuple gives :σα (not :ασ); important for reproducibility of MulTIx labels
    end

    @testset "single constituent: label passes through" begin
        α = upper(:α, 3)
        @test Qritical._autolabel((α,)) == :α   # `(α,)` = 1-element Tuple (trailing comma required in Julia too); single-element join = just the label itself
    end

    @testset "empty tuple returns :scalar" begin
        @test Qritical._autolabel(()) == :scalar   # empty tuple → no labels to join → special sentinel `:scalar`; physics: fusing zero legs = a scalar (0-dimensional) value
    end
end

@testset "MulTIx outer constructors (via _autolabel)" begin
    @testset "tuple constructor auto-generates label" begin
        α = upper(:α, 3)
        σ = lower(:σ, 2)
        g = MulTIx((α, σ))   # outer constructor: `MulTIx(indices::Tuple)` calls `MulTIx(_autolabel(indices), indices)` internally; no need to provide a label
        @test label(g) == :ασ   # auto-generated: "α"*"σ" = :ασ
        @test dim(g) == 6       # dim = prod of constituent dims = 3*2 = 6
    end

    @testset "varargs constructor is sugar for the tuple form" begin
        α = upper(:α, 3)
        σ = lower(:σ, 2)
        @test MulTIx(α, σ) == MulTIx((α, σ))   # `MulTIx(α, σ)` = varargs form; `α, σ` get collected into a Tuple `(α, σ)` via `...`; both produce the same MulTIx; Python: `*args` collecting
    end

    @testset "empty varargs yields a scalar-labelled MulTIx with dim 1" begin
        g = MulTIx()   # zero varargs → empty Tuple → _autolabel(()) = :scalar; dim = 1
        @test label(g) == :scalar   # label is the sentinel :scalar
        @test dim(g) == 1           # product of empty set = 1 (empty product convention)
    end
end

@testset "complement(p, A::QTensor): QTensor overload delegates to index tuple" begin
    vL = upper(:vL, 2)
    σ = upper(:σ, 3)
    vR = lower(:vR, 4)
    A = QTensor(rand(2, 3, 4), (vL, σ, vR))   # rank-3 MPS site tensor; legs = (vL, σ, vR)

    @testset "returns legs not in partition, preserving order" begin
        c = complement(Partition([vL, σ]), A)   # QTensor overload: extracts A.indices = (vL, σ, vR), then calls `complement(p, [vL, σ, vR])`; avoids writing `complement(p, A.indices)` by hand
        @test length(c) == 1   # only vR is not in [vL, σ]
        @test c[1] == vR       # the lone complement leg is vR
    end

    @testset "empty partition: all legs returned" begin
        c = complement(Partition([]), A)   # nothing excluded → returns all three legs
        @test c == [vL, σ, vR]   # original order preserved 
    end

    @testset "full partition: complement is empty" begin
        c = complement(Partition([vL, σ, vR]), A)   # all legs in partition → nothing left
        @test isempty(c)   # `isempty(c)` = Python `len(c) == 0`
    end
end

@testset "bipartition(left, A::QTensor): QTensor overload delegates to index tuple" begin
    vL = upper(:vL, 2)
    σ = upper(:σ, 3)
    vR = lower(:vR, 4)
    A = QTensor(rand(2, 3, 4), (vL, σ, vR))

    @testset "right side is complement of left" begin
        bp = bipartition(Partition([vL, σ]), A)   # QTensor overload: computes right = complement(left, A.indices) automatically
        @test bp.left == [vL, σ]   # left exactly as provided
        @test bp.right == [vR]      # right = complement of {vL,σ} in {vL,σ,vR} = {vR}
    end

    @testset "used together with group_legs" begin
        σ2 = upper(:σ, 2)
        vL2 = upper(:vL, 3)
        vR2 = lower(:vR, 4)
        data = rand(2, 3, 4)
        B = QTensor(copy(data), (σ2, vL2, vR2))
        bp = bipartition(Partition([σ2, vL2]), B)   # QTensor overload: right = complement([σ2,vL2], B) = [vR2]
        M = group_legs(B, bp)                       # `group_legs` reshapes B into a (dim(σ2)*dim(vL2)) × dim(vR2) = 6×4 matrix
        @test size(M) == (6, 4)
        @test reshape(M.data, dim(σ2), dim(vL2), dim(vR2)) ≈ data   # round-trip: reshape back to original; `≈` = isapprox with default tolerance
    end
end

@testset "bond_label: positional label generator" begin
    @test bond_label(:χ, 3) == :χ3    # `Symbol(:χ, 3)` = concatenate Symbol :χ with integer 3 → :χ3; used to name the virtual bond to the right of site 3 in an MPS chain
    @test bond_label(:α, 1) == :α1    # first bond
    @test bond_label(:χ, 12) == :χ12  # two-digit site index; Symbol concatenation handles integers of any width
    @test bond_label(:λ, 0) == :λ0    # site index 0; valid Symbol even though MPS sites are usually 1-indexed
end
