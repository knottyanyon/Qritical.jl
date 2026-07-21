const _autolabel = Qritical._autolabel

@testset "_autolabel: auto-generate MulTIx labels from constituent indices" begin
    @testset "concatenates labels of constituent indices" begin
        α = upper(:α, 3)
        σ = lower(:σ, 2)
        @test Qritical._autolabel((α, σ)) == :ασ
        @test Qritical._autolabel((σ, α)) == :σα
    end

    @testset "single constituent: label passes through" begin
        α = upper(:α, 3)
        @test Qritical._autolabel((α,)) == :α
    end

    @testset "empty tuple returns :scalar" begin
        @test Qritical._autolabel(()) == :scalar
    end
end

@testset "MulTIx outer constructors (via _autolabel)" begin
    @testset "tuple constructor auto-generates label" begin
        α = upper(:α, 3)
        σ = lower(:σ, 2)
        g = MulTIx((α, σ))
        @test label(g) == :ασ
        @test dim(g) == 6
    end

    @testset "varargs constructor is sugar for the tuple form" begin
        α = upper(:α, 3)
        σ = lower(:σ, 2)
        @test MulTIx(α, σ) == MulTIx((α, σ))
    end

    @testset "empty varargs yields a scalar-labelled MulTIx with dim 1" begin
        g = MulTIx()
        @test label(g) == :scalar
        @test dim(g) == 1
    end
end

@testset "complement(p, A::QTensor): QTensor overload delegates to index tuple" begin
    vL = upper(:vL, 2)
    σ  = upper(:σ,  3)
    vR = lower(:vR, 4)
    A  = QTensor(rand(2, 3, 4), (vL, σ, vR))

    @testset "returns legs not in partition, preserving order" begin
        c = complement(Partition([vL, σ]), A)
        @test length(c) == 1
        @test c[1] == vR
    end

    @testset "empty partition: all legs returned" begin
        c = complement(Partition([]), A)
        @test c == [vL, σ, vR]
    end

    @testset "full partition: complement is empty" begin
        c = complement(Partition([vL, σ, vR]), A)
        @test isempty(c)
    end
end

@testset "bipartition(left, A::QTensor): QTensor overload delegates to index tuple" begin
    vL = upper(:vL, 2)
    σ  = upper(:σ,  3)
    vR = lower(:vR, 4)
    A  = QTensor(rand(2, 3, 4), (vL, σ, vR))

    @testset "right side is complement of left" begin
        bp = bipartition(Partition([vL, σ]), A)
        @test bp.left  == [vL, σ]
        @test bp.right == [vR]
    end

    @testset "used together with group_legs" begin
        σ2  = upper(:σ,  2)
        vL2 = upper(:vL, 3)
        vR2 = lower(:vR, 4)
        data = rand(2, 3, 4)
        B  = QTensor(copy(data), (σ2, vL2, vR2))
        bp = bipartition(Partition([σ2, vL2]), B)
        M  = group_legs(B, bp)
        @test size(M) == (6, 4)
        @test reshape(M.data, dim(σ2), dim(vL2), dim(vR2)) ≈ data
    end
end

@testset "bond_label: positional label generator" begin
    @test bond_label(:χ, 3) == :χ3
    @test bond_label(:α, 1) == :α1
    @test bond_label(:χ, 12) == :χ12
    @test bond_label(:λ, 0) == :λ0
end
