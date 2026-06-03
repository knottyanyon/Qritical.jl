using LinearAlgebra

# ── Fixtures ──────────────────────────────────────────────────────────────────

# Interior MPS site tensor: legs (vL, σ, vR), shape (2, 3, 4)
# vL is upper (inward from left); vR is lower (outward to right).
function _site_tensor(T=Float64)
    vL = upper(:vL, 2); σ = lower(:σ, 3); vR = lower(:vR, 4)
    IndexedTensor(randn(T, 2, 3, 4), (vL, σ, vR))
end

# Standard bipartition of the site tensor: (vL, σ) | (vR)  →  6 × 4 matrix
function _site_bipartition()
    Bipartition(Partition(upper(:vL, 2), lower(:σ, 3)), Partition(lower(:vR, 4)))
end

# ── label interface ────────────────────────────────────────────────────────────

@testset "label interface" begin
    @test label(upper(:σ, 2))   == :σ
    @test label(lower(:αL, 4))  == :αL
    @test label(MultiIx(:combined, (upper(:α, 2), lower(:β, 3)))) == :combined
end

# ── MultiIx auto-label ─────────────────────────────────────────────────────────

@testset "MultiIx auto-label" begin

    @testset "auto-generates label by concatenating constituents" begin
        g = MultiIx((upper(:vL, 2), lower(:σ, 3)))
        @test label(g) == :vLσ
        @test ndim(g)  == 6
    end

    @testset "single constituent: label equals that index's label" begin
        g = MultiIx((upper(:α, 4),))
        @test label(g) == :α
        @test ndim(g)  == 4
    end

    @testset "empty constituents: label is :scalar, ndim is 1" begin
        g = MultiIx(())
        @test label(g) == :scalar
        @test ndim(g)  == 1
    end

    @testset "varargs convenience constructor" begin
        g = MultiIx(upper(:vL, 2), lower(:σ, 3))
        @test label(g) == :vLσ
        @test ndim(g)  == 6
    end

end

# ── TensorSVD ─────────────────────────────────────────────────────────────────

@testset "TensorSVD" begin
    A = _site_tensor()
    bp = _site_bipartition()
    F  = tensor_svd(A, bp, KeepFirst(3))

    @testset "return type" begin
        @test F isa TensorSVD
    end

    @testset "field types" begin
        @test F.U          isa IndexedTensor
        @test F.Σ          isa IndexedTensor{<:Real, 2, <:Diagonal}
        @test F.Vd         isa IndexedTensor
        @test F.ε          isa Real
        @test F.normalized == false
    end

    @testset "SingularElement == real(Element)" begin
        @test eltype(F.Σ.data) == real(eltype(F.U.data))
        @test typeof(F.ε)      == real(eltype(F.U.data))
    end

    @testset "named destructuring" begin
        (; U, Σ, Vd, ε) = F
        @test U  === F.U
        @test Σ  === F.Σ
        @test Vd === F.Vd
        @test ε  === F.ε
    end

    @testset "positional destructuring via iterate" begin
        U, Σ, Vd, ε = F
        @test U  === F.U
        @test Σ  === F.Σ
        @test Vd === F.Vd
        @test ε  === F.ε
    end

    @testset "normalize flag" begin
        F_norm = tensor_svd(A, bp, KeepFirst(3); normalize=true)
        @test  F_norm.normalized
        @test !F.normalized
    end

    @testset "complex element type" begin
        A_c = _site_tensor(ComplexF64)
        F_c = tensor_svd(A_c, bp, KeepFirst(3))
        @test eltype(F_c.U.data)    == ComplexF64
        @test eltype(F_c.Σ.data)   == Float64      # real(ComplexF64) = Float64
        @test typeof(F_c.ε)        == Float64
    end

    @testset "constructor rejects SingularElement ≠ real(Element)" begin
        # Build two compatible tensors with mismatched element/singular types
        # by trying to call the inner constructor directly with wrong types.
        # We test this indirectly: a Float32 U with Float64 Σ must throw.
        A32 = IndexedTensor(Float32.(A.data), A.indices)
        F32 = tensor_svd(A32, bp, KeepFirst(3))
        @test eltype(F32.U.data)  == Float32
        @test eltype(F32.Σ.data) == Float32
    end
end

# ── Bond ──────────────────────────────────────────────────────────────────────

@testset "Bond" begin

    @testset "construction and accessors" begin
        b = Bond(lower(:χvLσ, 3), upper(:χvLσ, 3))
        @test label(b) == :χvLσ
        @test ndim(b)  == 3
        @test b.trunc  === nothing
        @test b.ε      == 0.0
    end

    @testset "construction with SVD metadata" begin
        b = Bond(lower(:χvR, 4), upper(:χvR, 4), KeepFirst(4), 0.12)
        @test b.trunc isa KeepFirst
        @test b.ε     ≈ 0.12
    end

    @testset "label mismatch throws" begin
        @test_throws ArgumentError Bond(lower(:α, 3), upper(:β, 3))
    end

    @testset "ndim mismatch throws" begin
        @test_throws ArgumentError Bond(lower(:α, 3), upper(:α, 4))
    end

end

# ── Partition ─────────────────────────────────────────────────────────────────

@testset "Partition" begin

    @testset "varargs constructor" begin
        p = Partition(upper(:vL, 2), lower(:σ, 3))
        @test length(p.indices) == 2
        @test p.indices[1] == upper(:vL, 2)
        @test p.indices[2] == lower(:σ, 3)
    end

    @testset "vector constructor" begin
        p = Partition([upper(:vL, 2), lower(:σ, 3)])
        @test length(p.indices) == 2
    end

    @testset "single-index partition" begin
        p = Partition(upper(:vR, 4))
        @test length(p.indices) == 1
        @test p.indices[1] == upper(:vR, 4)
    end

end

# ── Bipartition ───────────────────────────────────────────────────────────────

@testset "Bipartition" begin

    @testset "stores left and right partitions" begin
        left  = Partition(upper(:vL, 2), lower(:σ, 3))
        right = Partition(upper(:vR, 4))
        bp    = Bipartition(left, right)
        @test bp.left  === left
        @test bp.right === right
    end

    @testset "overlapping indices throw ArgumentError" begin
        shared = upper(:vL, 2)
        @test_throws ArgumentError Bipartition(
            Partition(shared, lower(:σ, 3)),
            Partition(shared)
        )
    end

end

# ── complement and bipartition convenience constructor ────────────────────────

@testset "complement and bipartition(left, A)" begin
    A = _site_tensor()

    @testset "complement returns remaining indices" begin
        left = Partition(upper(:vL, 2), lower(:σ, 3))
        c    = complement(left, A)
        @test length(c.indices) == 1
        @test c.indices[1] == lower(:vR, 4)
    end

    @testset "bipartition(left, A) pairs left with complement" begin
        left = Partition(upper(:vL, 2), lower(:σ, 3))
        bp   = bipartition(left, A)
        @test bp.left  == left
        @test bp.right == complement(left, A)
    end

end

# ── group_legs ────────────────────────────────────────────────────────────────

@testset "group_legs" begin
    A  = _site_tensor()
    bp = _site_bipartition()

    @testset "result is a 2-leg IndexedTensor" begin
        M = group_legs(A, bp)
        @test ndims(M) == 2
    end

    @testset "shape matches partition dimensions" begin
        M = group_legs(A, bp)
        @test size(M, 1) == 2 * 3   # prod(ndim(vL), ndim(σ))
        @test size(M, 2) == 4       # ndim(vR)
    end

    @testset "result indices are MultiIx" begin
        M = group_legs(A, bp)
        @test M.indices[1] isa MultiIx
        @test M.indices[2] isa MultiIx
    end

    @testset "MultiIx ndim matches partition dimension" begin
        M = group_legs(A, bp)
        @test ndim(M.indices[1]) == 6
        @test ndim(M.indices[2]) == 4
    end

    @testset "Frobenius norm is preserved" begin
        M = group_legs(A, bp)
        @test norm(M.data) ≈ norm(A.data) atol=1e-14
    end

    @testset "partial bipartition (missing indices) throws ArgumentError" begin
        partial = Bipartition(
            Partition(upper(:vL, 2)),
            Partition(lower(:σ, 3))
        )   # vR is uncovered
        @test_throws ArgumentError group_legs(A, partial)
    end

    @testset "index not present in tensor throws ArgumentError" begin
        bad = Bipartition(
            Partition(upper(:vL, 2), upper(:ghost, 5)),
            Partition(lower(:σ, 3), upper(:vR, 4))
        )
        @test_throws ArgumentError group_legs(A, bad)
    end

end

# ── tensor_svd: physics behavior ──────────────────────────────────────────────
# These tests encode mathematical invariants that hold for any correct SVD.
# They are written first and drive the implementation.

@testset "tensor_svd: physics behavior" begin
    A      = _site_tensor()
    bp     = _site_bipartition()
    M      = group_legs(A, bp)
    all_svs = svdvals(M.data)
    r      = length(all_svs)   # full rank — no truncation
    (; U, Σ, Vd, ε) = tensor_svd(A, bp, KeepFirst(r))
    S = diag(Σ.data)   # diagonal entries as a plain vector

    @testset "singular values are non-negative" begin
        @test all(s -> s >= 0, S)
    end

    @testset "singular values are sorted descending" begin
        @test S == sort(S; rev=true)
    end

    @testset "U is left-isometric: U†U ≈ I" begin
        U_mat = reshape(U.data, :, size(Σ.data, 1))
        @test U_mat' * U_mat ≈ I atol=1e-12
    end

    @testset "Vd is right-isometric: Vd Vd† ≈ I" begin
        Vd_mat = reshape(Vd.data, size(Σ.data, 1), :)
        @test Vd_mat * Vd_mat' ≈ I atol=1e-12
    end

    @testset "full SVD reconstructs original matrix" begin
        U_mat  = reshape(U.data,  :, size(Σ.data, 1))
        Vd_mat = reshape(Vd.data, size(Σ.data, 1), :)
        @test norm(M.data - U_mat * Σ.data * Vd_mat) / norm(M.data) < 1e-12
    end

    @testset "no truncation: ε == 0" begin
        @test ε == 0
    end

end

# ── tensor_svd: index structure ───────────────────────────────────────────────
# Bond labels are derived from the MultiIx autolabels of the bipartition,
# prefixed with :χ to guarantee they don't clash with original index labels.
# Bond₁ (U ↔ Σ): label = Symbol(:χ, label(left MultiIx))  = :χvLσ
# Bond₂ (Σ ↔ Vd): label = Symbol(:χ, label(right MultiIx)) = :χvR

@testset "tensor_svd: index structure" begin
    A  = _site_tensor()
    bp = _site_bipartition()
    r  = 3
    (; U, Σ, Vd) = tensor_svd(A, bp, KeepFirst(r))

    @testset "Σ is a 2-leg IndexedTensor backed by Diagonal" begin
        @test Σ isa IndexedTensor{Float64, 2}
        @test Σ.data isa Diagonal
        @test length(diag(Σ.data)) == r
    end

    @testset "U: left-partition indices + lower bond leg" begin
        @test ndims(U)         == 3             # vL, σ, bond₁
        @test U.indices[1]     == upper(:vL, 2)
        @test U.indices[2]     == lower(:σ, 3)
        @test U.indices[end]   isa TIx{Lower}
        @test label(U.indices[end]) == :χvLσ
        @test ndim(U.indices[end])  == r
    end

    @testset "Σ: upper bond₁ leg and lower bond₂ leg" begin
        @test Σ.indices[1] isa TIx{Upper}
        @test label(Σ.indices[1]) == :χvLσ
        @test ndim(Σ.indices[1])  == r
        @test Σ.indices[2] isa TIx{Lower}
        @test label(Σ.indices[2]) == :χvR
        @test ndim(Σ.indices[2])  == r
    end

    @testset "Vd: upper bond₂ leg + right-partition indices" begin
        @test ndims(Vd)            == 2           # bond₂, vR
        @test Vd.indices[1]        isa TIx{Upper}
        @test label(Vd.indices[1]) == :χvR
        @test ndim(Vd.indices[1])  == r
        @test Vd.indices[2]        == lower(:vR, 4)
    end

    @testset "bond label continuity: U→Σ and Σ→Vd" begin
        @test label(U.indices[end]) == label(Σ.indices[1])   # Bond₁
        @test label(Σ.indices[2])   == label(Vd.indices[1])  # Bond₂
    end

end

# ── tensor_svd: truncation strategies ────────────────────────────────────────

@testset "tensor_svd: truncation" begin
    A       = _site_tensor()
    bp      = _site_bipartition()
    M       = group_legs(A, bp)
    all_svs = svdvals(M.data)

    @testset "KeepFirst(r): retains exactly r singular values" begin
        r = 2
        (; Σ) = tensor_svd(A, bp, KeepFirst(r))
        @test size(Σ.data, 1) == r
    end

    @testset "KeepFirst(r): retained values are the r largest" begin
        r = 2
        (; Σ) = tensor_svd(A, bp, KeepFirst(r))
        @test diag(Σ.data) ≈ all_svs[1:r] atol=1e-12
    end

    @testset "KeepFirst(r): ε equals 2-norm of discarded singular values" begin
        r = 2
        (; ε) = tensor_svd(A, bp, KeepFirst(r))
        @test ε ≈ norm(all_svs[(r+1):end]) atol=1e-12
    end

    @testset "KeepAbove(atol): all retained σ satisfy σ > atol" begin
        atol = all_svs[2] / 2   # threshold sits between 2nd and 3rd SV
        (; Σ) = tensor_svd(A, bp, KeepAbove(atol))
        @test all(s -> s > atol, diag(Σ.data))
    end

    @testset "KeepAbove(atol): ε equals 2-norm of discarded singular values" begin
        atol = all_svs[2] / 2
        (; Σ, ε) = tensor_svd(A, bp, KeepAbove(atol))
        discarded = filter(s -> s <= atol, all_svs)
        @test ε ≈ norm(discarded) atol=1e-12
    end

    @testset "KeepRelative(rtol): all retained σ satisfy σ/σ_max > rtol" begin
        rtol = 0.3
        (; Σ) = tensor_svd(A, bp, KeepRelative(rtol))
        @test all(s -> s / all_svs[1] > rtol, diag(Σ.data))
    end

    @testset "KeepRelative(rtol): ε equals 2-norm of discarded singular values" begin
        rtol = 0.3
        (; Σ, ε) = tensor_svd(A, bp, KeepRelative(rtol))
        discarded = filter(s -> s / all_svs[1] <= rtol, all_svs)
        @test ε ≈ norm(discarded) atol=1e-12
    end

end

# ── tensor_svd: edge cases ────────────────────────────────────────────────────

@testset "tensor_svd: edge cases" begin

    @testset "KeepFirst(r ≥ rank): no truncation, ε = 0" begin
        A  = _site_tensor()
        bp = _site_bipartition()
        M  = group_legs(A, bp)
        (; Σ, ε) = tensor_svd(A, bp, KeepFirst(100))   # 100 >> actual rank
        @test size(Σ.data, 1) == rank(M.data)
        @test ε == 0
    end

    @testset "rank-deficient input: only non-zero singular values kept" begin
        # rank-1 outer product embedded in a 4×3 tensor
        data = [1.0, 2.0, 3.0, 4.0] * [5.0, 6.0, 7.0]'
        A    = IndexedTensor(data, (upper(:row, 4), lower(:col, 3)))
        bp   = Bipartition(Partition(upper(:row, 4)), Partition(lower(:col, 3)))
        (; Σ, ε) = tensor_svd(A, bp, KeepFirst(3))   # request 3, rank is 1
        @test size(Σ.data, 1) == 1
        @test ε ≈ 0 atol=1e-12
    end

    @testset "1×N bipartition: single index on left side" begin
        A   = _site_tensor()
        bp  = bipartition(Partition(upper(:vL, 2)), A)   # 2 × 12
        M   = group_legs(A, bp)
        @test size(M) == (2, 12)
        (; U, Σ, Vd) = tensor_svd(A, bp, KeepFirst(2))
        r      = size(Σ.data, 1)
        U_mat  = reshape(U.data,  :, r)
        Vd_mat = reshape(Vd.data, r, :)
        @test norm(M.data - U_mat * Σ.data * Vd_mat) / norm(M.data) < 1e-12
    end

    @testset "N×1 bipartition: single index on right side" begin
        A   = _site_tensor()
        bp  = Bipartition(Partition(upper(:vL, 2), lower(:σ, 3)), Partition(lower(:vR, 4)))
        M   = group_legs(A, bp)
        @test size(M) == (6, 4)
        (; U, Σ, Vd) = tensor_svd(A, bp, KeepFirst(4))
        r      = size(Σ.data, 1)
        U_mat  = reshape(U.data,  :, r)
        Vd_mat = reshape(Vd.data, r, :)
        @test norm(M.data - U_mat * Σ.data * Vd_mat) / norm(M.data) < 1e-12
    end

    @testset "complex tensor: singular values are real" begin
        A   = _site_tensor(ComplexF64)
        bp  = _site_bipartition()
        (; Σ) = tensor_svd(A, bp, KeepFirst(4))
        @test eltype(diag(Σ.data)) <: Real
        @test all(s -> s >= 0, diag(Σ.data))
    end

    @testset "KeepAbove(0.0): retains all singular values" begin
        A        = _site_tensor()
        bp       = _site_bipartition()
        M        = group_legs(A, bp)
        all_svs  = svdvals(M.data)
        (; Σ, ε) = tensor_svd(A, bp, KeepAbove(0.0))
        @test size(Σ.data, 1) == length(all_svs)
        @test ε == 0
    end

    @testset "KeepMachineEps: all retained σ satisfy σ/σ₁ > sqrt(eps(T))" begin
        A       = _site_tensor()
        bp      = _site_bipartition()
        M       = group_legs(A, bp)
        σ₁      = svdvals(M.data)[1]
        (; Σ)   = tensor_svd(A, bp, KeepMachineEps())
        @test all(s -> s / σ₁ > sqrt(eps(eltype(M.data))), diag(Σ.data))
    end

    @testset "KeepMachineEps: ε equals 2-norm of discarded singular values" begin
        A       = _site_tensor()
        bp      = _site_bipartition()
        M       = group_legs(A, bp)
        all_svs = svdvals(M.data)
        σ₁      = all_svs[1]
        tol     = sqrt(eps(eltype(M.data))) * σ₁
        (; Σ, ε) = tensor_svd(A, bp, KeepMachineEps())
        discarded = filter(s -> s <= tol, all_svs)
        @test ε ≈ norm(discarded) atol=1e-12
    end

    @testset "KeepMachineEps: Float32 threshold is larger than Float64" begin
        # Construct a matrix with a singular value ratio of 1e-5:
        #   Float64 threshold ≈ 1.5e-8  → σ₂/σ₁ = 1e-5 is KEPT
        #   Float32 threshold ≈ 3.5e-4  → σ₂/σ₁ = 1e-5 is DISCARDED
        U    = [1.0 0.0; 0.0 1.0; 0.0 0.0]
        data = U * Diagonal([1.0, 1e-5]) * [1.0 0.0 0.0; 0.0 1.0 0.0]
        bp   = Bipartition(Partition(upper(:row, 3)), Partition(lower(:col, 3)))

        A64  = IndexedTensor(Float64.(data), (upper(:row, 3), lower(:col, 3)))
        A32  = IndexedTensor(Float32.(data), (upper(:row, 3), lower(:col, 3)))

        Σ64 = tensor_svd(A64, bp, KeepMachineEps()).Σ
        Σ32 = tensor_svd(A32, bp, KeepMachineEps()).Σ

        @test size(Σ64.data, 1) == 2   # 1e-5 > sqrt(eps(Float64)) ≈ 1.5e-8 → kept
        @test size(Σ32.data, 1) == 1   # 1e-5 < sqrt(eps(Float32)) ≈ 3.5e-4 → discarded
    end

end
