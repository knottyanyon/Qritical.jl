# Tests for: the index-variance & leg-position convention (von Delft covariant notation).
#
# This file exists to pin down, once and for all, which legs are `Upper` vs `Lower`
# and in which position they sit — the class of error we kept re-introducing. It is
# the *executable* statement of the convention; treat a red test here as the code
# disagreeing with the physics, never the other way round.
#
# ── The convention ────────────────────────────────────────────────────────────
# 
# The single source of truth for a leg's variance is its bond-arrow direction, and
# every bond arrow points TOWARD the orthogonality centre:
#
#     Upper  = arrow INTO the blob   = domain   = dual  V'   ⟹  which_space == :domain
#     Lower  = arrow OUT of the blob = codomain = primal V   ⟹  which_space == :codomain
#
# The physical leg σ is the contravariant ket-expansion coefficient A^σ (the numbers
# we store, contracted against the basis ket |σ⟩), so σ is `Upper` on EVERY tensor.
#
#     coefficient tensor A^{σ₁…σ_L} fed to to_mps/as_state : all σ Upper
#     left-canonical  site  A^{i,σ}_k  (bonds L→R) : (vL:Up,  σ:Up, vR:Low)
#     right-canonical site  B_k^{i,σ}  (bonds R→L) : (vL:Low, σ:Up, vR:Up )
#     mixed-form centre     (arrowheads converge)  : (vL:Up,  σ:Up, vR:Up )
#
# Adjoint (`Base.adjoint`, postfix A'; sugar `dagger`): flip every leg's variance
# AND reverse the leg order — (MN)† = N†M† — conjugating the data. Labels are
# NEVER primed: variance + position alone carry the dagger (von Delft's policy,
# TNB.3 degree table: [A†]^{β}_{σα} := conj(A^{ασ}_{β})).
#
# Flip (`flip`): raise/lower ONE leg — variance-only. No reversal, no conjugation,
# and with the trivial metric of an orthonormal basis the backing array is left
# untouched (TNB.2 addendum, eqs. (27)-(28)). Flipping a bond means flipping BOTH
# of its ends (arrows stay head-to-tail); a centre move flips exactly one bond.

# Upper ⇔ domain, Lower ⇔ codomain (see src/indices.jl:181-182).
is_upper(ix) = which_space(ix) === :domain
is_lower(ix) = which_space(ix) === :codomain

# A random normalised L-site coefficient tensor A^{σ₁…σ_L} with the correct Upper
# physical legs. Normalised so that EVERY site of a canonical form is isometric —
# for an unnormalised state the edge site carries ‖ψ‖ and fails A†A = I.
function rand_coeff_tensor(L::Int, d::Int)
    data = randn(ComplexF64, ntuple(_ -> d, L)...)
    data ./= norm(data)
    return QTensor(data, Tuple(upper(Symbol(:σ, i), d) for i in 1:L))
end

@testset "Index convention (variance & position)" begin

    @testset "as_state: coefficient-tensor physical legs are Upper" begin
        A = as_state(randn(ComplexF64, 2^4), fill(2, 4))
        @test length(A.indices) == 4
        for (i, ix) in enumerate(A.indices)
            @test is_upper(ix)                    # σ is contravariant, never Lower
            @test label(ix) == Symbol(:σ, i)      # labelled σ1, σ2, …
            @test dim(ix) == size(A.data, i)      # leg i sits on array axis i
        end
    end

    @testset "leg positions: stored order is (vL, σ, vR); leftmost = row" begin
        mps = to_mps(rand_coeff_tensor(4, 2); form = :left)
        for t in mps.tensors
            @test length(t.indices) == 3
            @test label(t.indices[1]) == :vL       # leftmost bond = matrix row
            @test label(t.indices[2]) == :σ        # physical leg in the middle
            @test label(t.indices[3]) == :vR       # rightmost bond = matrix column
            # positions must line up with the backing array axes
            @test dim(t.indices[1]) == size(t.data, 1)
            @test dim(t.indices[2]) == size(t.data, 2)
            @test dim(t.indices[3]) == size(t.data, 3)
        end
    end

    @testset "to_mps :left  → every site (vL:Up, σ:Up, vR:Low)" begin
        mps = to_mps(rand_coeff_tensor(4, 2); form = :left)
        for t in mps.tensors
            vL, σ, vR = t.indices
            @test is_upper(vL)
            @test is_upper(σ)
            @test is_lower(vR)
        end
    end

    @testset "to_mps :right → every site (vL:Low, σ:Up, vR:Up)" begin
        mps = to_mps(rand_coeff_tensor(4, 2); form = :right)
        for t in mps.tensors
            vL, σ, vR = t.indices
            @test is_lower(vL)
            @test is_upper(σ)
            @test is_upper(vR)
        end
    end

    @testset "physical σ is Upper regardless of canonical form" begin
        A = rand_coeff_tensor(4, 2)
        for form in (:left, :right)
            for t in to_mps(A; form = form).tensors
                @test is_upper(t.indices[2])
            end
        end
    end

    @testset "bonds are directional: each bond is Upper on one end, Lower on the other" begin
        A = rand_coeff_tensor(5, 2)
        for form in (:left, :right)
            mps = to_mps(A; form = form)
            for i in 1:(length(mps.tensors) - 1)
                vR_i   = mps.tensors[i].indices[3]     # right end of bond i
                vL_ip1 = mps.tensors[i + 1].indices[1] # left  end of bond i
                @test is_upper(vR_i) ⊻ is_upper(vL_ip1)   # exactly one is Upper
            end
        end
    end

    @testset "mixed form: arrows converge on the centre site" begin
        k = 3
        mixed = canonicalize(to_mps(rand_coeff_tensor(6, 2); form = :left), BondCanonical(k))
        L = length(mixed.tensors)
        # left of centre: left-canonical, arrows L→R
        for i in 1:(k - 1)
            vL, σ, vR = mixed.tensors[i].indices
            @test is_upper(vL) && is_upper(σ) && is_lower(vR)
        end
        # centre site k: both bond arrows point in ⟹ all Upper
        vLc, σc, vRc = mixed.tensors[k].indices
        @test is_upper(vLc) && is_upper(σc) && is_upper(vRc)
        # right of centre: right-canonical, arrows R→L
        for i in (k + 1):L
            vL, σ, vR = mixed.tensors[i].indices
            @test is_lower(vL) && is_upper(σ) && is_upper(vR)
        end
    end
end

@testset "Index convention (adjoint & flips)" begin

    @testset "adjoint: flip variance, reverse order, conjugate data" begin
        A = QTensor(randn(ComplexF64, 2, 3, 4),
                    (upper(:i, 2), upper(:σ, 3), lower(:k, 4)))
        Ad = A'
        N = length(A.indices)
        # index rule: p-th leg of A† = variance-flip of the (N+1-p)-th leg of A
        for p in 1:N
            orig = A.indices[N + 1 - p]
            @test label(Ad.indices[p]) == label(orig)       # labels never primed
            @test dim(Ad.indices[p]) == dim(orig)
            @test is_upper(Ad.indices[p]) ⊻ is_upper(orig)  # variance flipped
        end
        # data rule: the leg reversal is mirrored by the array axes, entries conjugated
        @test Ad.data ≈ conj(permutedims(A.data, (3, 2, 1)))
        # `dagger` is syntactic sugar for the same operation
        @test dagger(A).indices == Ad.indices
        @test dagger(A).data == Ad.data
    end

    @testset "adjoint is an involution: (A†)† = A" begin
        A = QTensor(randn(ComplexF64, 2, 3, 4),
                    (upper(:i, 2), upper(:σ, 3), lower(:k, 4)))
        Add = (A')'
        @test Add.indices == A.indices
        @test Add.data ≈ A.data
    end

    @testset "adjoint degree ladder (TNB.3): vector and matrix cases" begin
        # degree-1: A^σ → [A†]_σ, entries conjugated
        v = QTensor(randn(ComplexF64, 3), (upper(:σ, 3),))
        @test is_lower(v'.indices[1])
        @test label(v'.indices[1]) == :σ
        @test v'.data ≈ conj(v.data)
        # degree-2: T^σ_α → [T†]^α_σ, data = LinearAlgebra matrix adjoint
        T = QTensor(randn(ComplexF64, 2, 3), (upper(:σ, 2), lower(:α, 3)))
        Td = T'
        @test label(Td.indices[1]) == :α && is_upper(Td.indices[1])
        @test label(Td.indices[2]) == :σ && is_lower(Td.indices[2])
        @test Td.data ≈ adjoint(T.data)
    end

    @testset "adjoint isometry anchor: [A†]^{k'}_{σ,i} A^{i,σ}_k = δ^{k'}_k" begin
        mps = to_mps(rand_coeff_tensor(4, 2); form = :left)
        for t in mps.tensors
            td = t'                       # legs (vR, σ, vL), each variance-flipped
            # every contracted pair must be one Upper, one Lower
            @test is_upper(td.indices[3]) ⊻ is_upper(t.indices[1])   # vL pair
            @test is_upper(td.indices[2]) ⊻ is_upper(t.indices[2])   # σ  pair
            dL, d, dR = size(t.data)
            G = zeros(ComplexF64, dR, dR)
            for k′ in 1:dR, k in 1:dR, σ in 1:d, i in 1:dL
                G[k′, k] += td.data[k′, σ, i] * t.data[i, σ, k]
            end
            @test G ≈ Matrix{ComplexF64}(I, dR, dR)
        end
    end

    @testset "adjoint isometry anchor: B_k^{i,σ} [B†]_{σ,i}^{k'} = δ_k^{k'}" begin
        mps = to_mps(rand_coeff_tensor(4, 2); form = :right)
        for t in mps.tensors
            td = t'
            @test is_upper(td.indices[1]) ⊻ is_upper(t.indices[3])   # vR pair
            @test is_upper(td.indices[2]) ⊻ is_upper(t.indices[2])   # σ  pair
            dL, d, dR = size(t.data)
            G = zeros(ComplexF64, dL, dL)
            for k in 1:dL, k′ in 1:dL, σ in 1:d, j in 1:dR
                G[k, k′] += t.data[k, σ, j] * td.data[j, σ, k′]
            end
            @test G ≈ Matrix{ComplexF64}(I, dL, dL)
        end
    end

    @testset "flip primitive: variance-only, label/dim preserved, involutive" begin
        up, lo = upper(:a, 3), lower(:a, 3)
        @test flip(up) == lo
        @test flip(lo) == up
        @test flip(flip(up)) == up
        @test which_space(flip(up)) === :codomain
        @test which_space(flip(lo)) === :domain
    end

    @testset "flip on a tensor leg: trivial metric leaves the data untouched" begin
        A = QTensor(randn(ComplexF64, 2, 3), (upper(:σ, 2), lower(:α, 3)))
        B = flip(A, 1)
        @test is_lower(B.indices[1])
        @test label(B.indices[1]) == :σ
        @test B.indices[2] == A.indices[2]    # other legs untouched
        @test B.data == A.data                # raising/lowering is numerically free
    end

    @testset "adjoint ≠ flip-all: reversal and conjugation matter" begin
        A = QTensor(randn(ComplexF64, 2, 2), (upper(:σ, 2), lower(:α, 2)))
        F  = flip(flip(A, 1), 2)   # every leg flipped, order and data untouched
        Ad = A'
        @test F.indices != Ad.indices          # same legs, opposite order
        @test !(F.data ≈ Ad.data)              # no conj, no permute happened
    end

    @testset "centre transport: moving the centre flips exactly one bond" begin
        k = 3
        mps = to_mps(rand_coeff_tensor(6, 2); form = :left)
        m1 = canonicalize(mps, BondCanonical(k))
        m2 = canonicalize(m1, BondCanonical(k + 1))
        L = length(m1.tensors)
        # site k: was the centre (Up,Up,Up), now left-canonical (Up,Up,Low)
        vL, σ, vR = m2.tensors[k].indices
        @test is_upper(vL) && is_upper(σ) && is_lower(vR)
        # site k+1: was right-canonical (Low,Up,Up), now the centre (Up,Up,Up)
        vL, σ, vR = m2.tensors[k + 1].indices
        @test is_upper(vL) && is_upper(σ) && is_upper(vR)
        # every other site keeps its tags — only the (k, k+1) bond flipped
        for i in 1:L
            (i == k || i == k + 1) && continue
            @test m2.tensors[i].indices == m1.tensors[i].indices
        end
        # the move is pure gauge: the state is physically unchanged
        @test overlap(m2, m1) ≈ overlap(m1, m1)
    end
end
