# Tests for §6.2 — LatticeOperator/Hamiltonian, named constructors, MPO
# Physics invariants follow MasterPlan_Part2.md §5 and §7.

@testset "§6.2 LatticeOperator — term structures and named constructors" begin

    @testset "uniform coupling helper" begin
        @test uniform_coupling(4, 1.0) == fill(1.0, 4)
        @test uniform_coupling(3, 0.5) == [0.5, 0.5, 0.5]
        @test uniform_coupling(1, 2.0) == [2.0]
    end

    @testset "OneSiteTerm constructor" begin
        ops = algebra_generators(SpinHalf())
        lt = OneSiteTerm(2, ops.Sz, -1.5)
        @test lt.site     == 2
        @test lt.coupling == -1.5
        @test lt.op       ≈ ops.Sz
    end

    @testset "TwoSiteTerm constructor" begin
        ops = algebra_generators(SpinHalf())
        bt = TwoSiteTerm(1, 2, ops.Sp, ops.Sm, 0.5)
        @test bt.i        == 1
        @test bt.j        == 2
        @test bt.coupling == 0.5
        @test bt.op_i     ≈ ops.Sp
        @test bt.op_j     ≈ ops.Sm
    end

    @testset "XXZ — term count for open chain" begin
        g = Chain(4)   # 3 bonds, 4 sites
        H = XXZ(g; J=1.0, Jz=1.0, h=0.5)
        # L onsite terms (one Sz per site)
        @test length(H.onsite) == 4
        # 3 bonds × 3 term types = 9 bond terms
        @test length(H.bond) == 9
        @test H.dof isa Spin
    end

    @testset "XXZ — field sign convention: −ΣhᵢSᶻ" begin
        # Course convention: onsite coupling = −hᵢ, so H_onsite = −h·Sz at each site
        g = Chain(4)
        H = XXZ(g; J=0.0, Jz=0.0, h=1.0)
        ops = algebra_generators(SpinHalf())
        # Each onsite term should have coupling = −1.0 and op = Sz
        for lt in H.onsite
            @test lt.coupling ≈ -1.0  atol=1e-12
            @test lt.op       ≈ ops.Sz atol=1e-12
        end
    end

    @testset "XXZ — energy on all-up product state with h=1" begin
        # H = −h Σ Sz;  |↑↑↑↑⟩ has Sz = +½ per site → E = −1 × 4 × ½ = −2
        g = Chain(4)
        H = XXZ(g; J=0.0, Jz=0.0, h=1.0)
        up = ComplexF64[1, 0]
        E_onsite = sum(lt.coupling * dot(up, lt.op * up) for lt in H.onsite)
        @test real(E_onsite) ≈ -2.0  atol=1e-12
    end

    @testset "Heisenberg = XXZ(J=Jz)" begin
        g = Chain(5)
        H_xxz  = XXZ(g; J=1.0, Jz=1.0, h=0.0)
        H_heis = Heisenberg(g; J=1.0, h=0.0)
        @test length(H_heis.onsite) == length(H_xxz.onsite)
        @test length(H_heis.bond)   == length(H_xxz.bond)
        # Same couplings on corresponding bond terms
        for (b1, b2) in zip(H_heis.bond, H_xxz.bond)
            @test b1.coupling ≈ b2.coupling  atol=1e-12
            @test b1.i == b2.i && b1.j == b2.j
        end
    end

    @testset "Ising — transverse-field term count" begin
        g = Chain(4)
        H = Ising(g; J=1.0, h=1.0)
        # L onsite terms (transverse field h·Sx) + L-1 bond terms (Sz⊗Sz)
        @test length(H.onsite) == 4
        @test length(H.bond)   == 3
    end

    @testset "Ising — ZZ coupling sign and operator" begin
        g = Chain(3)
        H = Ising(g; J=1.0, h=0.0)
        ops = algebra_generators(SpinHalf())
        # Bond terms should be Sz⊗Sz with coupling J
        for bt in H.bond
            @test bt.op_i ≈ ops.Sz  atol=1e-12
            @test bt.op_j ≈ ops.Sz  atol=1e-12
            @test bt.coupling ≈ 1.0  atol=1e-12
        end
    end

end

@testset "§6.2 Observable constructors" begin

    @testset "total_magnetization — L onsite Sz terms with coupling +1" begin
        g = Chain(4)
        M = total_magnetization(g)
        ops = algebra_generators(SpinHalf())
        @test length(M.onsite) == 4
        @test isempty(M.bond)
        for lt in M.onsite
            @test lt.coupling ≈ 1.0   atol=1e-12
            @test lt.op       ≈ ops.Sz atol=1e-12
        end
    end

    @testset "staggered_magnetization — alternating sign" begin
        g = Chain(4)
        Ms = staggered_magnetization(g)
        # Couplings: (+1)^1=+1, (+1)^2... wait: (-1)^i
        expected_couplings = [(-1.0)^i for i in 1:4]
        for (lt, c) in zip(Ms.onsite, expected_couplings)
            @test lt.coupling ≈ c  atol=1e-12
        end
    end

    @testset "op_at_site — single-site observable" begin
        g = Chain(6)
        O = op_at_site(SpinHalf(), :Sz, 3)
        @test length(O.onsite) == 1
        @test O.onsite[1].site == 3
        ops = algebra_generators(SpinHalf())
        @test O.onsite[1].op ≈ ops.Sz  atol=1e-12
        @test O.onsite[1].coupling ≈ 1.0  atol=1e-12
    end

    @testset "two_point — single bond term" begin
        g = Chain(6)
        O = two_point(g, SpinHalf(), :Sz, 2, :Sz, 5)
        @test isempty(O.onsite)
        @test length(O.bond) == 1
        @test O.bond[1].i == 2 && O.bond[1].j == 5
        @test O.bond[1].coupling ≈ 1.0  atol=1e-12
    end

end

@testset "§6.2 Brute-force energy via dense_matrix" begin

    @testset "Heisenberg L=2 — singlet GS energy = −3/4" begin
        g = Chain(2)
        H = Heisenberg(g; J=1.0)
        mat = matrix_repr(H)
        # Heisenberg L=2 spectrum: three-fold degenerate +1/4, one singlet −3/4
        evals = sort(real.(eigvals(mat)))
        @test evals[1] ≈ -3/4  atol=1e-10
        @test evals[2] ≈  1/4  atol=1e-10
        @test evals[3] ≈  1/4  atol=1e-10
        @test evals[4] ≈  1/4  atol=1e-10
    end

    @testset "XXZ L=2 Ising limit (Jz=2, J=0) — doublet at +½, doublet at −½" begin
        g = Chain(2)
        H = XXZ(g; J=0.0, Jz=2.0, h=0.0)
        mat = matrix_repr(H)
        evals = sort(real.(eigvals(mat)))
        # Sz⊗Sz: diag(+½, −½, −½, +½) for Jz=2 → eigenvalues {+½, −½, −½, +½}
        @test evals ≈ [-0.5, -0.5, 0.5, 0.5]  atol=1e-10
    end

    @testset "Field-only Hamiltonian — energy = −h·(L/2) on all-up state" begin
        g = Chain(4)
        H = XXZ(g; J=0.0, Jz=0.0, h=1.0)
        mat = matrix_repr(H)
        # All-up state: kron([1,0],[1,0],[1,0],[1,0]) = e₁
        psi_up = zeros(ComplexF64, 16); psi_up[1] = 1.0
        E = real(dot(psi_up, mat * psi_up))
        @test E ≈ -2.0  atol=1e-10   # −h·L·(1/2) = −1·4·½ = −2
    end

    @testset "total_magnetization operator — all-up state has M=L/2" begin
        g = Chain(4)
        M = total_magnetization(g)
        mat = matrix_repr(M)
        psi_up = zeros(ComplexF64, 16); psi_up[1] = 1.0
        m = real(dot(psi_up, mat * psi_up))
        @test m ≈ 2.0  atol=1e-10   # L × ½ = 2
    end

    @testset "Heisenberg L=4 — GS energy below −1" begin
        # Known result: Heisenberg OBC L=4 GS ≈ −1.6160254...
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        mat = matrix_repr(H)
        E_gs = minimum(real.(eigvals(mat)))
        @test E_gs < -1.0
        @test E_gs ≈ -1.6160254 atol=1e-5
    end

end

@testset "§6.2 MPO construction" begin

    @testset "MPO tensor count = L" begin
        g = Chain(5)
        H = Heisenberg(g)
        mpo = MPO(H)
        @test length(mpo.tensors) == 5
    end

    @testset "MPO boundary bond dimensions are 1" begin
        g = Chain(4)
        H = Heisenberg(g)
        mpo = MPO(H)
        # Left site: χ_left = 1
        @test size(mpo.tensors[1], 1) == 1
        # Right site: χ_right = 1
        @test size(mpo.tensors[end], 4) == 1
    end

    @testset "MPO physical dimensions match DoF" begin
        g = Chain(4)
        H = Heisenberg(g)
        mpo = MPO(H)
        d = local_dim(H.dof)
        for W in mpo.tensors
            @test size(W, 2) == d   # d_out
            @test size(W, 3) == d   # d_in
        end
    end

    @testset "MPO Heisenberg L=2 energy on singlet state ≈ −3/4" begin
        g = Chain(2)
        H = Heisenberg(g; J=1.0)
        mpo = MPO(H)
        # Singlet: ψ[↑,↓] = 1/√2, ψ[↓,↑] = −1/√2 (as 2×2 array, then to_mps)
        data = ComplexF64[0 1/√2; -1/√2 0]   # ψ[σ₁, σ₂]
        ψ_t = as_state(vec(reshape(data, 4)), [2, 2])
        ψ   = to_mps(ψ_t; trunc=NoTrunc(), form=:left)
        E   = expect(ψ, mpo)
        @test real(E) ≈ -3/4  atol=1e-8
        @test abs(imag(E))    < 1e-10
    end

    @testset "MPO field-only L=4 energy on all-up MPS ≈ −2" begin
        g = Chain(4)
        H = XXZ(g; J=0.0, Jz=0.0, h=1.0)
        mpo = MPO(H)
        # All-up state: product state [1,0]⊗[1,0]⊗[1,0]⊗[1,0]
        psi_vec = zeros(ComplexF64, 2^4); psi_vec[1] = 1.0
        ψ_t = as_state(psi_vec, [2, 2, 2, 2])
        ψ   = to_mps(ψ_t; trunc=NoTrunc(), form=:left)
        E   = expect(ψ, mpo)
        @test real(E) ≈ -2.0  atol=1e-8
    end

    @testset "MPO Heisenberg L=4 energy matches dense_matrix GS" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        mpo = MPO(H)
        mat = matrix_repr(H)
        # GS from dense
        F   = eigen(Hermitian(mat))
        E_dense = F.values[1]
        ψ_dense = F.vectors[:, 1]
        # Build MPS from GS eigenvector
        ψ_t = as_state(ψ_dense, [2, 2, 2, 2])
        ψ   = to_mps(ψ_t; trunc=NoTrunc(), form=:left)
        E_mpo = real(expect(ψ, mpo))
        @test E_mpo ≈ E_dense  atol=1e-6
    end

    @testset "MPO identity check: ⟨ψ|I⊗…⊗I|ψ⟩ = ‖ψ‖²" begin
        # The identity operator as MPO should give the norm squared
        g = Chain(3)
        H_id = identity_operator(g, SpinHalf())
        mpo_id = MPO(H_id)
        ψ_vec = normalize(randn(ComplexF64, 2^3))
        ψ_t   = as_state(ψ_vec, [2, 2, 2])
        ψ     = to_mps(ψ_t; trunc=NoTrunc(), form=:left)
        @test real(expect(ψ, mpo_id)) ≈ 1.0  atol=1e-8   # normalized state
    end

    @testset "MPO non-adjacent two_point: ⟨Sz₁Sz₃⟩ via MPO matches dense_matrix (closes #78)" begin
        # Regression test for the FSM carry bug: a channel opened at site i must
        # propagate through intermediate sites via W[k+1,:,:,k+1]=Id before closing
        # at site j>i+1.  Without the fix, the MPO path returns 0 for all non-NN terms.
        g   = Chain(4)
        ops = algebra_generators(SpinHalf())
        # two_point operator A₁B₃ — non-nearest-neighbour
        O   = two_point(g, SpinHalf(), :Sz, 1, :Sz, 3)

        # Reference from dense_matrix on a random normalised state
        mat = matrix_repr(O)
        ψ_vec = normalize(randn(ComplexF64, 2^4))
        ref   = dot(ψ_vec, mat * ψ_vec)

        # MPO path
        mpo   = MPO(O)
        ψ_t   = as_state(ψ_vec, [2, 2, 2, 2])
        ψ     = to_mps(ψ_t; trunc=NoTrunc(), form=:left)
        result = expect(ψ, mpo)

        @test real(result) ≈ real(ref)  atol=1e-8
        @test abs(imag(result))         < 1e-10
    end

    @testset "MPO non-adjacent Sz₁Sz₄ on L=5 chain matches dense_matrix (closes #78)" begin
        g     = Chain(5)
        O     = two_point(g, SpinHalf(), :Sz, 1, :Sz, 4)
        mat   = matrix_repr(O)
        ψ_vec = normalize(randn(ComplexF64, 2^5))
        ref   = dot(ψ_vec, mat * ψ_vec)

        mpo   = MPO(O)
        ψ_t   = as_state(ψ_vec, [2, 2, 2, 2, 2])
        ψ     = to_mps(ψ_t; trunc=NoTrunc(), form=:left)
        result = expect(ψ, mpo)

        @test real(result) ≈ real(ref)  atol=1e-8
    end

end
