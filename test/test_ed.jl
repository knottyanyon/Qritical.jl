# Tests for §10 — ExactDiagonalization, sparse/dense solve.
# Physics invariants: real eigenvalues, GS energy bound, MPS ↔ ED cross-check.

using SparseArrays   # Julia package for sparse matrix types; `using X` imports all exported names from X into scope 

@testset "§10.1 dense Hamiltonian matrix" begin   # `@testset "name" begin...end` groups tests under a named label; failures are reported together 
    @testset "dense_matrix is Hermitian" begin
        g = Chain(3)   # create a 3-site open chain geometry
        H = Heisenberg(g; J=1.0, h=0.1)   # Heisenberg Hamiltonian with coupling J=1 and field h=0.1; `;` separates positional from keyword arguments
        M = matrix_repr(H)   # build the 2^3 × 2^3 = 8×8 dense matrix representation of H
        @test M ≈ M' atol=1e-12   # `@test` is a macro that runs the assertion; `≈` is `isapprox` (approx equal); `M'` is the conjugate transpose ; `atol` sets absolute tolerance; physics: Hamiltonian must be Hermitian for real eigenvalues
    end

    @testset "dense_matrix eigenvalues are real" begin
        g = Chain(4)
        H = XXZ(g; J=1.0, Jz=0.5, h=0.2)   # XXZ chain with anisotropy Δ=Jz/J=0.5 and transverse field
        M = matrix_repr(H)
        ev = eigvals(Hermitian(M))   # `eigvals(Hermitian(M))` tells Julia to treat M as Hermitian (uses a faster real-symmetric eigensolver); `eigvals` = Python `np.linalg.eigvalsh` for Hermitian matrices
        @test all(abs.(imag.(eigvals(M))) .< 1e-10)   # `all(pred)` over a vector; `abs.(...)` broadcasts abs over array; `imag.(...)` broadcasts imag; `.< 1e-10` is element-wise comparison; physics: Hermitian operators have real eigenvalues
    end

    @testset "matrix_repr(H, SparseFormat()) and matrix_repr(H) agree" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        M_dense = matrix_repr(H)   # dense 8×8 matrix
        M_sparse = matrix_repr(H, SparseFormat())   # sparse representation (CSC format by default in Julia's SparseArrays)
        @test M_dense ≈ Matrix(M_sparse) atol=1e-12   # `Matrix(M_sparse)` converts sparse → dense for comparison ; checks that both representations store the same entries
    end

    @testset "matrix_repr(H, SparseFormat()) is Hermitian" begin
        g = Chain(4)
        H = XXZ(g; J=1.0, Jz=0.5, h=0.0)
        M = matrix_repr(H, SparseFormat())   # sparse Hermitian Hamiltonian
        @test M ≈ M' atol=1e-12   # sparse matrix transpose and conjugate is still supported; physics: Hermiticity must hold in sparse form too
    end
end

@testset "§10.1 ExactDiagonalization — ground state" begin
    @testset "solve(H, GroundState(), ED(:ground)) returns EDResult" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        result = solve(H, GroundState(), ExactDiagonalization(:ground))   # dispatch on (LatticeOperator, GroundState, ExactDiagonalization{:ground}) — three-argument Julia dispatch 
        @test result isa EDResult   # `isa` is Julia's isinstance ; checks that the return type is correct
        @test result.energy isa Float64   # field access with `.energy`; `isa Float64` checks the type of the field
        @test result.state isa Vector{ComplexF64}   # `Vector{ComplexF64}` = Python `np.ndarray` with dtype=complex128 and ndim=1; GS vector stored as 1D complex vector
    end

    @testset "GS energy ≤ all eigenvalues" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        result = solve(H, GroundState(), ExactDiagonalization(:ground))   # compute ground state via Lanczos (KrylovKit)
        all_ev = real.(eigvals(Hermitian(matrix_repr(H))))   # full spectrum from dense diagonalisation 
        @test result.energy ≤ minimum(all_ev) + 1e-8   # ground state energy must be ≤ all eigenvalues; `minimum(iter)` = Python `min(iter)`; physics: E₀ = min eigenvalue
    end

    @testset "ED(:full) returns complete spectrum" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        result = solve(H, GroundState(), ExactDiagonalization(:full))   # `:full` mode: full diagonalization returns all eigenvalues/vectors
        d = local_dim(SpinHalf())   # d=2 for spin-1/2
        @test length(result.spectrum) == d^3   # 2^3 = 8 eigenvalues  # `^` is exponentiation. full Hilbert space dimension = d^L; `length(...)` = Python `len(...)`
    end

    @testset "ED(:ground) GS energy matches eigmin on L=4 Heisenberg" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        E_ed = minimum(real.(eigvals(Hermitian(matrix_repr(H)))))   # reference: minimum eigenvalue from full dense diagonalization
        result = solve(H, GroundState(), ExactDiagonalization(:ground))   # Lanczos ground state
        @test result.energy ≈ E_ed atol=1e-8   # `≈` with `atol` tolerance; Lanczos and full diagonalization should agree to machine precision
    end

    @testset "ED GS energy matches power method (L=4)" begin
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        ed_result = solve(H, GroundState(), ExactDiagonalization(:ground))   # ED ground state energy (exact)

        ψ_init = let   # `let...end` creates a local scope; variables defined inside don't leak. useful to avoid naming collisions
            psi_vec = normalize(randn(ComplexF64, 2^4))   # random complex vector, normalised; `randn(T, n)` draws n IID standard Gaussian random numbers of type T 
            to_mps(as_state(psi_vec, [2, 2, 2, 2]); trunc=NoTrunc(), form=:left)   # convert to MPS with no truncation and left-canonical form
        end
        pm_result = power_method(
            H, ψ_init; shift=4.0, tol=1e-8, maxiter=200, trunc=MaxBondDimTrunc(16)
        )   # power method: iterates (λI−H)^n until convergence; `shift=4.0` must exceed the spectral radius
        @test pm_result.energy ≈ ed_result.energy atol=1e-4   # cross-validation: two very different algorithms (matrix-based ED vs tensor-network power method) must agree on the ground-state energy
    end

    @testset "GS state is normalized" begin
        g = Chain(3)
        H = Heisenberg(g; J=1.0)
        result = solve(H, GroundState(), ExactDiagonalization(:ground))
        @test norm(result.state) ≈ 1.0 atol=1e-10   # `norm(v)` = ‖v‖₂ = ‖v‖ Euclidean norm ; ground state must be unit-normalised
    end

    @testset "XXZ GS energy at Δ=1 (isotropic) matches exact result L=4" begin
        # For L=4 OBC Heisenberg, E0 = -1.6160254...
        g = Chain(4)
        H = Heisenberg(g; J=1.0)
        result = solve(H, GroundState(), ExactDiagonalization(:ground))
        @test result.energy ≈ -1.6160254037844388 atol=1e-8   # known exact value for L=4 OBC Heisenberg; compared to a hardcoded reference (allows catching regressions)
    end
end

@testset "§10.1 sparse operator size guard" begin
    @testset "large Hilbert space is rejected by sparse" begin
        # If local_dim^L > 2^20 (≈1M), sparse should refuse
        g = Chain(25)   # d=2 → 2^25 > 1M  # L=25 gives Hilbert space dimension 2^25 ≈ 33M > the 2^20 limit
        H = Heisenberg(g; J=1.0)
        @test_throws ArgumentError matrix_repr(H, SparseFormat())   # `@test_throws ExceptionType expr` checks that `expr` throws the given exception type; physics: we refuse to allocate a 33M×33M matrix
    end
end

@testset "§10.1 dense_matrix size guard (closes #85)" begin
    @testset "dense_matrix rejects Hilbert spaces larger than 2^20" begin
        # solve(:time) and solve(:full) call dense_matrix, not sparse.
        # The same 2^20 guard must apply there too.
        g = Chain(25)   # 2^25 >> 2^20
        H = Heisenberg(g; J=1.0)
        @test_throws ArgumentError matrix_repr(H)   # dense matrix also has the size guard (the guard is in _check_dim)
    end

    @testset "solve(:full) rejects oversized Hilbert space via dense_matrix guard" begin
        g = Chain(25)
        H = Heisenberg(g; J=1.0)
        @test_throws ArgumentError solve(H, GroundState(), ExactDiagonalization(:full))   # `:full` calls dense_matrix internally so the same guard triggers
    end

    @testset "solve(:time) rejects oversized Hilbert space via dense_matrix guard" begin
        g = Chain(25)
        H = Heisenberg(g; J=1.0)
        v = StatevectorState(zeros(ComplexF64, 1))   # dummy — error fires before use  # construct a dummy initial state; the error should be thrown before v is ever used (the size check happens at matrix build time)
        p = ConstantProtocol{RealTime}(RealTime(), 0.1, 1, H)   # `ConstantProtocol{RealTime}` is the parameterized struct with time axis type baked in; `{RealTime}` is the type parameter (no Python equivalent)
        @test_throws ArgumentError solve(H, v, ExactDiagonalization(:time), p)   # `:time` also calls dense_matrix so the same guard fires
    end
end
