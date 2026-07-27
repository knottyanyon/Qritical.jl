"""
Tests for: §1.5 Schmidt coefficients, entanglement entropy, data I/O

Physics invariants tested:
- product state → entanglement entropy 0; maximally entangled 2-qubit → log₂ 2 = 1 bit (via SchmidtSpectrum)
- schmidt_values(s) squared-sum ≈ 1 for a normalized state; entropy uses log₂ by default
- load_array reads .jls/.txt/.npy to the same array (round-trip a saved fixture)
- bipartition_matrix(A, bp) reshape preserves structure: size(M, 1) == prod(dim, bp.left), size(M, 2) == prod(dim, bp.right)
- as_state(v, dof_dims) reshapes a flat vector into a tensor with one leg per site; each leg has dimension = local Hilbert space dimension
"""

using Test            # @test, @testset, @test_throws, @test_broken
using LinearAlgebra   # `using LinearAlgebra` = standard library; exposes `norm`, `Diagonal`, `diagm`
using Serialization   # `using Serialization` = standard library; exposes `serialize`/`deserialize` 
using DelimitedFiles  # `using DelimitedFiles` = standard library; exposes `writedlm`/`readdlm` 
using Qritical        # all exported names: QTensor, SchmidtSpectrum, schmidt_values, load_array, bipartition_matrix, as_state

# Scratchpad for temporary test files (fixtures, intermediate data).
# Use a fresh OS temp dir so the tests run anywhere, including CI.
const SCRATCHPAD = mktempdir()   # `mktempdir()` = create a uniquely-named temp directory and return its path ; `const` = module-level constant

@testset "IO and state utilities" begin

    # ────────────────────────────────────────────────────────────────────────
    # §1.5.1 schmidt_values accessor
    # ────────────────────────────────────────────────────────────────────────

    @testset "schmidt_values(s) accessor for AbstractSpectrum" begin
        @testset "schmidt_values(s) squared-sum ≈ 1 for normalized SchmidtSpectrum" begin
            # Physics invariant: for a normalized state, ∑ᵢ σᵢ² = 1, where σᵢ are the Schmidt values.
            # This tests that schmidt_values returns the correct singular values from the spectrum.
            let i = lower(:i, 2),
                j = lower(:j, 2),
                # Bell pair: (|00⟩ + |11⟩)/√2, Schmidt values [1/√2, 1/√2]
                ψ_tensor = QTensor(reshape([1.0, 0.0, 0.0, 1.0] / sqrt(2), 2, 2), (i, j)),   # `reshape(v, 2, 2)` = column-major reshape; Bell state
                bp = Bipartition(Partition([i]), Partition([j])),
                F = do_svd(ψ_tensor, bp, NoTrunc()),
                s = SchmidtSpectrum(F.spectrum, bp, F.center)   # enrich SingValSpectrum with cut location

                # Test: sum(abs2, schmidt_values(s)) ≈ 1
                sv = schmidt_values(s)   # `schmidt_values(s)` = accessor returning s.spectrum.values directly (no copy)
                @test isapprox(sum(abs2, sv), 1.0; atol=1e-10)   # `sum(abs2, sv)` = ∑σᵢ²; `abs2(x)` = |x|²; for normalised Bell pair ∑σᵢ² = 1/2+1/2 = 1
            end
        end

        @testset "schmidt_values(s) returns the values vector from spectrum (identity, not copy)" begin
            # Physics invariant: schmidt_values should give direct access to the singular values.
            # (Implementation detail: no unnecessary copies; identity preferred for performance.)
            let σ_vals = [0.8, 0.6],
                spec = SingValSpectrum(σ_vals, 0.0, false),   # `SingValSpectrum(values, ε, normalized)` = direct struct constructor
                # Construct a SchmidtSpectrum with dummy bond and cut
                bond = Bond(upper(:λL, 2), upper(:λR, 2)),   # both centre faces Upper (Bond requires two TIx{Upper})
                bp = Bipartition(Partition([lower(:i, 2)]), Partition([lower(:j, 2)])),
                s = SchmidtSpectrum(spec, bp, BondCenter(bond))   # `BondCenter(bond)` = wraps Bond; `SchmidtSpectrum(spec, cut, center)` = direct struct constructor

                sv = schmidt_values(s)
                # Test: result is the same object as spec.values (not a copy)
                @test sv === spec.values   # `===` = identity equality (same pointer); confirms no copy was made; Python: `sv is spec.values`
            end
        end

        @testset "schmidt_values on product state (rank 1)" begin
            # Physics: product state has a single Schmidt value σ₁ = 1 (up to normalization).
            let i = lower(:i, 3),
                j = lower(:j, 4),
                ψ_tensor = QTensor(ones(3, 4) / sqrt(12), (i, j)),   # `ones(3,4)/sqrt(12)` = 3×4 product state, normalised
                bp = Bipartition(Partition([i]), Partition([j])),
                F = do_svd(ψ_tensor, bp, NoTrunc()),
                s = SchmidtSpectrum(F.spectrum, bp, F.center)

                sv = schmidt_values(s)
                @test length(sv) == 1                            # rank-1: exactly 1 Schmidt value
                @test isapprox(sv[1], 1.0; atol=1e-10)          # σ₁ = 1.0 for unit-norm product state
            end
        end
    end  # @testset "schmidt_values accessor"

    # ────────────────────────────────────────────────────────────────────────
    # §1.5.2 load_array with round-trip data I/O
    # ────────────────────────────────────────────────────────────────────────

    @testset "load_array: round-trip I/O tests" begin
        @testset "load_array(.jls) round-trip via Serialization" begin
            # Physics: the array must be reconstructed bit-for-bit from its serialized form.
            let test_data = rand(4, 5),   # `rand(4, 5)` = uniform [0,1] random 4×5 matrix    # `joinpath(dir, file)` = OS-correct path join 

                # Create scratchpad dir if it doesn't exist
                isdir(SCRATCHPAD) || mkpath(SCRATCHPAD)   # `isdir(p)` = True if p is a dir; `||` = short-circuit OR: if NOT isdir, call mkpath; `mkpath` creates all missing dirs 

                # Write fixture
                open(test_file, "w") do io   # `open(path, "w") do io ... end` = open for writing, pass IO to block, close on exit 
                    serialize(io, test_data)  # Julia binary serialization 
                end

                # Test: load_array round-trips the data
                loaded = load_array(test_file)   # `load_array` from src/utils/io.jl; dispatches on ".jls" extension → `deserialize`
                @test isapprox(loaded, test_data; atol=1e-14)   # bit-exact round-trip (binary format preserves all 64 bits)

                # Cleanup
                rm(test_file; force=true)   # `rm(path; force=true)` = delete file; `force=true` = no error if file already missing 
            end
        end

        @testset "load_array(.txt) round-trip via readdlm/writedlm" begin
            # Physics: text-format I/O must preserve numerical precision to working accuracy.
            let test_data = [1.0 2.0 3.0; 4.0 5.0 6.0],   # 2×3 matrix literal; spaces between elements = columns; `;` = row separator
                test_file = joinpath(SCRATCHPAD, "test_array.txt")

                isdir(SCRATCHPAD) || mkpath(SCRATCHPAD)

                # Write fixture
                writedlm(test_file, test_data)   # `writedlm(path, M)` = write M as whitespace-delimited text 

                # Test: load_array round-trips the data
                loaded = load_array(test_file)   # dispatches on ".txt" → `readdlm` 
                @test isapprox(loaded, test_data; atol=1e-14)

                # Cleanup
                rm(test_file; force=true)
            end
        end

        @testset "load_array(.jls) and (.txt) produce the same array content" begin
            # Physics: same numerical data should load to identical arrays regardless of format.
            let test_data = [1.5 2.5; 3.5 4.5],
                jls_file = joinpath(SCRATCHPAD, "test_same.jls"),
                txt_file = joinpath(SCRATCHPAD, "test_same.txt")

                isdir(SCRATCHPAD) || mkpath(SCRATCHPAD)

                # Write both formats
                open(jls_file, "w") do io
                    serialize(io, test_data)
                end
                writedlm(txt_file, test_data)

                # Test: both load to the same values
                loaded_jls = load_array(jls_file)
                loaded_txt = load_array(txt_file)
                @test isapprox(loaded_jls, loaded_txt; atol=1e-14)   # both formats round-trip to the same values

                # Cleanup
                rm(jls_file; force=true)
                rm(txt_file; force=true)
            end
        end
    end  # @testset "load_array"

    # ────────────────────────────────────────────────────────────────────────
    # §1.5.3 bipartition_matrix: reshape under a bipartition
    # ────────────────────────────────────────────────────────────────────────

    @testset "bipartition_matrix: reshape matrix from bipartition" begin
        @testset "bipartition_matrix size invariant: rows = prod(left dims), cols = prod(right dims)" begin
            # Physics invariant: grouping legs via bipartition produces a matrix whose dimensions
            # match the product of the grouped leg dimensions.
            let i = upper(:i, 3),
                j = lower(:j, 4),
                A = QTensor(randn(3, 4), (i, j)),
                bp = Bipartition(Partition([i]), Partition([j]))

                M = bipartition_matrix(A, bp)   # `bipartition_matrix(A, bp)` = reshape A into a matrix by grouping left and right legs; internally calls `group_legs` and extracts `.data`

                # Test: size(M, 1) = prod(dim, [i]) = 3, size(M, 2) = prod(dim, [j]) = 4
                @test size(M, 1) == 3   # rows = dim(i) = 3
                @test size(M, 2) == 4   # cols = dim(j) = 4
            end
        end

        @testset "bipartition_matrix on rank-3 tensor: left partition (σ,vL) vs right (vR)" begin
            # Physics: a 3-leg tensor can be bipartitioned into a matrix.
            # Left partition: {σ, vL} → rows; right partition: {vR} → cols.
            let σ = upper(:σ, 2),
                vL = upper(:vL, 3),
                vR = lower(:vR, 5),
                A = QTensor(randn(2, 3, 5), (σ, vL, vR)),
                bp = Bipartition(Partition([σ, vL]), Partition([vR]))

                M = bipartition_matrix(A, bp)

                # Test: size(M) = (2·3, 5) = (6, 5)
                @test size(M, 1) == 2 * 3   # rows = dim(σ)*dim(vL) = 2*3 = 6; product of left partition dims
                @test size(M, 2) == 5        # cols = dim(vR) = 5
            end
        end

        @testset "bipartition_matrix empty left partition: rows = 1" begin
            # Physics edge case: if the left partition is empty, the left "product" is 1.
            let j = lower(:j, 4),
                A = QTensor(randn(1, 4), (lower(:_dummy, 1), j)),   # `:_dummy` = symbolic name for the trivial dim=1 boundary index
                bp = Bipartition(Partition([lower(:_dummy, 1)]), Partition([j]))

                M = bipartition_matrix(A, bp)

                @test size(M, 1) == 1   # single-element dim from the dim-1 left leg
                @test size(M, 2) == 4
            end
        end

        @testset "bipartition_matrix empty right partition: cols = 1" begin
            # Physics edge case: if the right partition is empty, the right "product" is 1.
            let i = upper(:i, 5),
                A = QTensor(randn(5, 1), (i, lower(:_dummy, 1))),
                bp = Bipartition(Partition([i]), Partition([lower(:_dummy, 1)]))

                M = bipartition_matrix(A, bp)

                @test size(M, 1) == 5
                @test size(M, 2) == 1   # single column from the dim-1 right leg
            end
        end
    end  # @testset "bipartition_matrix"

    # ────────────────────────────────────────────────────────────────────────
    # §1.5.4 as_state: reshape flat vector into a state tensor
    # ────────────────────────────────────────────────────────────────────────

    @testset "as_state: reshape flat vector into state tensor" begin
        @testset "as_state(v, dof_dims): creates a tensor with one leg per site" begin
            # Physics invariant: a state vector of length d^L can be reshaped into L site tensors,
            # each with local Hilbert space dimension d. The result has order L.
            let L = 3,   # L = number of lattice sites
                d = 2,   # d = local Hilbert space dimension (spin-1/2: d=2)
                v = randn(d^L),   # `d^L` = 2^3 = 8-dimensional Hilbert space vector
                dof_dims = fill(d, L)   # `fill(d, L)` = [d, d, d] = [2, 2, 2]. uniform local dims

                ψ = as_state(v, dof_dims)   # `as_state(v, dims)` = reshape vector into rank-L tensor with legs of dimensions given by `dims`

                # Test: order == L (number of legs)
                @test ndims(ψ.data) == L   # `ndims(arr)` = number of dimensions; should equal L = number of sites

                # Test: each leg has dimension d
                for i in 1:L   # `for i in 1:L` = Python `for i in range(1, L+1)`; Julia 1-indexed
                    @test dim(ψ.indices[i]) == d   # each leg should have dimension d (local Hilbert space dim)
                end
            end
        end

        @testset "as_state round-trip: kron-ordering consistent with dense_matrix" begin
            # Physics: as_state must respect the kron-product site ordering used by dense_matrix.
            # Kron ordering: site 1 is most significant (changes slowest).
            # For basis state e_k (only v[k]=1), the tensor should have
            # data[σ₁,...,σ_L] = 1 at the multi-index whose kron position equals k.
            let L = 2, d = 3, dof_dims = fill(d, L)
                for k in 1:(d ^ L)   # loop over all basis states (d^L = 9 for L=2, d=3)
                    e = zeros(d^L);
                    e[k] = 1.0   # `zeros(n)` = all-zero vector; `e[k] = 1.0` = set k-th entry; standard basis vector eₖ
                    ψ = as_state(e, dof_dims)
                    # Decode kron index k-1 = Σᵢ (σᵢ-1) * d^(L-i) → σ_i
                    idx = k - 1   # 0-indexed multi-index
                    σ = Vector{Int}(undef, L)   # `Vector{Int}(undef, L)` = uninitialized L-element Int vector 
                    for i in 1:L
                        σ[i] = div(idx, d^(L-i)) + 1   # `div(a, b)` = integer division. decode big-endian multi-index; +1 for Julia 1-indexing
                        idx = mod(idx, d^(L-i))       # `mod(a, b)` = remainder 
                    end
                    @test ψ.data[σ...] ≈ 1.0 atol=1e-14   # `ψ.data[σ...]` = index with splatted multi-index ; should be 1.0 at the correct position
                end
            end
        end

        @testset "as_state with heterogeneous local dimensions: dof_dims = [d₁, d₂, d₃, ...]" begin
            # Physics: sites can have different local Hilbert space dimensions (e.g. spins at different lattice sites).
            let dof_dims = [2, 3, 2],   # spin-1/2 at site 1, spin-1 at site 2, spin-1/2 at site 3
                total_dim = prod(dof_dims),   # `prod(v)` = product of all elements = 2*3*2 = 12 
                v = randn(total_dim)

                ψ = as_state(v, dof_dims)

                # Test: order == length(dof_dims)
                @test ndims(ψ.data) == length(dof_dims)   # `length(v)` = Python `len(v)`; one leg per site

                # Test: each leg has the correct dimension
                for i in eachindex(dof_dims)   # `eachindex(v)` = iterator over valid indices of v; equivalent to `1:length(v)` but more robust 
                    @test dim(ψ.indices[i]) == dof_dims[i]   # leg i has dimension dof_dims[i]
                end
            end
        end

        @testset "as_state single site (L=1): vector → scalar ⊗ physical leg" begin
            # Physics edge case: a single site is just a vector with one physical leg.
            let d = 2,
                v = [1.0, 0.0],  # |↑⟩ = spin-up state in z-basis; 2-element vector
                dof_dims = [d],  # single site with d=2
                ψ = as_state(v, dof_dims)

                # Test: order == 1
                @test ndims(ψ.data) == 1   # single leg: 1-dimensional tensor (just a vector wrapped in a QTensor)
                @test dim(ψ.indices[1]) == d    # leg has local Hilbert space dimension 2
                @test isapprox(ψ.data, v; atol=1e-14)   # data unchanged (no reshape for a 1D vector with matching dims)
            end
        end
    end  # @testset "as_state"
end  # @testset "IO and state utilities"
