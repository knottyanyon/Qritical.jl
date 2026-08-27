# §10  Exact diagonalisation — dense and sparse solve paths.
#
# `ExactDiagonalization(:ground)` uses KrylovKit Lanczos on the sparse matrix;
# `ExactDiagonalization(:full)` calls dense `eigen(Hermitian(Matrix(H)))`.
# Both paths use `matrix_repr(H, SparseFormat())` / `matrix_repr(H, DenseFormat())`
# for matrix construction.

using KrylovKit: KrylovKit   # import KrylovKit under its own namespace; provides iterative Krylov-subspace eigensolvers (Lanczos/Arnoldi)

# ----------------------------------------------------------------------------------------
# Sparse matrix from LatticeOperator
# ----------------------------------------------------------------------------------------

const _ED_MAX_DIM = 2^20   # module-level constant . caps Hilbert space at ~1 million; physics: d=2 (spin-1/2) allows at most L=20 sites with ED before memory explodes

function matrix_repr(H::LatticeOperator, ::SparseFormat)   # Julia multiple dispatch: `::SparseFormat` with no variable name means this argument is used only to SELECT this method, not inside the body (like a tag/flag); the compiler picks this method when second arg is `SparseFormat()`
    d = local_dim(H.dof)    # per-site Hilbert space dimension from the DoF object (e.g. d=2 for spin-1/2, d=4 for two-orbital models); physics: d is the local state count
    L = H.geom.L             # number of lattice sites; field access with `.` same as Python; `H.geom` is a `Chain` or other geometry struct
    D = d^L                  # total Hilbert space dimension = d^L; grows exponentially (2^10=1024, 2^20≈1M); Julia: `^` is the exponentiation operator
    D ≤ _ED_MAX_DIM || throw(
        ArgumentError(   # `||` is lazy OR: right side only executes if left is false; pattern: `condition_ok || throw(...)`; `≤` is typed with \le<Tab>
            "Hilbert space dimension $D = $(d)^$L exceeds the safety limit $_ED_MAX_DIM. " *   # Julia string interpolation: `$D` embeds variable; `$(expr)` embeds expressions; `*` concatenates strings (Python uses `+` or f-strings)
            "Use an MPS algorithm for large systems.",   # string literal split across lines by `*` concatenation
        ),
    )

    Id = sparse(Matrix{ComplexF64}(I, d, d))   # build d×d identity matrix: `I` is a lazy identity from LinearAlgebra (like np.eye but not materialised yet); `Matrix{ComplexF64}(I, d, d)` materialises it as a dense complex-float matrix; `sparse(...)` converts to CSC (Compressed Sparse Column) sparse format; result is like scipy.sparse.eye(d, format='csc', dtype=complex)
    Sp = SparseArrays.spzeros(ComplexF64, D, D)   # allocate a D×D sparse zero matrix to accumulate the full Hamiltonian; `SparseArrays.` is the module qualifier; `spzeros(T, m, n)` is like scipy.sparse.csr_matrix((m,n), dtype=T) — starts all-zero; physics: we will add each operator term to this matrix

    for lt in H.onsite   # iterate over all on-site terms; each `lt` is a `OneSiteTerm` struct with fields `.site`, `.op`, `.coupling`; physics: on-site terms are local fields like -h_i * S^z_i
        mats = [i == lt.site ? sparse(ComplexF64.(lt.op)) : Id for i in 1:L]   # list comprehension]`); `1:L` is an inclusive range; `? :` is ternary if-else; `ComplexF64.(lt.op)` broadcasts element-wise type conversion to ComplexF64`); result: L matrices, operator at its site and identity elsewhere
        Sp .+= lt.coupling .* foldl(kron, mats)   # `foldl(kron, mats)` reduces list with Kronecker product left-to-right: I⊗…⊗O_i⊗…⊗I`); `.*` broadcasts elementwise multiplication (like NumPy `*`); `.+=` broadcasts in-place addition; physics: kron-product embeds the local operator into the full D×D Hilbert space
    end

    for bt in H.bond   # iterate over two-site bond terms; each `bt` is `TwoSiteTerm` with `.i`, `.j`, `.op_i`, `.op_j`, `.coupling`; physics: bond terms are interactions like J*S^z_i⊗S^z_j
        i, j = bt.i, bt.j   # multiple assignment unpacks two values simultaneously
        mats = Vector{SparseArrays.SparseMatrixCSC{ComplexF64,Int}}(undef, L)   # pre-allocate a typed vector of L sparse matrices; `Vector{T}(undef, L)` allocates without initialising; `SparseMatrixCSC{ComplexF64,Int}` is the concrete sparse type with ComplexF64 values and Int index type; `undef` is Julia's uninitialized marker (faster than zeroing)
        for k in 1:L   # loop over all L sites to construct the Kronecker structure
            if k == i
                mats[k] = sparse(ComplexF64.(bt.op_i))   # left bond operator placed at site i; `bt.op_i` is the d×d matrix for the left site
            elseif k == j   # Julia: `elseif`
                mats[k] = sparse(ComplexF64.(bt.op_j))   # right bond operator placed at site j
            else
                mats[k] = Id   # identity at all other sites: these sites are unaffected by this bond term
            end   # Julia uses `end` to close every block — if/for/while/function/struct/module (Python uses indentation)
        end
        Sp .+= bt.coupling .* foldl(kron, mats)   # add J_ij * (I⊗…⊗O_i⊗I⊗…⊗I⊗O_j⊗I⊗…⊗I) to Hamiltonian; the Kronecker product structure ensures operators act only on their respective sites
    end

    return Sp   # return the assembled sparse Hamiltonian matrix; `return` same as Python (also optional — last expression value is implicitly returned in Julia)
end

# ----------------------------------------------------------------------------------------
# Study and algorithm types
# ----------------------------------------------------------------------------------------

"""
    GroundState

Study type indicating that the solve target is the lowest eigenvalue (ground state).
Passed as the second argument to `solve`.
"""
struct GroundState end   # zero-field singleton struct used as a dispatch tag; in Julia empty structs like this have exactly one instance and cost zero memory; physics: signals "find the ground state" to the solver dispatch

"""
    ExactDiagonalization{mode}

Algorithm type for exact diagonalisation.

  - `ExactDiagonalization(:ground)` — Lanczos on the sparse matrix; returns only
    the ground state energy and state vector. Cost: ``O(d^L \\chi_{\\text{Krylov}})``.
  - `ExactDiagonalization(:full)` — full diagonalisation via `eigen(Hermitian(...))`;
    returns all eigenvalues. Cost: ``O(d^{3L})``.
"""
struct ExactDiagonalization{mode}   # parametric struct with one type parameter `mode`; `{mode}` is like a Python generic class parameter; the mode (a Symbol like :ground or :full) is stored IN the type, not as a field — this lets the compiler pick the right `solve` method
    ExactDiagonalization(mode::Symbol) = new{mode}()   # inner constructor: `function StructName(...) = new{params}(fields...)` is Julia's way to define how the struct is built; `new{mode}()` constructs with `mode` baked into the type; `::Symbol` constrains argument to a Symbol
end

# ----------------------------------------------------------------------------------------
# Result type
# ----------------------------------------------------------------------------------------

"""
    EDResult

Return type of `solve` with `ExactDiagonalization`.

Fields:

  - `energy::Float64`           — ground-state energy (lowest eigenvalue).
  - `state::Vector{ComplexF64}` — ground-state vector (unit-norm; empty for `:full`).
  - `spectrum::Vector{Float64}` — all eigenvalues (sorted); non-empty only for `:full`.
"""
struct EDResult   # immutable data container` equivalent); all fields are set at construction and cannot be changed
    energy::Float64           # lowest eigenvalue; `::Float64` is Julia's 64-bit float (same as Python's `float` / `np.float64`); the `::` here is a field type annotation (mandatory for struct fields, unlike Python)
    state::Vector{ComplexF64} # ground-state eigenvector; `Vector{T}` is Julia's 1D array; `ComplexF64` is 64+64 bit complex
    spectrum::Vector{Float64} # all eigenvalues sorted ascending; empty (`Float64[]`) for :ground mode; non-empty for :full mode
end

# ----------------------------------------------------------------------------------------
# solve dispatch
# ----------------------------------------------------------------------------------------

"""
    solve(H, ::GroundState, ::ExactDiagonalization{:ground}) -> EDResult

Compute the ground state of `H` using the Lanczos algorithm on the sparse matrix.

Uses KrylovKit's `eigsolve` with a random initial vector; returns the lowest
eigenvalue and the corresponding eigenvector normalised to unit norm.
"""
function solve(H::LatticeOperator, ::GroundState, ::ExactDiagonalization{:ground})   # dispatch: selected when arg2 is GroundState() AND arg3 has type parameter :ground; all three args together uniquely identify this method
    M = matrix_repr(H, SparseFormat())   # build sparse Hamiltonian; sparse is used here to avoid O(D^2) memory for large D; Lanczos only needs matrix-vector products, not the full matrix
    D = size(M, 1)   # number of rows (= D = d^L); `size(M, 1)` extracts dimension 1
    v₀ = normalize(randn(ComplexF64, D))   # random complex unit vector as Krylov starting vector; `randn(ComplexF64, D)` draws D samples from CN(0,1) complex normal + 1j*rng.standard_normal(D)`); `normalize` makes ‖v₀‖=1; physics: must have nonzero overlap with GS — random vector achieves this with probability 1
    # eigsolve returns eigenvalues closest to 0 by default; use `which=:SR` for smallest real
    vals, vecs, _ = KrylovKit.eigsolve(   # returns tuple (eigenvalues, eigenvectors, convergence_info); `_` discards the third element; `KrylovKit.` is module-qualified call
        M,
        v₀,
        1,
        :SR;
        ishermitian=true,
        tol=1e-12,
        maxiter=300,   # `M` = matrix; `v₀` = start vector; `1` = want 1 eigenvalue; `:SR` = Smallest Real eigenvalue; `;` separates positional from keyword args; `ishermitian=true` triggers the more stable Lanczos algorithm; `tol=1e-12` convergence threshold; `maxiter=300` max Lanczos iterations
    )
    E = real(vals[1])   # extract ground-state energy; `real(...)` removes the negligible imaginary part (Hermitian H has real eigenvalues but KrylovKit uses ComplexF64 for generality); `vals[1]` is 1-indexed
    ψ = normalize(vecs[1])   # normalise the returned eigenvector; physics: state |ψ₀⟩ should satisfy ⟨ψ₀|ψ₀⟩=1 for expectation values to work correctly
    return EDResult(E, ψ, Float64[E])   # construct EDResult struct; `Float64[E]` creates a length-1 Float64 vector`); for :ground mode spectrum only stores the GS energy
end

"""
    solve(H, ::GroundState, ::ExactDiagonalization{:full}) -> EDResult

Full diagonalisation of `H`.

Returns all eigenvalues (sorted ascending) in `result.spectrum`.  `result.state`
is the ground-state eigenvector and `result.energy` is the smallest eigenvalue.
"""
function solve(H::LatticeOperator, ::GroundState, ::ExactDiagonalization{:full})   # :full mode: dispatched when type parameter is :full; computes ALL eigenvalues — exponentially more expensive than :ground
    M = matrix_repr(H, DenseFormat())   # build dense matrix; :full needs all eigenvectors so we use dense LAPACK (Lanczos only gives a few eigenvectors)
    F = eigen(Hermitian(M))   # full dense diagonalisation via LAPACK dsyevd; `Hermitian(M)` wraps M in a type that tells Julia it's Hermitian, enabling the symmetric solver; `F` is an `Eigen` factorisation object with `.values` (eigenvalues) and `.vectors` (column eigenvectors)
    evs = real.(F.values)   # broadcast `real` over eigenvalue vector: `f.(args)` is Julia's broadcasting notation(F.values)` or just `np.real(F.values)`); Hermitian matrices have real eigenvalues but floating-point gives tiny imaginary parts
    ψ = normalize(F.vectors[:, 1])   # ground-state eigenvector = first column; `[:, 1]` is Julia's "all rows, first column" slice; Julia is 1-indexed
    return EDResult(evs[1], ψ, sort(evs))   # `sort(evs)` returns eigenvalues in ascending order`); physics: full spectrum reveals spectral gap, degeneracies, quantum numbers
end

# ----------------------------------------------------------------------------------------
# §10.2  ED time propagation  exp(-iHt)|ψ⟩  or  exp(-τH)|ψ⟩
# ----------------------------------------------------------------------------------------

"""
    StatevectorState

A thin wrapper around a dense state vector, used as the initial condition for ED
time propagation. Think of it as the quantum-mechanics analogue of "start here":
it bundles the full ``d^L``-dimensional vector ``|\\psi_0\\rangle`` into a typed
container so that `solve` can dispatch on it correctly.

Construct one with [`as_statevector`](@ref) rather than calling the constructor
directly — that helper also handles element-type conversion.

# Fields

  - `v::Vector{ComplexF64}`: the initial state in the Kronecker-product (big-endian)
    basis ordering. Index ``k`` corresponds to the computational basis state
    ``|\\sigma_1, \\sigma_2, \\ldots, \\sigma_L\\rangle`` with
    ``k = 1 + \\sum_{i=1}^{L} (\\sigma_i - 1)\\, d^{L-i}``.

# See also

[`as_statevector`](@ref), [`EDTimeResult`](@ref),
`solve(::LatticeOperator, ::StatevectorState, ::ExactDiagonalization{:time}, ::ConstantProtocol)`
"""
struct StatevectorState   # dispatch-tag wrapper for the initial state vector; physics: holds |ψ₀⟩ as a d^L-dimensional complex vector in the computational basis
    v::Vector{ComplexF64}   # the state vector; index k encodes a computational basis state via big-endian kron ordering (site 1 is most significant)
end

"""
    as_statevector(v) -> StatevectorState

Wrap a dense vector `v` as the initial state for ED time propagation.

This is the standard entry point when you have a plain `Vector` (say, the
ground-state vector from [`EDResult`](@ref)) and want to hand it to
`solve` with `ExactDiagonalization(:time)`. The helper converts `v` to
`ComplexF64` so you do not have to think about element types.

The vector must be ordered in the big-endian Kronecker-product basis used
by [`matrix_repr`](@ref): component ``k`` of the returned
[`StatevectorState`](@ref) corresponds to the computational basis state
``|\\sigma_1, \\sigma_2, \\ldots, \\sigma_L\\rangle`` where

```math
k = 1 + \\sum_{i=1}^{L} (\\sigma_i - 1)\\, d^{L-i}.
```

In practice, any vector you obtain from [`EDResult`](@ref) or from
[`neel_state`](@ref) is already in this ordering.

# Arguments

  - `v::AbstractVector`: a real or complex state vector of length ``d^L``.

# Returns

  - `StatevectorState`: the wrapped vector, element-type promoted to
    `ComplexF64`.

# Examples

```jldoctest
julia> using Qritical, LinearAlgebra

julia> H = Heisenberg(Chain(4); J=1.0);

julia> v = zeros(ComplexF64, 2^4);
       v[7] = 1.0;   # a basis state of the 4-site chain

julia> sv = as_statevector(v);

julia> p = ConstantProtocol(RealTime(), 0.1, 20, H);   # axis, dt, nsteps, Hamiltonian

julia> result = solve(H, sv, ExactDiagonalization(:time), p);

julia> round(norm(result.state); digits=10)   # real-time propagation is unitary
1.0
```
"""
as_statevector(v::AbstractVector) = StatevectorState(convert(Vector{ComplexF64}, v))   # one-line function (no `function...end` needed for single-expression bodies); `AbstractVector` accepts any 1D array subtype (Vector, SubArray, etc.); `convert(Vector{ComplexF64}, v)` promotes element type to ComplexF64`)

"""
    EDTimeResult

The result returned by `solve` with `ExactDiagonalization(:time)`. It holds the
propagated state and the total time elapsed, which you can use to compute
expectation values ``\\langle O(t) \\rangle`` or to continue evolving further.

# Fields

  - `state::Vector{ComplexF64}`: the final state vector ``|\\psi(T)\\rangle``.
    For real-time propagation this is exactly unit-norm (``\\|\\psi(T)\\| = 1``).
    For imaginary-time propagation it is **not** normalised — call
    `normalize(result.state)` to get the unit-norm ground-state approximation.
  - `time::Float64`: the total propagation time ``T = \\Delta t \\cdot N_{\\text{steps}}``,
    in whatever units your Hamiltonian coupling constants carry.

# See also

[`StatevectorState`](@ref), [`as_statevector`](@ref),
`solve(::LatticeOperator, ::StatevectorState, ::ExactDiagonalization{:time}, ::ConstantProtocol)`
"""
struct EDTimeResult   # immutable result container for time-domain ED; physics: holds |ψ(T)⟩ and the total time T
    state::Vector{ComplexF64}   # final propagated state |ψ(T)⟩; unit-norm for real time (unitary evolution), NOT normalised for imaginary time
    time::Float64               # total propagation time T = dt * nsteps
end

"""
    solve(H, sv, ::ExactDiagonalization{:time}, p) -> EDTimeResult

Propagate the initial state `sv` under the Hamiltonian `H` for the total time
given by protocol `p`, using exact eigendecomposition — no Trotter splitting,
no truncation.

## What this computes

For **real-time** propagation (`p` carries `RealTime`), this applies the
unitary time-evolution operator:

```math
|\\psi(T)\\rangle = e^{-i H T}\\,|\\psi_0\\rangle,
\\qquad T = \\Delta t \\cdot N_{\\text{steps}}.
```

The norm is exactly preserved: ``\\|\\psi(T)\\| = \\|\\psi_0\\|``.

For **imaginary-time** propagation (`p` carries `ImaginaryTime`), the
non-unitary projector is applied instead:

```math
|\\psi(\\tau)\\rangle = e^{-H\\tau}\\,|\\psi_0\\rangle \\quad (\\text{unnormalised}).
```

Each eigencomponent decays at its own rate ``e^{-E_n \\tau}``, so the ground
state is amplified relative to excited states. The rate of convergence is set
by the spectral gap ``\\Delta``: components above the ground state are suppressed
by a factor ``e^{-\\Delta\\tau}``. **Normalisation is not applied automatically**
for imaginary time — use `normalize(result.state)` afterwards.

## The eigendecomposition trick

The key idea is to diagonalise ``H = V \\Lambda V^\\dagger`` **once** and then
evaluate any time ``T`` cheaply:

```math
e^{-i H T}\\,|\\psi_0\\rangle
= V\\,\\mathrm{diag}\\bigl(e^{-i\\lambda_1 T},\\, \\ldots,\\, e^{-i\\lambda_D T}\\bigr)\\,V^\\dagger\\,|\\psi_0\\rangle.
```

The projection ``V^\\dagger |\\psi_0\\rangle`` costs ``O(d^{2L})``, and so does
the back-transform ``V (\\cdots)``. Diagonalisation itself costs ``O(d^{3L})``,
but you pay that once no matter how long ``T`` is or how many times you ask for
observables.

## Cross-validation use case

Because there is no Trotter error here, ED time evolution is the ideal reference
for validating TEBD. Run both on the same small system (``L \\leq 10``) and
compare ``\\langle O(t) \\rangle`` at several times: any disagreement is Trotter
error, and its magnitude should scale as ``(\\Delta t)^p`` for a ``p``-th order
Trotter formula.

# Arguments

  - `H::LatticeOperator`: the Hamiltonian. Must fit within the ``2^{20}`` Hilbert-space
    guard enforced by [`matrix_repr`](@ref) (the same limit applies to both the dense
    and sparse storage formats).
  - `sv::StatevectorState`: the initial state, constructed via
    [`as_statevector`](@ref).
  - `::ExactDiagonalization{:time}`: selects this exact-propagation path.
  - `p::ConstantProtocol`: the evolution schedule. The `axis` field of `p`
    selects `RealTime` or `ImaginaryTime`; `p.dt` and `p.nsteps` together
    determine ``T = p.dt \\times p.nsteps``.

# Returns

  - `EDTimeResult`: holds `result.state` (the propagated vector) and
    `result.time` (the total time ``T``).

# Examples

```julia
julia> norm(r.state)   # always 1.0 for real time
# Real-time evolution of the ground state (should stay put — it's an eigenstate)
```

# See also

[`StatevectorState`](@ref), [`as_statevector`](@ref), [`EDTimeResult`](@ref),
[`ConstantProtocol`](@ref), [`RealTime`](@ref), [`ImaginaryTime`](@ref)
"""
function solve(
    H::LatticeOperator,
    sv::StatevectorState,
    ::ExactDiagonalization{:time},
    p::ConstantProtocol,   # multi-line signature: Julia allows splitting across lines; dispatched when arg3 is ExactDiagonalization{:time} and arg4 is any ConstantProtocol; `p::ConstantProtocol` gives `p` a name so we can access p.axis, p.dt, p.nsteps
)
    M = matrix_repr(H, DenseFormat())   # build dense Hamiltonian; time propagation needs all eigenvectors for the basis transform, so dense is appropriate here
    F = eigen(Hermitian(M))   # diagonalise H = V Λ V†: `F.values` are eigenvalues λ_k, `F.vectors` columns are eigenvectors φ_k; this O(D^3) cost is paid once and amortises over arbitrary T
    T = total_time(p)   # p.dt * p.nsteps; total propagation time T; physics: whether T is a real time t or imaginary time τ depends on p.axis

    if p.axis isa RealTime   # `isa` checks if a value's type is (or subtypes) the given type`); physics: real time → unitary evolution preserving norm
        phases = exp.(-im .* T .* real.(F.values))   # e^{-iE_k T} for each eigenvalue; `exp.()` broadcasts exp over an array)`); `.-` and `.*` are broadcasting subtraction/multiplication; `im` is the imaginary unit; `real.()` removes numerical noise in imaginary part of eigenvalues
    else  # ImaginaryTime
        phases = exp.(-T .* real.(F.values))   # e^{-E_k τ} real damping factor; physics: each eigencomponent decays proportional to its energy — the GS component (lowest E_k) decays slowest and dominates at large τ
    end

    coefs = F.vectors' * sv.v          # project initial state onto eigenbasis: c_k = ⟨φ_k|ψ₀⟩ = (V†)_k · v; `F.vectors'` is the conjugate transpose (adjoint operator) of V.T`); `*` is matrix-vector multiply; result is a D-vector of expansion coefficients
    ψ_final = F.vectors * (phases .* coefs)  # back-transform: V · (e^{-iHT} ⊙ c) = Σ_k e^{-iE_k T} c_k |φ_k⟩; `phases .* coefs` elementwise scales each eigenvector coefficient; then `F.vectors *` transforms back to original basis; physics: this is the exact solution of the Schrödinger equation (no Trotter error)
    return EDTimeResult(ψ_final, T)   # wrap results; for real time ‖ψ_final‖=1 exactly (unitary evolution); for imaginary time norm decreases — call normalize() before using the state
end
