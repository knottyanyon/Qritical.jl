# §10  Exact diagonalisation — dense and sparse solve paths.
#
# `ExactDiagonalization(:ground)` uses KrylovKit Lanczos on the sparse matrix;
# `ExactDiagonalization(:full)` calls dense `eigen(Hermitian(Matrix(H)))`.
# Both paths share `sparse(H)` as the entry point for matrix construction.

import KrylovKit

# ─────────────────────────────────────────────────────────────────────────────
# Sparse matrix from Operator
# ─────────────────────────────────────────────────────────────────────────────

const _ED_MAX_DIM = 2^20   # refuse Hilbert spaces larger than ~1 M

"""
    sparse(H::Operator) -> SparseMatrixCSC{ComplexF64}

Build the sparse ``d^L \\times d^L`` Hamiltonian matrix for the operator `H`.

The matrix is assembled by sparse Kronecker products of local operators onto the
full Hilbert space ``\\mathcal{H} = \\bigotimes_{i=1}^{L} \\mathcal{h}_i``.

Raises `ArgumentError` if ``d^L > 2^{20} \\approx 10^6`` (refuse to silently allocate
multi-GB arrays).
"""
function sparse(H::Operator)
    d = local_dim(H.dof)
    L = H.geom.L
    D = d^L
    D ≤ _ED_MAX_DIM || throw(ArgumentError(
        "Hilbert space dimension $D = $(d)^$L exceeds the safety limit $_ED_MAX_DIM. " *
        "Use an MPS algorithm for large systems."))

    Id = sparse(Matrix{ComplexF64}(I, d, d))
    Sp = SparseArrays.spzeros(ComplexF64, D, D)

    for lt in H.onsite
        mats = [i == lt.site ? sparse(ComplexF64.(lt.op)) : Id for i in 1:L]
        Sp .+= lt.coupling .* foldl(kron, mats)
    end

    for bt in H.bond
        i, j = bt.i, bt.j
        mats = Vector{SparseArrays.SparseMatrixCSC{ComplexF64, Int}}(undef, L)
        for k in 1:L
            if k == i
                mats[k] = sparse(ComplexF64.(bt.op_i))
            elseif k == j
                mats[k] = sparse(ComplexF64.(bt.op_j))
            else
                mats[k] = Id
            end
        end
        Sp .+= bt.coupling .* foldl(kron, mats)
    end

    return Sp
end

# ─────────────────────────────────────────────────────────────────────────────
# Study and algorithm types
# ─────────────────────────────────────────────────────────────────────────────

"""
    GroundState

Study type indicating that the solve target is the lowest eigenvalue (ground state).
Passed as the second argument to `solve`.
"""
struct GroundState end

"""
    ExactDiagonalization{mode}

Algorithm type for exact diagonalisation.

- `ExactDiagonalization(:ground)` — Lanczos on the sparse matrix; returns only
  the ground state energy and state vector. Cost: ``O(d^L \\chi_{\\text{Krylov}})``.
- `ExactDiagonalization(:full)` — full diagonalisation via `eigen(Hermitian(...))`;
  returns all eigenvalues. Cost: ``O(d^{3L})``.
"""
struct ExactDiagonalization{mode}
    ExactDiagonalization(mode::Symbol) = new{mode}()
end

# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

"""
    EDResult

Return type of `solve` with `ExactDiagonalization`.

Fields:
- `energy::Float64`           — ground-state energy (lowest eigenvalue).
- `state::Vector{ComplexF64}` — ground-state vector (unit-norm; empty for `:full`).
- `spectrum::Vector{Float64}` — all eigenvalues (sorted); non-empty only for `:full`.
"""
struct EDResult
    energy::Float64
    state::Vector{ComplexF64}
    spectrum::Vector{Float64}
end

# ─────────────────────────────────────────────────────────────────────────────
# solve dispatch
# ─────────────────────────────────────────────────────────────────────────────

"""
    solve(H, ::GroundState, ::ExactDiagonalization{:ground}) -> EDResult

Compute the ground state of `H` using the Lanczos algorithm on the sparse matrix.

Uses KrylovKit's `eigsolve` with a random initial vector; returns the lowest
eigenvalue and the corresponding eigenvector normalised to unit norm.
"""
function solve(H::Operator, ::GroundState, ::ExactDiagonalization{:ground})
    M   = sparse(H)
    D   = size(M, 1)
    v₀  = normalize(randn(ComplexF64, D))
    # eigsolve returns eigenvalues closest to 0 by default; use `which=:SR` for smallest real
    vals, vecs, _ = KrylovKit.eigsolve(M, v₀, 1, :SR; ishermitian=true, tol=1e-12, maxiter=300)
    E   = real(vals[1])
    ψ   = normalize(vecs[1])
    EDResult(E, ψ, Float64[E])
end

"""
    solve(H, ::GroundState, ::ExactDiagonalization{:full}) -> EDResult

Full diagonalisation of `H`.

Returns all eigenvalues (sorted ascending) in `result.spectrum`.  `result.state`
is the ground-state eigenvector and `result.energy` is the smallest eigenvalue.
"""
function solve(H::Operator, ::GroundState, ::ExactDiagonalization{:full})
    M   = dense_matrix(H)
    F   = eigen(Hermitian(M))
    evs = real.(F.values)
    ψ   = normalize(F.vectors[:, 1])
    EDResult(evs[1], ψ, sort(evs))
end

# ─────────────────────────────────────────────────────────────────────────────
# §10.2  ED time propagation  exp(-iHt)|ψ⟩  or  exp(-τH)|ψ⟩
# ─────────────────────────────────────────────────────────────────────────────

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
`solve(::Operator, ::StatevectorState, ::ExactDiagonalization{:time}, ::ConstantProtocol)`
"""
struct StatevectorState
    v::Vector{ComplexF64}
end

"""
    as_statevector(v) -> StatevectorState

Wrap a dense vector `v` as the initial state for ED time propagation.

This is the standard entry point when you have a plain `Vector` (say, the
ground-state vector from [`EDResult`](@ref)) and want to hand it to
`solve` with `ExactDiagonalization(:time)`. The helper converts `v` to
`ComplexF64` so you do not have to think about element types.

The vector must be ordered in the big-endian Kronecker-product basis used
by [`dense_matrix`](@ref): component ``k`` of the returned
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
```julia
julia> gs = solve(H, GroundState(), ExactDiagonalization(:ground));
julia> sv = as_statevector(gs.state);
julia> result = solve(H, sv, ExactDiagonalization(:time), ConstantProtocol{RealTime}(0.1, 20));
```
"""
as_statevector(v::AbstractVector) = StatevectorState(convert(Vector{ComplexF64}, v))

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
`solve(::Operator, ::StatevectorState, ::ExactDiagonalization{:time}, ::ConstantProtocol)`
"""
struct EDTimeResult
    state::Vector{ComplexF64}
    time::Float64
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
- `H::Operator`: the Hamiltonian. Must fit within the ``2^{20}`` Hilbert-space
  guard checked by [`sparse`](@ref) internally.
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
julia> # Real-time evolution of the ground state (should stay put — it's an eigenstate)
julia> gs = solve(H, GroundState(), ExactDiagonalization(:ground));
julia> sv = as_statevector(gs.state);
julia> p  = ConstantProtocol{RealTime}(0.05, 40);   # T = 2.0
julia> r  = solve(H, sv, ExactDiagonalization(:time), p);
julia> norm(r.state)   # always 1.0 for real time
1.0
```

# See also
[`StatevectorState`](@ref), [`as_statevector`](@ref), [`EDTimeResult`](@ref),
[`ConstantProtocol`](@ref), [`RealTime`](@ref), [`ImaginaryTime`](@ref)
"""
function solve(H::Operator, sv::StatevectorState, ::ExactDiagonalization{:time},
               p::ConstantProtocol)
    M   = dense_matrix(H)
    F   = eigen(Hermitian(M))
    T   = total_time(p)   # p.dt * p.nsteps

    if p.axis isa RealTime
        phases = exp.(-im .* T .* real.(F.values))
    else  # ImaginaryTime
        phases = exp.(-T .* real.(F.values))
    end

    coefs    = F.vectors' * sv.v          # project onto eigenbasis
    ψ_final  = F.vectors * (phases .* coefs)  # evolve + back-transform
    EDTimeResult(ψ_final, T)
end
