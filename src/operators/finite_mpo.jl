# §6.2–6.3  Finite MPO — construction and expectation value.
#
# An MPO for a chain of L sites is a sequence of rank-4 tensors
# W_i[α, σ_out, σ_in, β], where α/β are the auxiliary (bond) indices and
# σ_out/σ_in are the physical (ket/bra) indices.
#
# Construction uses the finite-state-machine (W-matrix) method:
#   - Index 1       : "done" state (right boundary — sum accumulated)
#   - Indices 2..K+1: open channels, one per unique NN bond operator type
#   - Index K+2     : "start" state (left boundary — haven't begun)
# Bond dimension: χ = 2 + K.

"""
    FiniteMPO

A Matrix Product LatticeOperator (MPO) for a finite chain of `L` sites.

An MPO represents a linear operator on the full Hilbert space
``\\mathcal{H} = \\bigotimes_{i=1}^{L} \\mathcal{h}_i`` as a product of rank-4
tensors, one per site.  The tensor at site ``i`` has the index layout

```math
W_i[\\alpha,\\, \\sigma_{\\text{out}},\\, \\sigma_{\\text{in}},\\, \\beta]
```

where:

  - ``\\alpha`` — left auxiliary (MPO bond) index of dimension ``\\chi_L``
  - ``\\sigma_{\\text{out}}`` — ket-side (output) physical index of dimension ``d``:
    the contravariant ``\\sigma'`` of the operator coefficient
    ``O^{\\sigma'}_{\\sigma}`` in ``\\hat{O} = |\\sigma'\\rangle O^{\\sigma'}_{\\sigma} \\langle\\sigma|``
    (`Upper` in the covariant convention)
  - ``\\sigma_{\\text{in}}``  — bra-side (input) physical index of dimension ``d``:
    the covariant ``\\sigma`` that contracts the input state's ``\\psi^{\\sigma}``
    (`Lower` in the covariant convention)
  - ``\\beta`` — right auxiliary (MPO bond) index of dimension ``\\chi_R``

The stored tensors are raw `Array`s (no variance tags); the covariant reading
above governs how they contract with `QTensor` legs in [`expect`](@ref) and
[`apply_mpo`](@ref).

The full operator matrix is recovered by contracting all ``W_i`` along their
auxiliary indices with the boundary conditions ``\\alpha_0 = \\beta_L = 1``.

Boundary tensors have dimension 1 on their outer auxiliary side, so the stored
tensors at sites 1 and ``L`` have shapes ``(1, d, d, \\chi)`` and
``(\\chi, d, d, 1)`` respectively.

Construct a `FiniteMPO` from an [`LatticeOperator`](@ref) via [`MPO`](@ref).
Evaluate expectation values via [`expect`](@ref).

# Fields

  - `tensors::Vector{Array{ComplexF64,4}}` — one ``(\\chi_L, d, d, \\chi_R)`` tensor
    per site.
  - `d::Int` — local physical dimension (same for all sites).
  - `L::Int` — number of sites.

See also: [`MPO`](@ref), [`expect`](@ref), [`LatticeOperator`](@ref)
"""
struct FiniteMPO   # immutable container for the MPO representation; physics: an MPO is the operator analogue of an MPS — a chain of rank-4 tensors connected by auxiliary bonds
    tensors::Vector{Array{ComplexF64,4}}   # (χ_L, d_out, d_in, χ_R) per site  # length-L vector of rank-4 complex arrays; `Array{ComplexF64,4}` is a 4D array of ComplexF64 
    d::Int   # local physical dimension, same at every site (d=2 for spin-1/2)
    L::Int   # chain length; stored for convenience so you don't need length(tensors) everywhere
end

# ----------------------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------------------

# Two operator matrices are "the same type" if they are approximately equal.
_ops_eq(A, B) = size(A) == size(B) && maximum(abs, A - B) < 1e-12   # check if two matrices represent the same operator up to numerical noise; `size(A) == size(B)` checks dimensions match first; `maximum(abs, A - B)` computes max(|A-B|) in one call; used by FSM builder to identify equal operator types across bond terms

# Collect unique (op_i, op_j) pairs from the bond term list (preserving order).
function _bond_types(bond_terms)   # find the set of distinct bond operator type pairs; each type defines one FSM channel; number of types K determines MPO bond dimension χ=2+K
    types = Tuple{Matrix{ComplexF64},Matrix{ComplexF64}}[]   # empty vector of (Matrix, Matrix) tuples; `Tuple{T1,T2}[]` is a typed empty vector 
    for bt in bond_terms   # loop over all bond terms
        pair = (ComplexF64.(bt.op_i), ComplexF64.(bt.op_j))   # extract the (op_i, op_j) pair for this term and convert to ComplexF64; `ComplexF64.(...)` broadcasts element-wise type conversion
        if !any(t -> _ops_eq(t[1], pair[1]) && _ops_eq(t[2], pair[2]), types)   # `!any(predicate, collection)` is true if NO existing type matches this pair; `t[1]` and `t[2]` access tuple elements (1-indexed!); `&&` is short-circuit AND
            push!(types, pair)   # `push!(vec, elem)` appends to a vector IN PLACE (the `!` suffix is Julia's convention for mutating functions — Python equivalent: `types.append(pair)`); only add if this operator pair is new (not seen before)
        end
    end
    return types   # list of unique (op_i, op_j) type pairs; length K = number of FSM channels
end

# ----------------------------------------------------------------------------------------
# MPO constructor from LatticeOperator
# ----------------------------------------------------------------------------------------

"""
    MPO(H::LatticeOperator) -> FiniteMPO

Build a [`FiniteMPO`](@ref) from the term list of `H` using the
finite-state-machine (FSM) W-matrix method.

The FSM idea is elegant: think of building the operator left-to-right as a
finite automaton that carries a "state" summarising what has been started but
not yet finished.  The auxiliary index of the MPO tensor at site ``i`` labels
the current FSM state.  There are three kinds of state:

  - **"done" state** (auxiliary index 1) — we have already accumulated a
    complete local operator string and just need to propagate the identity to
    the right.  The ``W`` entry `done → done` is the identity.
  - **channel ``k``** (auxiliary indices ``2, \\ldots, K+1``) — the left operator
    ``O^{(k)}_i`` of bond type ``k`` was placed at some site to the left and we
    are now carrying it to its right partner.  The ``W`` entry `channel k → done`
    closes the channel by placing ``O^{(k)}_j``.
  - **"start" state** (auxiliary index ``K+2``) — we have not placed any operator
    yet.  On-site terms and new bond openings come from this state.

For an `LatticeOperator` with ``K`` distinct nearest-neighbour bond-operator pairs
``(O^{(k)}_i, O^{(k)}_j)`` the interior bond dimension is

```math
\\chi = 2 + K.
```

The boundary tensors at sites 1 and ``L`` are sliced to dimension 1 on their
outer auxiliary side (only the "start" row is reachable from the left boundary;
only the "done" column is reachable from the right boundary).

**Special case:** if `H` has no terms at all (e.g. from [`identity_operator`](@ref)),
the result is a bond-dimension-1 all-identity MPO with ``\\chi = 1``.

# Arguments

  - `H::LatticeOperator` — the operator to convert; produced by e.g. [`XXZ`](@ref),
    [`Heisenberg`](@ref), [`Ising`](@ref), or any of the observable constructors.

# Returns

  - [`FiniteMPO`](@ref) with tensors of shape ``(\\chi_L, d, d, \\chi_R)`` per site.

# Examples

```jldoctest
julia> g = Chain(4);

julia> H = Heisenberg(g);

julia> W = MPO(H);

julia> W.L
4

julia> W.d
2

julia> size(W.tensors[2])   # interior site: full χ on both sides
(5, 2, 2, 5)
```    # Reject periodic geometries: the FSM builder assumes bonds are ordered left-to-right
```
"""
function MPO(H::LatticeOperator)   # construct a FiniteMPO from a LatticeOperator using the W-matrix FSM method; physics: the FSM encodes operator ordering as an automaton that walks left-to-right through the chain
    # Reject periodic geometries: the FSM builder assumes bonds are ordered left-to-right
    # (i < j).  The wrap bond (L,1) has i > j and would corrupt the channel logic.  Fixes #79.
    if any(bt -> bt.i > bt.j, H.bond)   # check for reverse bonds (periodic chain); `any(predicate, collection)` = Python `any(bt.i > bt.j for bt in H.bond)`
        throw(
            ArgumentError(   # `throw(ArgumentError(...))` raises a typed exception. `ArgumentError` indicates a bad function argument
                "MPO does not support periodic boundary conditions: bond " *
                "$(first(bt for bt in H.bond if bt.i > bt.j)) has i > j.  " *   # `first(gen)` extracts the first element of a generator. string interpolation with `$(...)`
                "Use an open-boundary Chain instead.",
            ),
        )
    end
    L = H.geom.L   # chain length
    d = local_dim(H.dof)   # local dimension
    Id = Matrix{ComplexF64}(I, d, d)   # d×d identity matrix; used for pass-through entries in the W tensor

    # Special case: empty operator (no terms) → bond-dim-1 all-identity MPO.
    # This represents I₁⊗I₂⊗…⊗I_L; its expectation value is ‖ψ‖².
    if isempty(H.onsite) && isempty(H.bond)   # `isempty(collection)` checks if length is 0. both term lists empty → identity operator
        tensors = [reshape(copy(Id), 1, d, d, 1) for _ in 1:L]   # for each site: reshape Id from (d,d) to (1,d,d,1) — a χ=1 MPO tensor; `copy(Id)` makes a fresh copy; `reshape(A, dims...)` same as numpy but column-major; `_` discards loop variable 
        return FiniteMPO(tensors, d, L)   # early return with χ=1 all-identity MPO
    end

    # Identify the distinct bond operator types — these define the FSM channels.
    btypes = _bond_types(H.bond)   # collect unique (op_i, op_j) type pairs; K = length(btypes) unique types
    K = length(btypes)   # number of distinct bond operator types = number of FSM channels
    χ = 2 + K          # full interior bond dimension  # FSM states: 1 "done" + K channels + 1 "start" = K+2 = χ
    # Index convention:
    #   1     = "done"  (right boundary state)
    #   2..K+1 = channel k (bond type k is open)
    #   K+2   = "start" (left boundary state)

    tensors = Vector{Array{ComplexF64,4}}(undef, L)   # pre-allocate L uninitialized 4D arrays; `Vector{T}(undef, n)` is like `[None]*n` but typed; each will be filled in the loop below

    for i in 1:L   # build the W tensor at each site i
        W = zeros(ComplexF64, χ, d, d, χ)   # χ×d×d×χ tensor initialised to zero; `zeros(T, dims...)` = Python `np.zeros(dims, dtype=complex)`; shape is (MPO_bond_left, phys_out, phys_in, MPO_bond_right)

        # Identity pass-through for both boundary states.
        W[1, :, :, 1] = Id     # done → done  # W[done_in, :, :, done_out] = I; propagates "already accumulated a complete term" by doing nothing to the physical legs; `[:, :]` selects all elements of the d×d physical block
        W[K + 2, :, :, K + 2] = Id     # start → start  # W[start_in, :, :, start_out] = I; propagates "haven't started yet" by doing nothing; this allows the FSM to reach the term starting site

        # Channel carry: W[k+1,:,:,k+1]=Id is needed at site i only when channel k is
        # "in flight" — i.e., some bond of type k has bt.i < i < bt.j.  For NN bonds
        # (bt.j = bt.i+1) there are no such intermediate sites, so no carry is needed
        # and adding it unconditionally would allow channels to stay open past their
        # closing site and produce spurious contributions.  Fixes #78.
        for (k, (Ak, Bk)) in enumerate(btypes)   # iterate over all K channel types; `enumerate(collection)` returns (index, value) pairs — same as Python enumerate; `(Ak, Bk)` unpacks the operator pair tuple
            in_flight = any(H.bond) do bt   # `any(collection) do element ... end` is Julia's block-form of `any(predicate, collection)`; the `do bt` creates an anonymous function over `bt`; `any(H.bond) do bt...` checks if any bond of type k is "in flight" at site i
                return bt.i < i < bt.j &&   # `a < b < c` is chained comparison (same as Python!); true only if i is strictly between bt.i and bt.j
                       _ops_eq(ComplexF64.(bt.op_i), Ak) &&   # check the left operator matches channel k
                       _ops_eq(ComplexF64.(bt.op_j), Bk)   # and the right operator matches; all three conditions must hold
            end
            in_flight && (W[k + 1, :, :, k + 1] = Id)   # `condition && expr` evaluates `expr` only if condition is true (short-circuit AND used as one-line conditional); set channel carry: W[channel_in, :, :, channel_out] = I only when channel k is in-flight between two non-adjacent sites
        end

        # Onsite terms: start → done
        for lt in H.onsite   # iterate over on-site terms to place them in W
            lt.site == i || continue   # skip if this term is not at site i; `condition || continue` = `if !condition; continue; end`
            W[K + 2, :, :, 1] .+= lt.coupling .* ComplexF64.(lt.op)   # add coupling × operator to W[start→done] block; `K+2` = "start" row, `1` = "done" column; `.+=` broadcasts in-place addition over the d×d physical block; physics: on-site term fired from "start" immediately closes to "done"
        end

        # Open channel k at site i (bond starting here): start → channel k
        for (k, (Ak, _)) in enumerate(btypes)   # `_` discards the right operator Bk (we only need the left operator type here)
            for bt in H.bond   # find the bond term that opens channel k at site i
                bt.i == i || continue   # skip if bond doesn't start at site i
                _ops_eq(ComplexF64.(bt.op_i), Ak) || continue   # skip if left operator doesn't match channel k type
                W[K + 2, :, :, k + 1] .+= bt.coupling .* ComplexF64.(bt.op_i)   # open channel: W[start, :, :, channel_k] += J * O_i; physics: records that we've placed the left operator of bond type k at site i, carrying the coupling J into the channel
                break   # at most one bond starts at each site per type (NN chain)  # `break` exits the innermost for loop (same as Python); for NN chains each site has at most one bond of each type starting here
            end
        end

        # Close channel k at site i (bond ending here): channel k → done
        for (k, (_, Bk)) in enumerate(btypes)   # now we need the right operator Bk; `_` discards Ak
            for bt in H.bond   # find the bond term that closes channel k at site i
                bt.j == i || continue   # skip if bond doesn't end at site i
                _ops_eq(ComplexF64.(bt.op_j), Bk) || continue   # skip if right operator doesn't match channel k
                W[k + 1, :, :, 1] .+= ComplexF64.(bt.op_j)   # close channel: W[channel_k, :, :, done] += O_j; note: the coupling J was already absorbed when OPENING the channel, so we don't multiply by coupling here
                break   # at most one bond ends at each site per type
            end
        end

        tensors[i] = W   # store the completed W tensor for site i
    end

    # Compress boundary bond dimensions: site 1 is only ever entered from
    # the "start" row; site L is only ever exited to the "done" column.
    tensors[1] = tensors[1][(K + 2):(K + 2), :, :, :]   # (1, d, d, χ)  # slice only the "start" row: at the left boundary the FSM always starts in state K+2; `(K+2):(K+2)` is a range of length 1 that keeps the dimension (unlike `K+2` scalar indexing which would drop it); result shape: (1, d, d, χ)
    tensors[L] = tensors[L][:, :, :, 1:1]        # (χ, d, d, 1)  # slice only the "done" column: at the right boundary only "done" results contribute; `1:1` range keeps the dimension; result shape: (χ, d, d, 1)

    return FiniteMPO(tensors, d, L)   # construct and return the FiniteMPO
end

# ----------------------------------------------------------------------------------------
# Expectation value ⟨ψ|O|ψ⟩  (§6.3)
# ----------------------------------------------------------------------------------------

"""
    expect(ψ::FiniteMPS, O::FiniteMPO) -> ComplexF64

Compute the expectation value ``\\langle \\psi | O | \\psi \\rangle`` by a single
left-to-right sweep of a three-legged environment tensor.

The algorithm is the standard MPO–MPS "zipper" contraction.  Starting from a
``1 \\times 1 \\times 1`` scalar environment on the left boundary, at each site
``i`` the environment is updated by contracting with the MPS site tensor (bra
and ket) and the MPO tensor:

```math
E'[\\alpha', a', \\beta'] = \\sum_{\\alpha, a, \\beta, \\sigma, \\sigma'}
  E[\\alpha, a, \\beta]\\,
  \\overline{A_i[\\alpha, \\sigma, \\alpha']}\\,
  W_i[a, \\sigma, \\sigma', a']\\,
  A_i[\\beta, \\sigma', \\beta'],
```

where ``A_i`` is the MPS tensor at site ``i`` and bar denotes complex conjugate.

The environment tensor has shape ``(\\chi_{\\text{bra}}, \\chi_{\\text{mpo}}, \\chi_{\\text{ket}})``
throughout the sweep.  After all ``L`` sites the environment is a ``1 \\times 1 \\times 1``
scalar equal to ``\\langle \\psi | O | \\psi \\rangle``.

The per-site cost is ``O(\\chi^2 d \\chi_{\\text{mpo}})`` where ``\\chi`` is the MPS
bond dimension; the total cost for the full sweep is
``O(L \\chi^2 d \\chi_{\\text{mpo}})``.

# Arguments

  - `ψ::FiniteMPS`   — the MPS state; does not need to be normalised.
  - `O::FiniteMPO`   — the MPO operator; must have the same `L` and `d` as `ψ`.

# Returns

  - `ComplexF64` — the expectation value ``\\langle \\psi | O | \\psi \\rangle``.
    For Hermitian operators and real states this should have a negligible imaginary
    part.

# Examples

```jldoctest
julia> using LinearAlgebra

julia> g = Chain(2);

julia> H = Heisenberg(g);

julia> W = MPO(H);

julia> ψ = to_mps(normalize([1.0, 0.0, 0.0, -1.0] / √2));

julia> isapprox(real(expect(ψ, W)), -0.75; atol=1e-10)
true
```
"""
function expect(ψ::FiniteMPS, O::FiniteMPO)   # compute ⟨ψ|O|ψ⟩ by the MPS-MPO zipper contraction; sweeps left-to-right growing a 3-legged environment tensor
    L = O.L   # chain length
    # Initialise: 1×1×1 environment = scalar 1
    env = ones(ComplexF64, 1, 1, 1)   # (χ_bra, χ_mpo, χ_ket)  # initial 1×1×1 environment tensor representing the left boundary; `ones(T, dims...)` = Python `np.ones(dims, dtype=complex)`; shape (χ_bra, χ_mpo, χ_ket) grows as we sweep right

    for i in 1:L   # sweep left-to-right through all L sites
        A = ψ.tensors[i].data   # (χ_L, d, χ_R)  # MPS tensor at site i; `.data` extracts raw array from QTensor wrapper; shape (left bond, physical, right bond)
        W = O.tensors[i]        # (χ_mpo_L, d_out, d_in, χ_mpo_R)  # MPO tensor at site i; shape (left MPO bond, phys out, phys in, right MPO bond)
        # Contract: env[α,a,β] conj(A[α,σ,α']) W[a,σ,σ',a'] A[β,σ',β'] → new_env[α',a',β']
        @tensor new_env[α′, a′, β′] :=   # `@tensor` macro (TensorOperations.jl): Einstein summation; repeated indices are contracted (summed over); `:=` defines new_env (not in-place); `α′` is α-prime (typed as \alpha<Tab>\prime<Tab>) — Julia Unicode variable names work!
            env[α, a, β] * conj(A[α, σ, α′]) * W[a, σ, σ′, a′] * A[β, σ′, β′]   # expand the environment: `conj(A[...])` is the bra ⟨ψ| (complex conjugate); `W[a,σ,σ′,a′]` is the MPO insertion; `A[β,σ′,β′]` is the ket |ψ⟩; all shared indices (α,a,β,σ,σ′) are contracted (summed); result grows the environment by one site
        env = new_env   # update the environment for the next site (immutable rebinding — no mutation)
    end

    return env[1, 1, 1]   # after all L sites the environment is 1×1×1; `[1,1,1]` extracts the single scalar value. this is ⟨ψ|O|ψ⟩
end

# ----------------------------------------------------------------------------------------
# MPO application: O|ψ⟩  (§6.3 / §8)
# ----------------------------------------------------------------------------------------

"""
    apply_mpo(O::FiniteMPO, ψ::FiniteMPS; trunc=NoTrunc()) -> FiniteMPS

Apply the MPO `O` to the MPS `ψ`, returning the MPS representation of `O|ψ⟩`.

At each site the MPO tensor `W[a, σ_out, σ_in, b]` is contracted with the MPS
site tensor `A[α, σ_in, β]`, fusing the auxiliary indices `(a, α) → aα` and
`(b, β) → bβ` to produce a "thick" MPS with bond dimension `χ_mpo × χ_mps`.
The result is then compressed back to `trunc` via a left-canonical SVD sweep.

The zip-contract is exact up to the SVD truncation controlled by `trunc`: each
site-by-site contraction introduces no additional approximation beyond `trunc`.
"""
function apply_mpo(O::FiniteMPO, ψ::FiniteMPS; trunc::AbstractTrunc=NoTrunc())   # compute O|ψ⟩ as a new FiniteMPS; `trunc` controls bond dimension of the result; default NoTrunc() = keep all singular values
    L = O.L   # chain length
    d = O.d   # local physical dimension

    # Zip contraction: build thick MPS tensors (χ_mpo*χ_mps bonds) site by site,
    # then do a single left-canonical SVD sweep with the requested truncation.
    thick = Vector{Array{ComplexF64,3}}(undef, L)   # pre-allocate L uninitialized rank-3 arrays; these will hold the "thick" MPS tensors with fused bond indices before compression

    for i in 1:L   # contract MPO and MPS at each site
        A = ψ.tensors[i].data   # (χ_L, d, χ_R)  # MPS tensor at site i
        W = O.tensors[i]        # (χ_mpo_L, d_out, d_in, χ_mpo_R)  # MPO tensor at site i
        χ_L, _, χ_R = size(A)   # destructure MPS bond dimensions; `_` discards the physical dimension d
        χW_L, _, _, χW_R = size(W)   # destructure MPO bond dimensions; two `_` discard both physical dimensions

        # Contract over physical index d_in=σ, fuse auxiliary bonds:
        # new[(a,α), σ_out, (b,β)] = Σ_σ W[a,σ_out,σ,b] * A[α,σ,β]
        # Use @tensor then reshape to fuse (a,α) and (b,β).
        @tensor Θ[a, α, σo, b, β] := W[a, σo, σ, b] * A[α, σ, β]   # contract over physical index σ; result is a rank-5 tensor: (χW_L, χ_L, d, χW_R, χ_R); σ is summed (contracted), σo becomes the new physical output index
        # Fuse: reshape (χW_L, χ_L, d, χW_R, χ_R) → (χW_L*χ_L, d, χW_R*χ_R)
        thick[i] = reshape(Θ, χW_L * χ_L, d, χW_R * χ_R)   # fuse left bond indices (a,α)→aα and right bond indices (b,β)→bβ; `reshape(A, m, n, p)` reinterprets the array with new shape (column-major); the "thick" MPS has bond dimension χW×χ
    end

    # Compress via left-canonical SVD sweep (reuse MPS machinery directly)
    return _compress_thick_mps(thick, d, L, trunc)   # compress the thick MPS back to target bond dimension using SVD; delegates to the internal helper below
end

# Left-canonical SVD sweep on thick MPS tensors (mirrors _left_sweep_mps! in canonicalize.jl).
function _compress_thick_mps(   # internal function (name starts with `_` by convention, like Python's `_private_function`); performs a left-canonical SVD sweep to compress a "thick" MPS
    tensors::Vector{Array{ComplexF64,3}},
    d::Int,
    L::Int,
    trunc::AbstractTrunc,   # positional arguments with type annotations; `Vector{Array{T,3}}` = vector of 3D arrays
)
    result = Vector{QTensor}(undef, L)   # pre-allocate output vector of QTensors
    bond_svs = Vector{SingValSpectrum}(undef, L + 1)   # singular value spectra at each bond (L+1 boundaries for L sites)
    ε_total = 0.0   # accumulated truncation error; `0.0` is a Float64 literal

    bond_svs[1] = SingValSpectrum([1.0], 0.0, true)   # left boundary: trivial singular value spectrum (norm=1); `[1.0]` is a length-1 Float64 vector; `true` = normalised

    carry = tensors[1]   # start with the first tensor; `carry` will hold the current "carry" between SVD steps (the part that gets passed rightward and merged with the next tensor)
    χ_left = size(carry, 1)   # left bond dimension of the carry tensor (dimension 1)

    for i in 1:(L - 1)   # SVD sweep: process all but the last site
        M = reshape(carry, χ_left * d, size(carry, 3))   # reshape carry from (χ_left, d, χ_right) to (χ_left*d, χ_right) for SVD; this groups the left bond + physical indices for the left factor of the SVD

        F = _robust_svd(M)   # thin SVD: M = U Σ Vt; `F.U` = left singular vectors, `F.S` = singular values, `F.Vt` = right singular vectors (V-transpose)
        tol = length(F.S) * eps(eltype(F.S)) * (isempty(F.S) ? 1.0 : F.S[1])   # numerical noise floor for singular values; `eps(eltype(F.S))` = machine epsilon for the element type; `F.S[1]` = largest singular value (SVD sorts descending); `isempty(F.S) ? 1.0 : F.S[1]` safely handles the empty case
        S_clean = filter(s -> s > tol, F.S)   # remove numerically zero singular values; `filter(pred, iter)` = Python `list(filter(pred, iter))`
        r, ε_bond = _truncate_singular_values(S_clean, trunc)   # choose how many to keep: `r` = kept count, `ε_bond` = truncation error for this bond; multiple return values unpacked via Julia tuple assignment
        svs = F.S[1:r]   # keep only the r largest singular values
        ε_total += ε_bond   # accumulate truncation error across all bonds

        normalized = isapprox(sum(abs2, svs), 1.0; atol=sqrt(eps(eltype(svs))))   # check if ‖svs‖²≈1; `sum(abs2, svs)` maps `abs2` (|x|²) then sums. `isapprox(a,b;atol=tol)` is a tolerance-aware equality check
        bond_svs[i + 1] = SingValSpectrum(svs, ε_bond, normalized)   # store singular value spectrum at bond i+1
        result[i] = QTensor(   # construct left-canonical QTensor at site i
            reshape(F.U[:, 1:r], χ_left, d, r),   # reshape truncated U columns into (χ_left, d, r) 3D tensor; `F.U[:, 1:r]` = first r columns
            (upper(:vL, χ_left), upper(:σ, d), lower(:vR, r)),   # index variance tags: upper(contravariant) for left bond and physical, lower(covariant) for right bond
        )

        χ_next_L, _, χ_next_R = size(tensors[i + 1])   # dimensions of the NEXT site tensor (before carry absorption)
        carry = reshape(   # compute the new carry = Σ·Vt merged with the next tensor
            Diagonal(svs) * F.Vt[1:r, :] * reshape(tensors[i + 1], χ_next_L, d * χ_next_R),   # Σ·Vt is the right factor; multiplied by next tensor reshaped as (χ_next_L, d*χ_next_R) to fuse; result (r, d*χ_next_R); `Diagonal(svs)` creates diagonal matrix from vector
            r,   # first dimension of new carry = r (left bond = truncated rank)
            d,   # second dimension = physical
            χ_next_R,   # third dimension = right bond of next tensor
        )
        χ_left = r   # update left bond dimension for the next iteration
    end

    χ_R = size(carry, 3)   # right bond dimension of the last tensor (boundary = 1 for normalized MPS)
    result[L] = QTensor(carry, (upper(:vL, χ_left), upper(:σ, d), lower(:vR, χ_R)))   # store the last tensor (no SVD needed — it absorbs all remaining norm)
    bond_svs[L + 1] = SingValSpectrum([1.0], 0.0, true)   # right boundary singular value spectrum

    return FiniteMPS(result, bond_svs, CanonicalForm(L, L + 1), ε_total)   # construct the compressed FiniteMPS in left-canonical form; `CanonicalForm(L, L+1)` marks it as left-canonical (orthogonality centre at or past the last site)
end
