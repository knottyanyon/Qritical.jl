# §7.1–7.2  TEBD: time-evolution by block exponentiation + Trotter decomposition.
#
# A TEBD step applies a sequence of two-site gates drawn from the Suzuki-Trotter
# decomposition of e^{-iΔtH} (real time) or e^{-ΔτH} (imaginary time).
# For a nearest-neighbour chain the even/odd bond groups commute within each group,
# so each single-bond exponential is exact; only inter-group commutators contribute
# Trotter error.

# ----------------------------------------------------------------------------------------
# Time axis types
# ----------------------------------------------------------------------------------------

"""
    TimeAxis

Abstract supertype for the axis of time evolution.

Two concrete singletons:

  - [`RealTime`](@ref)       — unitary evolution ``e^{-i \\Delta t H}``.
  - [`ImaginaryTime`](@ref)  — non-unitary, PSD evolution ``e^{-\\Delta\\tau H}``;
    requires explicit renormalization after each step.
"""
abstract type TimeAxis end   # abstract type declaration: `abstract type Name end`; this is Julia's equivalent of a Python ABC (Abstract Base Class); you cannot instantiate this type — it exists only to group RealTime and ImaginaryTime under a common supertype for dispatch

"""
    RealTime <: TimeAxis

Selects unitary evolution ``e^{-i \\Delta t H}``.  A gate built with this axis
satisfies ``G^\\dagger G = I`` and preserves the MPS norm exactly.
"""
struct RealTime <: TimeAxis end   # concrete singleton type inheriting from TimeAxis; `<:` is the subtype operator. being a subtype means `RealTime() isa TimeAxis` is true; zero-field struct = zero memory, used only for dispatch

"""
    ImaginaryTime <: TimeAxis

Selects imaginary-time evolution ``e^{-\\Delta\\tau H}``.  A gate built with this
axis is Hermitian and positive semidefinite but not unitary; the evolved state
must be renormalized at each step to keep it normalized.
"""
struct ImaginaryTime <: TimeAxis end   # imaginary time: τ = it, so e^{-iHt} → e^{-Hτ}; this operator is real, symmetric (PSD for bounded-below H), and contracts the norm — physics: used to project onto the ground state (highest eigenvalue of e^{-Hτ} = lowest energy eigenstate)

# ----------------------------------------------------------------------------------------
# LatticeOperator class tags
# ----------------------------------------------------------------------------------------

"""
    Unitary

Tag returned by `opclass` for a real-time `Propagator`.  Indicates ``G^\\dagger G = I``.
"""
struct Unitary end   # dispatch tag signalling that a gate is unitary (G†G = I); physics: real-time evolution preserves norms and probabilities

"""
    HermitianPSD

Tag returned by `opclass` for an imaginary-time `Propagator`.
Indicates the gate is Hermitian and positive semidefinite.
"""
struct HermitianPSD end   # dispatch tag for imaginary-time gates; PSD = positive semidefinite (all eigenvalues ≥ 0); physics: e^{-τh} is Hermitian and PSD for any Hermitian h

# ----------------------------------------------------------------------------------------
# Propagator — the gate tensor plus metadata
# ----------------------------------------------------------------------------------------

"""
    Propagator{A<:TimeAxis}

A two-site evolution gate carrying its time axis `A` and time step `dt`.

Fields:

  - `data::Matrix{ComplexF64}` — the ``d^2 \\times d^2`` gate matrix.
  - `axis::A`                  — `RealTime()` or `ImaginaryTime()`.
  - `dt::Float64`              — the time step used to build the gate.

Construct via [`gate`](@ref).  Query the physical class via [`opclass`](@ref).
"""
struct Propagator{A<:TimeAxis}   # parametric struct: `{A<:TimeAxis}` constrains the type parameter A to be a subtype of TimeAxis; this is like a generic class in Python `class Propagator(Generic[A])` but with the constraint `A extends TimeAxis`; the type parameter A is RealTime or ImaginaryTime and affects dispatch
    data::Matrix{ComplexF64}   # the d²×d² gate matrix G; `Matrix{T}` is Julia's built-in 2D array type (same as `Array{T,2}`); physics: G = exp(-i dt h) for a two-site bond Hamiltonian h
    axis::A                    # instance of the time axis type: either RealTime() or ImaginaryTime(); stored so callers can query it without knowing the parametric type
    dt::Float64                # the time step used to construct this gate; stored for record-keeping and gate composition
end

"""
    opclass(G::Propagator{RealTime})    -> Unitary()
    opclass(G::Propagator{ImaginaryTime}) -> HermitianPSD()

Return the algebraic class of the gate, derived purely from the time axis type.
"""
opclass(::Propagator{RealTime}) = Unitary()        # dispatch: called when gate has type parameter RealTime; returns the singleton Unitary(); physics: real-time gates satisfy G†G = I (unitarity)
opclass(::Propagator{ImaginaryTime}) = HermitianPSD()   # dispatch: imaginary-time gates are Hermitian PSD; no argument name needed since we don't use the value, only its type

# ----------------------------------------------------------------------------------------
# Gate construction from a bond Hamiltonian
# ----------------------------------------------------------------------------------------

"""
    gate(h::AbstractMatrix, dt::Real, ::RealTime)    -> Propagator{RealTime}
    gate(h::AbstractMatrix, dt::Real, ::ImaginaryTime) -> Propagator{ImaginaryTime}

Exponentiate a bond Hamiltonian `h` to produce a two-site gate.

For real time the gate is ``G = e^{-i \\Delta t h}``; for imaginary time
``G = e^{-\\Delta\\tau h}``.  The matrix exponential is evaluated by diagonalising
`h` (which must be Hermitian) and applying the scalar exponential to eigenvalues.
"""
function gate(h::AbstractMatrix, dt::Real, axis::RealTime)   # build a real-time gate G = e^{-i dt h}; `AbstractMatrix` accepts any 2D array (Matrix, Hermitian, Sparse, etc.); `Real` accepts any real numeric type (Float64, Int, etc.)
    phase = -im * dt   # the phase exponent: -i*dt; `im` is the imaginary unit combining with eigenvalues gives e^{-i dt λ_k}
    F = eigen(Hermitian(ComplexF64.(h)))   # diagonalise h = V Λ V†; `ComplexF64.(h)` broadcasts element-wise type conversion; `Hermitian(...)` tells LAPACK the matrix is Hermitian (uses more efficient symmetric eigensolver); physics: the gate h must be Hermitian for unitarity of e^{-i dt h}
    data = F.vectors * Diagonal(exp.(phase .* F.values)) * F.vectors'   # compute G = V exp(-i dt Λ) V†; `Diagonal(v)` creates a diagonal matrix from vector v ; `exp.(phase .* F.values)` broadcasts exp over the eigenvalue vector; `*` is matrix multiply; `F.vectors'` is conjugate transpose V†
    return Propagator{RealTime}(data, axis, Float64(dt))   # construct the typed Propagator; `Propagator{RealTime}(...)` explicitly sets the type parameter; `Float64(dt)` converts dt to Float64
end

function gate(h::AbstractMatrix, dt::Real, axis::ImaginaryTime)   # imaginary-time version: G = e^{-dt h}; different sign and no imaginary unit compared to real-time
    F = eigen(Hermitian(ComplexF64.(h)))   # same diagonalisation as real-time; for imaginary time h must be Hermitian for G to be PSD
    data = F.vectors * Diagonal(exp.(-dt .* F.values)) * F.vectors'   # G = V exp(-dt Λ) V†; no `im` here — purely real decay; physics: eigencomponents with high energy λ_k are suppressed faster
    return Propagator{ImaginaryTime}(data, axis, Float64(dt))   # construct imaginary-time Propagator with correct type parameter
end

# ----------------------------------------------------------------------------------------
# ConstantProtocol — a fixed-Hamiltonian evolution schedule
# ----------------------------------------------------------------------------------------

"""
    ConstantProtocol{A<:TimeAxis}

An evolution protocol with a constant Hamiltonian, fixed time step, and fixed number
of steps.

Fields:

  - `axis::A`         — `RealTime()` or `ImaginaryTime()`.
  - `dt::Float64`     — time step per Trotter step.
  - `nsteps::Int`     — total number of Trotter steps.
  - `hamiltonian`     — the `LatticeOperator` driving the evolution.
"""
struct ConstantProtocol{A<:TimeAxis}   # parametric struct: A is RealTime or ImaginaryTime, baked into the type for dispatch; `{A<:TimeAxis}` constrains A to subtypes of TimeAxis
    axis::A           # the time axis singleton: RealTime() or ImaginaryTime(); physics: determines whether we do unitary or dissipative evolution
    dt::Float64       # time step size Δt; physics: should be small enough that Trotter error (O(dt^{p+1}) for order-p formula) is acceptable
    nsteps::Int       # total number of Trotter steps N; `Int` is Julia's default integer type (machine word size, typically 64-bit); total time T = dt * nsteps
    hamiltonian::Any   # LatticeOperator — typed as Any to avoid circular dependency  # the Hamiltonian operator; `Any` is Julia's top type (every value is an Any); used here to avoid a forward-declaration circular dependency — in Python, this isn't needed because Python doesn't compile type dependencies the same way
end

"""
    total_time(p::ConstantProtocol) -> Float64

Return the total evolution time ``\\Delta t \\times n_{\\text{steps}}``.
"""
total_time(p::ConstantProtocol) = p.dt * p.nsteps   # one-line function: total time T = Δt × N; field access `.dt` and `.nsteps` same as Python attribute access

"""
    gate(h::AbstractMatrix, p::ConstantProtocol{A}) -> Propagator{A}

Build a gate from bond Hamiltonian `h` and the protocol's time step and axis.
The gate axis matches the protocol axis, so the axis type is **never lost**.
"""
gate(h::AbstractMatrix, p::ConstantProtocol{A}) where {A} = gate(h, p.dt, p.axis)   # convenience overload: extracts dt and axis from a protocol; `where {A}` is a type constraint that captures the type parameter A from the function signature; dispatches to the 3-argument `gate` above

# ----------------------------------------------------------------------------------------
# Bond Hamiltonian extraction from an LatticeOperator
# ----------------------------------------------------------------------------------------

"""
    bond_hamiltonian(H::LatticeOperator, b::Int) -> Matrix{ComplexF64}

Extract the ``d^2 \\times d^2`` two-site Hamiltonian for the ``b``-th bond of `H`.

The bond Hamiltonian is the sum of all bond terms acting on that bond pair
plus the **on-site fields split evenly** across its two endpoints:

```math
h_{ij} = \\sum_{\\text{BondTerms on }(i,j)} J_{ij}\\, O_i \\otimes O_j
        - \\frac{1}{2} h_i\\, S^z_i \\otimes I_j
        - \\frac{1}{2} h_j\\, I_i \\otimes S^z_j
```

Boundary sites appear in only one bond, so their full on-site contribution is
assigned to that bond (split-half accounting restores the correct total).
"""
function bond_hamiltonian(H::LatticeOperator, b::Int)   # extract the two-site Hamiltonian for bond b; physics: TEBD acts one bond at a time, so we need h_{i,j} (the d²×d² local Hamiltonian) to build the gate
    d = local_dim(H.dof)   # local dimension (d=2 for spin-1/2); the two-site Hilbert space is d² dimensional
    Id = Matrix{ComplexF64}(I, d, d)   # d×d identity matrix as dense; will be used for kron products when only one site acts
    bnd = bonds(H.geom)   # list of bond pairs `[(i₁,j₁), (i₂,j₂), ...]` from the geometry; `bonds()` returns the ordered list of NN bond pairs
    i, j = bnd[b]   # unpack the site indices for the b-th bond (1-indexed); e.g. for a chain, bnd[1]=(1,2), bnd[2]=(2,3), etc.
    L = H.geom.L   # number of sites; needed for counting how many bonds each site belongs to

    h = zeros(ComplexF64, d^2, d^2)   # initialise d²×d² bond Hamiltonian to zero; will accumulate all contributions; `zeros(T, m, n)` is like np.zeros((m,n), dtype=T)

    # Bond operator contributions
    for bt in H.bond   # iterate over all two-site bond terms in the full Hamiltonian
        (bt.i == i && bt.j == j) || continue   # skip terms not acting on bond (i,j); `||` lazy OR: if condition is false execute `continue` (skip to next iteration); `&&` is short-circuit AND; `continue` same as Python
        h .+= bt.coupling .* kron(ComplexF64.(bt.op_i), ComplexF64.(bt.op_j))   # add J_ij * (O_i ⊗ O_j) to local bond Hamiltonian; `kron` computes Kronecker product of two d×d matrices giving d²×d²; physics: this is the two-site interaction part of H
    end

    # On-site contributions: split each site's weight between its neighbouring bonds
    for lt in H.onsite   # distribute on-site terms evenly across all bonds that site participates in
        # Count how many bonds site lt.site participates in
        n_bonds = count(bnd_pair -> lt.site ∈ bnd_pair, bnd)   # count how many bonds contain site lt.site; `count(predicate, collection)` counts elements satisfying the predicate; `->` creates an anonymous function; `∈` is the membership test
        # Weight = 1/n_bonds for each bond
        weight = lt.coupling / n_bonds   # split coupling equally across bonds; interior sites have 2 bonds (each gets ½ of the on-site term); boundary sites have 1 bond (gets the full term); physics: ensures that when we sum over all bond Hamiltonians h_b, each on-site term appears exactly once
        if lt.site == i
            h .+= weight .* kron(ComplexF64.(lt.op), Id)   # on-site term at left site i: O_i ⊗ I_j; `kron(O, I)` embeds O into the d²×d² space acting on the left tensor factor
        elseif lt.site == j
            h .+= weight .* kron(Id, ComplexF64.(lt.op))   # on-site term at right site j: I_i ⊗ O_j; `kron(I, O)` acts on the right tensor factor
        end
    end

    return h   # return the assembled d²×d² bond Hamiltonian; physics: h is Hermitian and has the same spectrum as the 2-site reduction of H
end

# ----------------------------------------------------------------------------------------
# Two-site gate application
# ----------------------------------------------------------------------------------------

"""
    apply_gate(ψ::FiniteMPS, G::Propagator, bond::Int; trunc) -> FiniteMPS

Apply a two-site gate `G` at bond `bond` (connecting sites `bond` and `bond+1`).

Steps:

 1. Merge the two MPS tensors: ``\\Theta[\\alpha, \\sigma_1, \\sigma_2, \\beta]``.
 2. Contract with gate ``G[\\sigma_1', \\sigma_2', \\sigma_1, \\sigma_2]``.
 3. Reshape and SVD-split with truncation `trunc`.
 4. Store left-canonical ``A[\\alpha, \\sigma_1', r]`` and remainder ``B[r, \\sigma_2', \\beta]``.

For `RealTime` gates the norm of the MPS is preserved exactly (up to floating-point
error). For `ImaginaryTime` gates the norm decreases; the caller is responsible for
renormalization.
"""
function apply_gate(
    ψ::FiniteMPS,
    G::Propagator,
    bond::Int;
    trunc::AbstractTrunc=NoTrunc(),   # `bond::Int` requires an integer bond index; `;` separates positional from keyword arguments; `trunc::AbstractTrunc=NoTrunc()` is a keyword argument with a default value 
)::FiniteMPS   # `::FiniteMPS` after the closing `)` is a return type annotation — optional in Julia but helps catch bugs and enables compiler optimisation
    L = length(ψ.tensors)   # number of sites; `length(collection)` = Python `len(collection)` but also works on arrays
    i, j = bond, bond + 1   # sites connected by this bond; TEBD always acts on adjacent sites
    (1 ≤ i < L) || throw(ArgumentError("bond $bond out of range [1, $(L-1)]"))   # bounds check: bond must be a valid NN pair; `$(L-1)` evaluates expression in string interpolation

    A1 = ψ.tensors[i].data   # (χL, d, χM)  # MPS tensor at site i; shape: (left bond dim, phys dim, middle bond dim); `.data` accesses the raw array from the QTensor wrapper
    A2 = ψ.tensors[j].data   # (χM, d, χR)  # MPS tensor at site j; same left bond dim as A1's right bond dim
    χL, d, χM = size(A1)   # destructure shape tuple; `size(A)` returns a tuple of dimensions; Julia: `size(A1)` is a Tuple (not a numpy shape object) and can be destructured directly
    _, _, χR = size(A2)   # use `_` to skip first two dimensions and extract only χR; Julia multiple assignment

    # Merge: Θ[χL, d, d, χR]
    @tensor Θ[α, σ1, σ2, β] := A1[α, σ1, m] * A2[m, σ2, β]   # TensorOperations.jl `@tensor` macro for Einstein summation; this contracts the shared bond index `m`, merging the two site tensors into a two-site tensor Θ (Python analogy: `np.einsum('αsm,msb->αsb', A1, A2)` but with named indices); `:=` defines a new tensor (vs `=` which would update in-place); physics: Θ[α,σ1,σ2,β] = Σ_m A1[α,σ1,m] × A2[m,σ2,β] is the merged two-site wavefunction

    # Apply gate: G.data is a (d²,d²) matrix in kron ordering where
    # G.data[(σ1'-1)*d+σ2', (σ1-1)*d+σ2] = ⟨σ1',σ2'|G|σ1,σ2⟩.
    # Julia column-major reshape(G.data, d,d,d,d) gives
    # G_mat[a,b,c,d_] = G.data[a+(b-1)*d, c+(d_-1)*d], i.e. G_mat[σ2',σ1',σ2,σ1].
    G_mat = reshape(G.data, d, d, d, d)   # column-major: [σ2', σ1', σ2, σ1]  # reshape the flat (d²,d²) gate matrix into a rank-4 tensor; `reshape(A, dims...)` same as numpy but COLUMN-MAJOR (Julia stores arrays in column-major order, i.e. first index varies fastest); the comment explains the index ordering that results from Julia's column-major convention
    @tensor Θ_new[α, σ1p, σ2p, β] := G_mat[σ2p, σ1p, σ2, σ1] * Θ[α, σ1, σ2, β]   # apply the gate by contracting physical indices σ1,σ2 → σ1p,σ2p; the `@tensor` macro handles the summation over σ1 and σ2; physics: G|Θ⟩ maps the old physical states σ1,σ2 to new states σ1p,σ2p

    # SVD split: reshape to (χL*d, d*χR)
    M = reshape(Θ_new, χL * d, d * χR)   # reshape Θ_new into a matrix for SVD; the legs are already in the order (α, σ1p, σ2p, β), so a plain column-major reshape groups the left pair against the right pair for the bipartition SVD — no permutation needed
    F = svd(M)   # thin SVD: M = U Σ Vt; `F.U` is χL*d × r, `F.S` is length-r singular values, `F.Vt` is r × d*χR
    tol = length(F.S) * eps(eltype(F.S)) * (isempty(F.S) ? 1.0 : F.S[1])   # machine-precision-based truncation floor; `eps(T)` returns machine epsilon for type T ; physics: singular values below numerical noise level are meaningless
    S_clean = filter(s -> s > tol, F.S)   # remove numerically zero singular values; `filter(predicate, collection)` returns elements satisfying predicate 
    r, ε_bond = _truncate_singular_values(S_clean, trunc)   # determine how many singular values to keep given truncation scheme; `r` is the kept bond dimension; `ε_bond` is the 2-norm of the singular values thrown away at this bond; physics: truncation controls bond dimension growth (key to TEBD efficiency), and ε_bond² is the Schmidt weight that growth cost us
    svs = F.S[1:r]   # keep the first r (largest) singular values; `1:r` is a range 1 to r inclusive; Julia is 1-indexed 

    A1_new = reshape(F.U[:, 1:r], χL, d, r)           # left-canonical  # reshape truncated U into new left tensor; `[:, 1:r]` = all rows, first r columns. physics: A1_new satisfies A†A = I (left-canonical form)
    A2_new = reshape(Diagonal(svs) * F.Vt[1:r, :], r, d, χR)  # absorb Σ  # absorb singular values into right tensor: Σ Vt; `Diagonal(svs)` creates a diagonal matrix (like np.diag); `F.Vt[1:r, :]` = first r rows, all columns. physics: absorbing Σ into the right tensor gives mixed-canonical form

    # Build new tensor list with the two updated sites
    tensors = copy(ψ.tensors)   # shallow copy of the tensor list; `copy` copies the container but not the elements inside ; we replace only tensors[i] and tensors[j] below
    bond_svs = copy(ψ.bond_svs)   # shallow copy of the singular value spectrum list; same pattern

    # Outer legs keep their old variance — their partner sites are untouched, so
    # re-tagging them would break the one-up-one-down bond pairing. The fresh inner
    # bond points toward site j (arrow in = Upper on j), which absorbed Σ.
    tensors[i] = QTensor(A1_new, (ψ.tensors[i].indices[1], upper(:σ, d), lower(:vR, r)))   # construct a new QTensor at site i with updated data and index metadata; `upper(:σ, d)` creates an Upper-variance physical index (contravariant); `lower(:vR, r)` creates a Lower-variance right-bond index pointing toward j; the left index is preserved from the original tensor
    tensors[j] = QTensor(A2_new, (upper(:vL, r), upper(:σ, d), ψ.tensors[j].indices[3]))   # new QTensor at site j; `upper(:vL, r)` for left-bond index; the right index of j is preserved unchanged; physics: the new inner bond dimension is r (after truncation)
    normalized = isapprox(sum(abs2, svs), 1.0; atol=sqrt(eps(eltype(svs))))   # check if ‖ψ‖² = Σ s_k² ≈ 1; `isapprox(a, b; atol=...)` is Julia's `≈` with explicit tolerance. `sum(abs2, svs)` applies `abs2` (|x|²) to each element then sums ; this determines whether the SingValSpectrum is marked as normalised
    bond_svs[i + 1] = SingValSpectrum(svs, ε_bond, normalized)   # store the new singular value spectrum at bond i+1 (between sites i and j), together with the error this gate's truncation introduced there; this per-bond figure is what the TEBD progress logger reports as `ε_max`

    return FiniteMPS(tensors, bond_svs, ArbitraryForm(), hypot(ψ.ε, ε_bond))   # construct new MPS with updated tensors; `ArbitraryForm()` indicates the canonical form is unknown after the gate application; `hypot(ψ.ε, ε_bond)` adds this gate's error to the running total in quadrature — ε is a 2-norm, so it is the squares (the discarded weights) that accumulate
end

# ----------------------------------------------------------------------------------------
# Suzuki-Trotter decomposition
# ----------------------------------------------------------------------------------------

"""
    TrotterSubstep

One substep in a Suzuki-Trotter decomposition: the index of the bond and the
time step to apply.
"""
struct TrotterSubstep   # plain data struct holding one bond-gate instruction; physics: each substep = "apply gate at bond b with time step dt"
    bond::Int      # which bond (1-indexed) to apply the gate at
    dt::Float64    # the fractional time step for this substep (may be dt/2 for 2nd-order)
end

"""
    SuzukiTrotter{order}

Suzuki-Trotter product formula of the given `order`.

  - `SuzukiTrotter(1)` — first-order (Lie-Trotter): one sequential sweep through all bonds.
    Error ``O(\\Delta t^2)`` per step.
  - `SuzukiTrotter(2)` — second-order (Strang / symmetric): symmetric palindrome.
    Error ``O(\\Delta t^3)`` per step.

The higher-order schemes achieve better accuracy at the same ``\\Delta t`` by choosing
carefully fractional sub-steps that cancel leading commutator error terms.
"""
struct SuzukiTrotter{order} end   # parametric singleton struct: `{order}` stores the Trotter order as a type parameter (integer, like 1 or 2) rather than a field; allows dispatch on order without runtime checks; physics: higher-order Trotter formulas reduce the error e^{A+B} ≈ e^A e^B to higher powers of dt
SuzukiTrotter(order::Int) = SuzukiTrotter{order}()   # convenience constructor: `SuzukiTrotter(2)` creates `SuzukiTrotter{2}()` by wrapping the integer into the type parameter; same as `ExactDiagonalization(mode)`

"""
    trotter_steps(formula::SuzukiTrotter, H::LatticeOperator, dt::Real) -> Vector{TrotterSubstep}

Return the ordered list of single-bond gate applications for one Trotter step.

Each `TrotterSubstep` carries a bond index and the sub-step time `dt` to pass to
[`gate`](@ref).  The returned sequence, applied left-to-right, approximates
``e^{-i \\Delta t H}`` to the order of the formula.
"""
function trotter_steps(::SuzukiTrotter{1}, H::LatticeOperator, dt::Real)   # 1st-order Trotter (Lie product formula): sweep through bonds left-to-right once; error = O(dt²) per step due to commutator [h_1, h_2]
    bnd = bonds(H.geom)   # list of bond pairs from the geometry
    return [TrotterSubstep(b, Float64(dt)) for b in 1:length(bnd)]   # list comprehension: one substep per bond with full dt; `length(bnd)` = number of bonds ` converts to Float64 if dt is e.g. Int
end

function trotter_steps(::SuzukiTrotter{2}, H::LatticeOperator, dt::Real)   # 2nd-order Strang splitting: palindrome of half-steps; error = O(dt³) per step; used because e^{(A+B)dt} ≈ e^{A dt/2} e^{B dt} e^{A dt/2} up to O(dt³)
    bnd = bonds(H.geom)   # get bond list
    n = length(bnd)   # number of bonds
    half = Float64(dt) / 2   # half the time step; each bond gate uses dt/2 for the symmetric palindrome; `Float64(dt)` ensures floating-point division
    # Forward half-step for all but last bond, full step for last, then reverse
    fwd = [TrotterSubstep(b, half) for b in 1:n]   # forward sweep: bonds 1,2,...,n each with dt/2
    bwd = [TrotterSubstep(b, half) for b in n:-1:1]   # backward sweep: bonds n,n-1,...,1 each with dt/2; `n:-1:1` is a decreasing range 
    # True 2nd-order: dt/2 each bond forward, then dt/2 each bond backward
    # (palindrome: same bond sequence in reverse at half dt)
    return vcat(fwd, bwd)   # concatenate forward and backward sweep vectors; `vcat` = vertical concatenate for arrays/vectors ; result is the complete palindrome sequence
end

# ----------------------------------------------------------------------------------------
# Single Trotter step
# ----------------------------------------------------------------------------------------

"""
    trotter_step(ψ::FiniteMPS, H::LatticeOperator, dt::Real,
                 formula::SuzukiTrotter; trunc) -> FiniteMPS

Apply one Trotter step of the evolution ``e^{-i \\Delta t H}`` to `ψ`.

The step decomposes `H` into bond sub-steps via [`trotter_steps`](@ref), computes
the gate for each bond, and applies them in sequence via [`apply_gate`](@ref).

For real-time evolution (`RealTime`) the norm is preserved.  For imaginary-time
evolution (`ImaginaryTime`) the caller should renormalize after the step.
"""
function trotter_step(
    ψ::FiniteMPS,
    H::LatticeOperator,
    dt::Real,
    formula::SuzukiTrotter;   # `formula::SuzukiTrotter` matches any SuzukiTrotter{order}; the specific order is dispatched inside `trotter_steps`; `;` here ends the positional args and starts keyword args
    trunc::AbstractTrunc=NoTrunc(),   # keyword argument with default: no truncation by default; `AbstractTrunc` accepts any truncation strategy (MaxBondDimTrunc, TruncEps, etc.)
    axis::TimeAxis=RealTime(),   # keyword argument: which time axis to use; defaults to real time
)
    substeps = trotter_steps(formula, H, dt)   # get the ordered list of bond substeps for this Trotter formula; dispatches on `formula::SuzukiTrotter{order}` to pick 1st or 2nd order
    cur = ψ   # `cur` holds the current state (starts as ψ, updated after each gate); Julia convention: `cur` is a local immutable binding (new FiniteMPS is returned by apply_gate, not modified in place)
    for sub in substeps   # iterate over each TrotterSubstep in the decomposition
        h_b = bond_hamiltonian(H, sub.bond)   # extract the local d²×d² bond Hamiltonian for this bond
        G = gate(h_b, sub.dt, axis)   # compute the gate G = exp(-i sub.dt h_b) or exp(-sub.dt h_b) depending on axis; each substep may use a fractional dt (e.g. dt/2 for 2nd order)
        cur = apply_gate(cur, G, sub.bond; trunc=trunc)   # apply the two-site gate to the MPS; `trunc=trunc` passes the keyword argument through. returns a new FiniteMPS (no mutation)
    end
    return cur   # return the final MPS after all substeps; Julia: last expression is the implicit return value (no `return` needed)
end
