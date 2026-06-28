# §4–5 / §6.2  Operator/Hamiltonian layer.
#
# An Operator is a linear map on the Hilbert space, represented as a sum of
# weighted products of on-site operators over a geometry and DoF.  The
# Hamiltonian is the instance that drives dynamics; observables are other
# instances.  Both are constructed identically and measured the same way.

# ─────────────────────────────────────────────────────────────────────────────
# Coupling helpers (§4)
# ─────────────────────────────────────────────────────────────────────────────

"""
    uniform(n::Int, x) -> Vector

Return a length-`n` vector with every entry equal to `x`.

This is a tiny convenience helper used by the named-model constructors (e.g.
[`XXZ`](@ref), [`Ising`](@ref)) to turn a scalar coupling into a
site-/bond-indexed array.  When you pass a scalar `J` to one of those
constructors it internally calls `uniform(nb, J)` so the downstream term
builders can always index by bond number without a special case.

# Examples
```jldoctest
julia> uniform(3, 1.5)
3-element Vector{Float64}:
 1.5
 1.5
 1.5

julia> uniform(2, 0.0)
2-element Vector{Float64}:
 0.0
 0.0
```
"""
uniform(n::Int, x) = fill(x, n)

# ─────────────────────────────────────────────────────────────────────────────
# Term types
# ─────────────────────────────────────────────────────────────────────────────

"""
    LocalTerm{O}

A single-site contribution ``h \\cdot O_i`` to a larger operator.

Think of this as the atomic unit of an on-site field term: a coupling strength
times a single-site operator matrix located at a specific site.  A list of
`LocalTerm`s inside an [`Operator`](@ref) represents a sum
``\\sum_i h_i \\, O_i``.

# Fields
- `site::Int`         — which lattice site the operator acts on.
- `op::O`             — the on-site operator matrix (typically a `Matrix{ComplexF64}`
  drawn from [`operators`](@ref)).
- `coupling::Float64` — prefactor ``h_i``; can be site-dependent.

See also: [`BondTerm`](@ref), [`Operator`](@ref)
"""
struct LocalTerm{O}
    site::Int
    op::O
    coupling::Float64
end

"""
    BondTerm{O1,O2}

A two-site contribution ``J \\cdot (O_i \\otimes O_j)`` at sites `i` and `j`.

This is the atomic unit of a nearest-neighbour exchange term: a coupling
strength times a tensor product of two single-site operator matrices at sites
``i < j``.  A list of `BondTerm`s inside an [`Operator`](@ref) represents a
sum ``\\sum_{\\langle i,j \\rangle} J_{ij} \\, O_i \\otimes O_j``.

Note that the coupling ``J`` is stored in the `BondTerm`, not split between the
two sites.  This matters for the MPO FSM builder, which multiplies the coupling
into the operator that opens the channel (at site ``i``).

# Fields
- `i::Int`            — left site index.
- `j::Int`            — right site index (usually ``j = i + 1`` for NN bonds).
- `op_i::O1`          — operator matrix acting at site ``i``.
- `op_j::O2`          — operator matrix acting at site ``j``.
- `coupling::Float64` — bond coupling ``J_{ij}``.

See also: [`LocalTerm`](@ref), [`Operator`](@ref)
"""
struct BondTerm{O1,O2}
    i::Int
    j::Int
    op_i::O1
    op_j::O2
    coupling::Float64
end

# ─────────────────────────────────────────────────────────────────────────────
# Operator type
# ─────────────────────────────────────────────────────────────────────────────

"""
    Operator{D,G,LT,BT}

A linear operator on the full Hilbert space, represented as a sum of on-site
and two-site contributions over a degree of freedom `D` and a geometry `G`.

In physics terms, an `Operator` captures the idea that most physically relevant
operators — Hamiltonians, conserved charges, observables — can be written as

```math
\\hat{O} = \\sum_i h_i \\, O_i + \\sum_{\\langle i,j \\rangle} J_{ij} \\, O_i \\otimes O_j,
```

where the first sum runs over sites and the second over bonds.  Both the
Hamiltonian (time-evolution generator) and observables (expectation values) are
`Operator` instances — the role is determined by how you use them:

- pass to `MPO` → [`FiniteMPO`](@ref) for variational energy optimisation
- evaluate via `MPO(obs)` and [`expect`](@ref) → scalar ``\\langle \\psi | \\hat{O} | \\psi \\rangle``

`Hamiltonian` is just a type alias for `Operator`.

# Type parameters
- `D <: AbstractDoF`        — the local degree of freedom (sets operator algebra and ``d``).
- `G <: AbstractGeometry`   — the lattice geometry (sets site and bond lists).
- `LT`                      — concrete `LocalTerm` type.
- `BT`                      — concrete `BondTerm` type.

# Fields
- `dof::D`               — the site degree of freedom.
- `geom::G`              — the lattice geometry.
- `onsite::Vector{LT}`   — list of single-site terms.
- `bond::Vector{BT}`     — list of two-site bond terms.

See also: [`LocalTerm`](@ref), [`BondTerm`](@ref), [`MPO`](@ref),
[`XXZ`](@ref), [`Heisenberg`](@ref), [`Ising`](@ref)
"""
struct Operator{D<:AbstractDoF, G<:AbstractGeometry, LT, BT}
    dof::D
    geom::G
    onsite::Vector{LT}
    bond::Vector{BT}
end

"""
    Hamiltonian

Type alias for [`Operator`](@ref).

There is no separate `Hamiltonian` type — the role of an `Operator` (whether it
generates dynamics or measures an observable) is determined by how it is used,
not by its type.  The alias exists purely for readability: writing
`H = XXZ(g)` and treating it as a `Hamiltonian` makes the intent obvious at
the call site.
"""
const Hamiltonian = Operator

# ─────────────────────────────────────────────────────────────────────────────
# Named constructors — spin models (§7)
# ─────────────────────────────────────────────────────────────────────────────

"""
    XXZ(g::Chain; J=1.0, Jz=1.0, h=0.0) -> Operator

Build the XXZ spin-½ Hamiltonian on the chain `g`.

The XXZ model is the canonical anisotropic spin-½ chain, interpolating between
the isotropic Heisenberg point (``J_z = J``) and the XX model (``J_z = 0``).
Its Hamiltonian is

```math
H = \\frac{J}{2} \\sum_{\\langle i,j \\rangle} \\bigl( S^+_i S^-_j + S^-_i S^+_j \\bigr)
    + J_z \\sum_{\\langle i,j \\rangle} S^z_i S^z_j
    - h \\sum_i S^z_i.
```

The factor of ``\\tfrac{1}{2}`` on the flip-flop terms is the standard
convention that ensures ``J = J_z`` reproduces the isotropic Heisenberg
exchange ``J \\mathbf{S}_i \\cdot \\mathbf{S}_j``.  The field term follows the
course convention ``-h_i S^z_i`` (see Exercise 3/6 of the SS26 course).

# Arguments
- `g::Chain`    — lattice geometry.
- `J::Union{Number, AbstractVector}`  — transverse (flip-flop) coupling; scalar
  applies uniformly to every bond, a vector sets each bond individually.
  Default: `1.0`.
- `Jz::Union{Number, AbstractVector}` — longitudinal (Ising) coupling.
  Default: `1.0`.
- `h::Union{Number, AbstractVector}`  — magnetic field (couples to ``-S^z``);
  scalar or per-site vector.  Default: `0.0`.

# Returns
- `Operator` with `dof = SpinHalf()`, suitable for `MPO(H)` or [`dense_matrix`](@ref).

# Examples
```jldoctest
julia> g = Chain(4);

julia> H = XXZ(g; J=1.0, Jz=0.5);

julia> H.dof
Spin{1//2}()

julia> length(H.bond)   # 3 bonds × 3 term types (Sp⊗Sm, Sm⊗Sp, Sz⊗Sz)
9
```
"""
function XXZ(g::Chain; J=1.0, Jz=1.0, h=0.0)
    nb  = length(bonds(g))
    Jv  = J  isa Number ? uniform(nb, Float64(J))  : Float64.(J)
    Jzv = Jz isa Number ? uniform(nb, Float64(Jz)) : Float64.(Jz)
    hv  = h  isa Number ? uniform(g.L, Float64(h)) : Float64.(h)
    ops = operators(SpinHalf())

    onsite = [LocalTerm(i, ops.Sz, -hv[i]) for i in sites(g)]

    bond = vcat(
        [BondTerm(i, j, ops.Sp, ops.Sm, 0.5Jv[b])  for (b, (i, j)) in enumerate(bonds(g))],
        [BondTerm(i, j, ops.Sm, ops.Sp, 0.5Jv[b])  for (b, (i, j)) in enumerate(bonds(g))],
        [BondTerm(i, j, ops.Sz, ops.Sz, Jzv[b])    for (b, (i, j)) in enumerate(bonds(g))],
    )

    Operator(SpinHalf(), g, onsite, bond)
end

"""
    Heisenberg(g::Chain; J=1.0, h=0.0) -> Operator

Build the isotropic Heisenberg spin-½ Hamiltonian on the chain `g`.

This is the special case ``J_z = J`` of [`XXZ`](@ref), giving the fully
SU(2)-symmetric exchange interaction:

```math
H = J \\sum_{\\langle i,j \\rangle} \\mathbf{S}_i \\cdot \\mathbf{S}_j - h \\sum_i S^z_i.
```

At ``h = 0`` and ``J > 0`` (antiferromagnetic convention) this is the
paradigmatic gapless spin-½ chain, exactly solvable via the Bethe ansatz and
described by a ``c = 1`` conformal field theory.

# Arguments
- `g::Chain`                          — chain geometry.
- `J::Union{Number, AbstractVector}`  — exchange coupling; `J > 0` is antiferromagnetic.
  Default: `1.0`.
- `h::Union{Number, AbstractVector}`  — longitudinal field. Default: `0.0`.

# Examples
```jldoctest
julia> H = Heisenberg(Chain(4));

julia> H.dof
Spin{1//2}()
```
"""
Heisenberg(g::Chain; J=1.0, h=0.0) = XXZ(g; J=J, Jz=J, h=h)

"""
    Ising(g::Chain; J=1.0, h=0.0) -> Operator

Build the transverse-field Ising Hamiltonian on the chain `g`.

The transverse-field Ising model is the simplest quantum model with a quantum
phase transition.  Its Hamiltonian is

```math
H = J \\sum_{\\langle i,j \\rangle} S^z_i S^z_j - h \\sum_i S^x_i.
```

Note that the field couples to ``S^x`` (transverse), not ``S^z``.  At the
quantum critical point ``|h/J| = 1/2`` (in the ``S^z S^z`` convention used here)
the model is exactly solvable via a Jordan–Wigner transformation.

# Arguments
- `g::Chain`                          — chain geometry.
- `J::Union{Number, AbstractVector}`  — longitudinal (Ising) coupling.
  Default: `1.0`.
- `h::Union{Number, AbstractVector}`  — transverse-field strength (couples to
  ``-S^x``).  Default: `0.0`.

# Examples
```jldoctest
julia> H = Ising(Chain(4); J=1.0, h=0.5);

julia> length(H.onsite)  # transverse-field term at every site
4

julia> length(H.bond)    # Sz⊗Sz on every NN bond
3
```
"""
function Ising(g::Chain; J=1.0, h=0.0)
    nb  = length(bonds(g))
    Jv  = J isa Number ? uniform(nb, Float64(J)) : Float64.(J)
    hv  = h isa Number ? uniform(g.L, Float64(h)) : Float64.(h)
    ops = operators(SpinHalf())

    onsite = [LocalTerm(i, ops.Sx, -hv[i]) for i in sites(g)]
    bond   = [BondTerm(i, j, ops.Sz, ops.Sz, Jv[b]) for (b, (i, j)) in enumerate(bonds(g))]

    Operator(SpinHalf(), g, onsite, bond)
end

# ─────────────────────────────────────────────────────────────────────────────
# Observable constructors (§5 / §7)
# ─────────────────────────────────────────────────────────────────────────────

"""
    total_magnetization(g::AbstractGeometry; dof=SpinHalf()) -> Operator

Build the total magnetization operator ``M = \\sum_i S^z_i``.

This is a conserved quantity of the XXZ and Heisenberg models (it commutes with
the Hamiltonian when ``h = 0``).  Measuring it after a variational optimisation
is a useful sanity check: the ground state of an antiferromagnet at zero field
on an even-``L`` chain should have ``\\langle M \\rangle = 0``.

# Arguments
- `g::AbstractGeometry` — lattice geometry.
- `dof`                 — degree of freedom; its ``S^z`` operator is used.
  Default: `SpinHalf()`.

# Returns
- `Operator` with one `LocalTerm` (coupling = +1) per site and no bond terms.

# Examples
```jldoctest
julia> M = total_magnetization(Chain(4));

julia> length(M.onsite)
4

julia> isempty(M.bond)
true
```
"""
function total_magnetization(g::AbstractGeometry; dof=SpinHalf())
    ops = operators(dof)
    Operator(dof, g, [LocalTerm(i, ops.Sz, 1.0) for i in sites(g)], BondTerm[])
end

"""
    staggered_magnetization(g::AbstractGeometry; dof=SpinHalf()) -> Operator

Build the staggered magnetization (Néel order parameter)
``M^s = \\sum_i (-1)^i S^z_i``.

In a Néel state, nearest-neighbour spins point in opposite directions, so the
alternating signs ``(-1)^i`` make all contributions add constructively.  A
non-zero expectation value ``\\langle M^s \\rangle \\neq 0`` signals broken
translational symmetry and antiferromagnetic long-range order.

For a true 1D antiferromagnet, quantum fluctuations prevent long-range Néel
order at zero temperature (Mermin–Wagner), but the staggered structure factor
still diverges with system size in a characteristic way — making this observable
very useful for finite-size scaling studies.

# Arguments
- `g::AbstractGeometry` — lattice geometry.
- `dof`                 — degree of freedom. Default: `SpinHalf()`.

# Returns
- `Operator` with site couplings ``(-1)^i`` and no bond terms.

# Examples
```jldoctest
julia> Ms = staggered_magnetization(Chain(4));

julia> [t.coupling for t in Ms.onsite]
4-element Vector{Float64}:
 -1.0
  1.0
 -1.0
  1.0
```
"""
function staggered_magnetization(g::AbstractGeometry; dof=SpinHalf())
    ops = operators(dof)
    Operator(dof, g, [LocalTerm(i, ops.Sz, (-1.0)^i) for i in sites(g)], BondTerm[])
end

"""
    local_op(dof::AbstractDoF, sym::Symbol, site::Int) -> Operator

Build a single-site observable: the operator named `sym` of `dof` at `site`.

Use this to measure any on-site quantity — ``S^z_i``, ``n_i``, ``c_i``, etc.
— after a variational calculation.  The returned `Operator` can be passed to
[`expect`](@ref) by first converting it to a `FiniteMPO` via `MPO`.

# Arguments
- `dof::AbstractDoF` — the local degree of freedom (determines the operator
  algebra).
- `sym::Symbol`      — name of the operator to retrieve from [`operators`](@ref),
  e.g. `:Sz`, `:n`, `:cup`.
- `site::Int`        — which lattice site the operator acts on.

# Returns
- `Operator` on a minimal `Chain(site)` geometry with a single `LocalTerm`
  (coupling = 1) and no bond terms.

# Examples
```jldoctest
julia> O = local_op(SpinHalf(), :Sz, 2);

julia> O.onsite[1].site
2

julia> O.onsite[1].coupling
1.0
```
"""
function local_op(dof::AbstractDoF, sym::Symbol, site::Int)
    op  = getproperty(operators(dof), sym)
    # Minimal geometry: a chain containing just this site (no bonds needed)
    g   = Chain(site)
    Operator(dof, g, [LocalTerm(site, op, 1.0)], BondTerm[])
end

"""
    two_point(g::AbstractGeometry, dof::AbstractDoF,
              opA::Symbol, iA::Int, opB::Symbol, iB::Int) -> Operator

Build the two-point operator ``A_{i_A} B_{i_B}`` whose expectation value gives
``\\langle A_{i_A} B_{i_B} \\rangle``.

This is the building block for two-point correlation functions: spin–spin
correlators ``\\langle S^z_i S^z_j \\rangle``, density–density correlators
``\\langle n_i n_j \\rangle``, and so on.  The returned `Operator` contains a
single `BondTerm` with coupling 1 and no on-site terms.

!!! note "Fermionic statistics"
    For fermionic DoFs and non-adjacent sites ``i_A < i_B``, the correct
    correlator requires a Jordan–Wigner string
    ``\\prod_{k=i_A}^{i_B-1} (-1)^{n_k}`` between the two operators.
    The MPO route via [`expect`](@ref) does not insert this string automatically
    in the current version; use [`dense_matrix`](@ref) on small systems or handle
    JW strings by hand for fermionic models.

# Arguments
- `g::AbstractGeometry` — lattice geometry (used to construct the `Operator`
  container; the pair ``(i_A, i_B)`` need not be a NN bond of `g`).
- `dof::AbstractDoF`    — local degree of freedom.
- `opA::Symbol`         — name of the left operator, e.g. `:Sz`.
- `iA::Int`             — site of the left operator.
- `opB::Symbol`         — name of the right operator.
- `iB::Int`             — site of the right operator.

# Returns
- `Operator` with a single `BondTerm(iA, iB, Amat, Bmat, 1.0)` and empty
  `onsite`.

# Examples
```jldoctest
julia> g = Chain(6);

julia> O = two_point(g, SpinHalf(), :Sz, 1, :Sz, 4);

julia> length(O.bond)
1

julia> O.bond[1].i, O.bond[1].j
(1, 4)
```
"""
function two_point(g::AbstractGeometry, dof::AbstractDoF,
                   opA::Symbol, iA::Int, opB::Symbol, iB::Int)
    ops  = operators(dof)
    Amat = getproperty(ops, opA)
    Bmat = getproperty(ops, opB)
    Operator(dof, g, LocalTerm[], [BondTerm(iA, iB, Amat, Bmat, 1.0)])
end

"""
    identity_operator(g::AbstractGeometry, dof::AbstractDoF) -> Operator

Build the identity operator
``I = I_1 \\otimes I_2 \\otimes \\cdots \\otimes I_L``.

The identity has no on-site or bond terms — both term lists are empty.  When
converted via `MPO(identity_operator(...))` this produces a bond-dimension-1
all-identity MPO.

The expectation value ``\\langle \\psi | I | \\psi \\rangle = \\|\\psi\\|^2``, which
makes this useful as a norm check when the MPS is not in a normalised form.

# Examples
```jldoctest
julia> g = Chain(4);

julia> I_op = identity_operator(g, SpinHalf());

julia> isempty(I_op.onsite) && isempty(I_op.bond)
true
```
"""
function identity_operator(g::AbstractGeometry, dof::AbstractDoF)
    Operator(dof, g, LocalTerm[], BondTerm[])
end

# ─────────────────────────────────────────────────────────────────────────────
# Operator arithmetic
# ─────────────────────────────────────────────────────────────────────────────

"""
    Base.:*(c::Real, H::Operator) -> Operator
    Base.:*(H::Operator, c::Real) -> Operator

Scale all couplings of `H` by scalar `c`.
"""
function Base.:*(c::Real, H::Operator)
    onsite = [LocalTerm(lt.site, lt.op, c * lt.coupling) for lt in H.onsite]
    bond   = [BondTerm(bt.i, bt.j, bt.op_i, bt.op_j, c * bt.coupling) for bt in H.bond]
    Operator(H.dof, H.geom, onsite, bond)
end
Base.:*(H::Operator, c::Real) = c * H

"""
    Base.:+(A::Operator, B::Operator) -> Operator

Merge the term lists of two operators (both must share the same DoF and geometry).
"""
function Base.:+(A::Operator, B::Operator)
    onsite = vcat(A.onsite, B.onsite)
    bond   = vcat(A.bond,   B.bond)
    Operator(A.dof, A.geom, onsite, bond)
end

# ─────────────────────────────────────────────────────────────────────────────
# Dense matrix from term list — used for testing and small-system ED
# ─────────────────────────────────────────────────────────────────────────────

"""
    dense_matrix(H::Operator) -> Matrix{ComplexF64}

Build the full ``d^L \\times d^L`` dense matrix of operator `H` by Kronecker
product construction.

For each on-site term ``h_i \\, O_i`` the contribution is

```math
I_{d^{i-1}} \\otimes O_i \\otimes I_{d^{L-i}},
```

and for each bond term ``J_{ij} \\, O_i \\otimes O_j`` (``j > i``) it is

```math
I_{d^{i-1}} \\otimes O_i \\otimes I_{d^{j-i-1}} \\otimes O_j \\otimes I_{d^{L-j}}.
```

All contributions are summed into a single ``d^L \\times d^L`` matrix.

This is the go-to tool for exact-diagonalisation checks on small systems
(``L \\lesssim 14`` for ``d = 2``).  For large systems use the MPO route instead.

!!! warning "Commuting statistics only"
    The Kronecker product construction is only correct for bosonic (commuting)
    DoFs such as [`Spin`](@ref) and [`HardCoreBoson`](@ref).  For fermionic DoFs
    the Jordan–Wigner string between sites ``i`` and ``j`` is not included;
    use `basis_change(H, SpinHalf())` first to embed fermionic operators into a
    spin representation with explicit JW strings.

# Returns
- `Matrix{ComplexF64}` of shape ``(d^L, d^L)``.

# Examples
```jldoctest
julia> H = Heisenberg(Chain(2));

julia> M = dense_matrix(H);

julia> size(M)
(4, 4)

julia> isapprox(M, M', atol=1e-12)
true
```
"""
function dense_matrix(H::Operator)
    # Reject periodic geometries: the wrap bond (L,1) has i>j, which makes the
    # intermediate-identity dimension d^(j-i-1) = d^(negative) meaningless.  The
    # caller should use open boundary conditions for dense_matrix.  Fixes #79.
    if any(bt -> bt.i > bt.j, H.bond)
        throw(ArgumentError(
            "dense_matrix does not support periodic boundary conditions: bond " *
            "$(first(bt for bt in H.bond if bt.i > bt.j)) has i > j.  " *
            "Use an open-boundary Chain or the sparse ED path instead."))
    end
    L = H.geom.L
    d = local_dim(H.dof)
    N = d^L
    # Apply the same Hilbert-space size guard as sparse(H) (2^20 ≈ 1M): allocating a
    # d^L × d^L dense matrix for L≥21 (d=2) silently consumes multi-GB.  Fixes #85.
    N ≤ 2^20 || throw(ArgumentError(
        "Hilbert space dimension $N = $(d)^$L exceeds the safety limit 2^20. " *
        "Use the sparse ED path or an MPS algorithm for large systems."))
    mat = zeros(ComplexF64, N, N)

    Id(n) = Matrix{ComplexF64}(I, n, n)

    for lt in H.onsite
        i     = lt.site
        left  = d^(i - 1)
        right = d^(L - i)
        mat .+= lt.coupling .* kron(kron(Id(left), ComplexF64.(lt.op)), Id(right))
    end

    for bt in H.bond
        i = bt.i; j = bt.j
        left   = d^(i - 1)
        middle = d^(j - i - 1)   # identity string between sites i and j
        right  = d^(L - j)
        if middle == 1
            op2 = kron(ComplexF64.(bt.op_i), ComplexF64.(bt.op_j))
        else
            op2 = kron(kron(ComplexF64.(bt.op_i), Id(middle)), ComplexF64.(bt.op_j))
        end
        mat .+= bt.coupling .* kron(kron(Id(left), op2), Id(right))
    end

    return mat
end
