# §4–5 / §6.2  LatticeOperator / Hamiltonian layer.
#
# A LatticeOperator is a linear map on the Hilbert space, represented as a sum of
# weighted products of on-site operators over a geometry and DoF.  The
# Hamiltonian is the instance that drives dynamics; observables are other
# instances.  Both are constructed identically and measured the same way.

# ----------------------------------------------------------------------------------------
# Coupling helpers (§4)
# ----------------------------------------------------------------------------------------

"""
    uniform_coupling(n::Int, x) -> Vector

Return a length-`n` vector with every entry equal to `x`.

This is a tiny convenience helper used by the named-model constructors (e.g.
[`XXZ`](@ref), [`Ising`](@ref)) to turn a scalar coupling into a
site-/bond-indexed array.  When you pass a scalar `J` to one of those
constructors it internally calls `uniform_coupling(nb, J)` so the downstream term
builders can always index by bond number without a special case.

# Examples

```jldoctest
julia> uniform_coupling(3, 1.5)
3-element Vector{Float64}:
 1.5
 1.5
 1.5

julia> uniform_coupling(2, 0.0)
2-element Vector{Float64}:
 0.0
 0.0
```
"""

uniform_coupling(n::Int, x) = fill(x, n)

# ----------------------------------------------------------------------------------------
# Term types
# ----------------------------------------------------------------------------------------

"""
    OneSiteTerm{O}


A single-site contribution ``h \\cdot O_i`` to a larger operator.

Think of this as the atomic unit of an on-site field term: a coupling strength
times a single-site operator matrix located at a specific site.  A list of
`OneSiteTerm`s inside an [`LatticeOperator`](@ref) represents a sum
``\\sum_i h_i \\, O_i``.

# Fields

  - `site::Int`         — which lattice site the operator acts on.
  - `op::O`             — the on-site operator matrix (typically a `Matrix{ComplexF64}`
    drawn from [`algebra_generators`](@ref)).
  - `coupling::Float64` — prefactor ``h_i``; can be site-dependent.

See also: [`TwoSiteTerm`](@ref), [`LatticeOperator`](@ref)
"""
struct OneSiteTerm{O}
    site::Int
    op::O
    coupling::Float64
end

"""
    TwoSiteTerm{O1,O2}

A two-site contribution ``J \\cdot (O_i \\otimes O_j)`` at sites `i` and `j`.

This is the atomic unit of a nearest-neighbour exchange term: a coupling
strength times a tensor product of two single-site operator matrices at sites
``i < j``.  A list of `TwoSiteTerm`s inside an [`LatticeOperator`](@ref) represents a
sum ``\\sum_{\\langle i,j \\rangle} J_{ij} \\, O_i \\otimes O_j``.

Note that the coupling ``J`` is stored in the `TwoSiteTerm`, not split between the
two sites.  This matters for the MPO FSM builder, which multiplies the coupling
into the operator that opens the channel (at site ``i``).

# Fields

  - `i::Int`            — left site index.
  - `j::Int`            — right site index (usually ``j = i + 1`` for NN bonds).
  - `op_i::O1`          — operator matrix acting at site ``i``.
  - `op_j::O2`          — operator matrix acting at site ``j``.
  - `coupling::Float64` — bond coupling ``J_{ij}``.

See also: [`OneSiteTerm`](@ref), [`LatticeOperator`](@ref)
"""
struct TwoSiteTerm{O1,O2}
    i::Int
    j::Int
    op_i::O1
    op_j::O2
    coupling::Float64
end

# ----------------------------------------------------------------------------------------
# LatticeOperator type
# ----------------------------------------------------------------------------------------

"""
    LatticeOperator{D,G,LT,BT}

A linear operator on the full Hilbert space, represented as a sum of on-site
and two-site contributions over a degree of freedom `D` and a geometry `G`.

In physics terms, an `LatticeOperator` captures the idea that most physically relevant
operators — Hamiltonians, conserved charges, observables — can be written as

```math
\\hat{O} = \\sum_i h_i \\, O_i + \\sum_{\\langle i,j \\rangle} J_{ij} \\, O_i \\otimes O_j,
```

where the first sum runs over sites and the second over bonds.  Both the
Hamiltonian (time-evolution generator) and observables (expectation values) are
`LatticeOperator` instances — the role is determined by how you use them:

  - pass to `MPO` → [`FiniteMPO`](@ref) for variational energy optimisation
  - evaluate via `MPO(obs)` and [`expect`](@ref) → scalar ``\\langle \\psi | \\hat{O} | \\psi \\rangle``

`Hamiltonian` is just a type alias for `LatticeOperator`.

# Type parameters

  - `D <: AbstractDoF`        — the local degree of freedom (sets operator algebra and ``d``).
  - `G <: AbstractGeometry`   — the lattice geometry (sets site and bond lists).
  - `LT`                      — concrete `OneSiteTerm` type.
  - `BT`                      — concrete `TwoSiteTerm` type.

# Fields

  - `dof::D`               — the site degree of freedom.
  - `geom::G`              — the lattice geometry.
  - `onsite::Vector{LT}`   — list of single-site terms.
  - `bond::Vector{BT}`     — list of two-site bond terms.

See also: [`OneSiteTerm`](@ref), [`TwoSiteTerm`](@ref), [`MPO`](@ref),
[`XXZ`](@ref), [`Heisenberg`](@ref), [`Ising`](@ref)
"""
struct LatticeOperator{D<:AbstractDoF,G<:AbstractGeometry,LT,BT}
    dof::D
    geom::G
    onsite::Vector{LT}
    bond::Vector{BT}
end

"""
    Hamiltonian

Type alias for [`LatticeOperator`](@ref).

There is no separate `Hamiltonian` type — the role of an `LatticeOperator` (whether it
generates dynamics or measures an observable) is determined by how it is used,
not by its type.  The alias exists purely for readability: writing
`H = XXZ(g)` and treating it as a `Hamiltonian` makes the intent obvious at
the call site.
"""
const Hamiltonian = LatticeOperator

# ----------------------------------------------------------------------------------------
# Named constructors — spin models (§7)
# ----------------------------------------------------------------------------------------

"""
    XXZ(g::Chain; J=1.0, Jz=1.0, h=0.0) -> LatticeOperator

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
  - `h::Union{Number, AbstractVector}`  — external magnetic field (couples to ``-S^z``);
    scalar or per-site vector.  Default: `0.0`.

# Returns

  - `LatticeOperator` with `dof = SpinHalf()`, suitable for `MPO(H)` or [`dense_matrix`](@ref).

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
    nb = length(bonds(g))
    Jxy = J isa Number ? uniform_coupling(nb, Float64(J)) : Float64.(J)
    Jzv = Jz isa Number ? uniform_coupling(nb, Float64(Jz)) : Float64.(Jz)
    hv = h isa Number ? uniform_coupling(g.L, Float64(h)) : Float64.(h)
    ops = algebra_generators(SpinHalf())

    onsite = [OneSiteTerm(i, ops.Sz, -hv[i]) for i in sites(g)]

    bond = vcat(
        [TwoSiteTerm(i, j, ops.Sp, ops.Sm, 0.5Jxy[b]) for (b, (i, j)) in enumerate(bonds(g))],
        [TwoSiteTerm(i, j, ops.Sm, ops.Sp, 0.5Jxy[b]) for (b, (i, j)) in enumerate(bonds(g))],
        [TwoSiteTerm(i, j, ops.Sz, ops.Sz, Jzv[b]) for (b, (i, j)) in enumerate(bonds(g))],
    )

    LatticeOperator(SpinHalf(), g, onsite, bond)
end

"""
    Heisenberg(g::Chain; J=1.0, h=0.0) -> LatticeOperator

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
    Ising(g::Chain; J=1.0, h=0.0) -> LatticeOperator

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
    nb = length(bonds(g))
    Jxy = J isa Number ? uniform_coupling(nb, Float64(J)) : Float64.(J)
    hv = h isa Number ? uniform_coupling(g.L, Float64(h)) : Float64.(h)
    ops = algebra_generators(SpinHalf())

    onsite = [OneSiteTerm(i, ops.Sx, -hv[i]) for i in sites(g)]
    bond = [TwoSiteTerm(i, j, ops.Sz, ops.Sz, Jxy[b]) for (b, (i, j)) in enumerate(bonds(g))]

    LatticeOperator(SpinHalf(), g, onsite, bond)
end

# ----------------------------------------------------------------------------------------
# Observable constructors (§5 / §7)
# ----------------------------------------------------------------------------------------

"""
    total_magnetization(g::AbstractGeometry; dof=SpinHalf()) -> LatticeOperator

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

  - `LatticeOperator` with one `OneSiteTerm` (coupling = +1) per site and no bond terms.

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
    ops = algebra_generators(dof)
    LatticeOperator(dof, g, [OneSiteTerm(i, ops.Sz, 1.0) for i in sites(g)], TwoSiteTerm[])
end

"""
    staggered_magnetization(g::AbstractGeometry; dof=SpinHalf()) -> LatticeOperator

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

  - `LatticeOperator` with site couplings ``(-1)^i`` and no bond terms.

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
    ops = algebra_generators(dof)
    LatticeOperator(dof, g, [OneSiteTerm(i, ops.Sz, (-1.0)^i) for i in sites(g)], TwoSiteTerm[])
end

"""
    op_at_site(dof::AbstractDoF, label::Symbol, site::Int) -> LatticeOperator

Build a single-site observable: the operator named `label` of `dof` at `site`.

Use this to measure any on-site quantity — ``S^z_i``, ``n_i``, ``c_i``, etc.
— after a variational calculation.  The returned `LatticeOperator` can be passed to
[`expect`](@ref) by first converting it to a `FiniteMPO` via `MPO`.

# Arguments

  - `dof::AbstractDoF` — the local degree of freedom (determines the operator
    algebra).
  - `label::Symbol`      — name of the operator to retrieve from [`algebra_generators`](@ref),
    e.g. `:Sz`, `:n`, `:cup`.
  - `site::Int`        — which lattice site the operator acts on.

# Returns

  - `LatticeOperator` on a minimal `Chain(site)` geometry with a single `OneSiteTerm`
    (coupling = 1) and no bond terms.

# Examples

```jldoctest
julia> O = op_at_site(SpinHalf(), :Sz, 2);

julia> O.onsite[1].site
2

julia> O.onsite[1].coupling
1.0
```
"""
function op_at_site(dof::AbstractDoF, label::Symbol, site::Int)
    op = getproperty(algebra_generators(dof), label)
    # Minimal geometry: a chain containing just this site (no bonds needed)
    g = Chain(site)
    LatticeOperator(dof, g, [OneSiteTerm(site, op, 1.0)], TwoSiteTerm[])
end

"""
    two_point(g::AbstractGeometry, dof::AbstractDoF,
              opA::Symbol, iA::Int, opB::Symbol, iB::Int) -> LatticeOperator

Build the two-point operator ``A_{i_A} B_{i_B}`` whose expectation value gives
``\\langle A_{i_A} B_{i_B} \\rangle``.

This is the building block for two-point correlation functions: spin–spin
correlators ``\\langle S^z_i S^z_j \\rangle``, density–density correlators
``\\langle n_i n_j \\rangle``, and so on.  The returned `LatticeOperator` contains a
single `TwoSiteTerm` with coupling 1 and no on-site terms.

!!! note "Fermionic statistics"

    For fermionic DoFs and non-adjacent sites ``i_A < i_B``, the correct
    correlator requires a Jordan–Wigner string
    ``\\prod_{k=i_A}^{i_B-1} (-1)^{n_k}`` between the two operators.
    The MPO route via [`expect`](@ref) does not insert this string automatically
    in the current version; use [`dense_matrix`](@ref) on small systems or handle
    JW strings by hand for fermionic models.

# Arguments

  - `g::AbstractGeometry` — lattice geometry (used to construct the `LatticeOperator`
    container; the pair ``(i_A, i_B)`` need not be a NN bond of `g`).
  - `dof::AbstractDoF`    — local degree of freedom.
  - `opA::Symbol`         — name of the left operator, e.g. `:Sz`.
  - `iA::Int`             — site of the left operator.
  - `opB::Symbol`         — name of the right operator.
  - `iB::Int`             — site of the right operator.

# Returns

  - `LatticeOperator` with a single `TwoSiteTerm(iA, iB, Amat, Bmat, 1.0)` and empty
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
function two_point(
    g::AbstractGeometry, dof::AbstractDoF, opA::Symbol, iA::Int, opB::Symbol, iB::Int
)
    ops = algebra_generators(dof)
    Amat = getproperty(ops, opA)
    Bmat = getproperty(ops, opB)
    LatticeOperator(dof, g, OneSiteTerm[], [TwoSiteTerm(iA, iB, Amat, Bmat, 1.0)])
end

"""
    identity_operator(g::AbstractGeometry, dof::AbstractDoF) -> LatticeOperator

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
    LatticeOperator(dof, g, OneSiteTerm[], TwoSiteTerm[])
end

# ----------------------------------------------------------------------------------------
# LatticeOperator arithmetic
# ----------------------------------------------------------------------------------------

"""
    Base.:*(c::Real, H::LatticeOperator) -> LatticeOperator
    Base.:*(H::LatticeOperator, c::Real) -> LatticeOperator

Scale all couplings of `H` by scalar `c`.
"""
function Base.:*(c::Real, H::LatticeOperator)
    onsite = [OneSiteTerm(lt.site, lt.op, c * lt.coupling) for lt in H.onsite]
    bond = [TwoSiteTerm(bt.i, bt.j, bt.op_i, bt.op_j, c * bt.coupling) for bt in H.bond]
    LatticeOperator(H.dof, H.geom, onsite, bond)
end
Base.:*(H::LatticeOperator, c::Real) = c * H

"""
    Base.:+(A::LatticeOperator, B::LatticeOperator) -> LatticeOperator

Merge the term lists of two operators (both must share the same DoF and geometry).
"""
function Base.:+(A::LatticeOperator, B::LatticeOperator)
    onsite = vcat(A.onsite, B.onsite)
    bond = vcat(A.bond, B.bond)
    LatticeOperator(A.dof, A.geom, onsite, bond)
end

# ----------------------------------------------------------------------------------------
# Dense matrix from term list — used for testing and small-system ED
# ----------------------------------------------------------------------------------------

"""
    matrix_repr(H::LatticeOperator, [fmt::StorageFormat]) -> AbstractMatrix{ComplexF64}

Materialise `H` as a ``d^L \\times d^L`` matrix, with storage layout chosen by `fmt`.

  - `matrix_repr(H)` or `matrix_repr(H, DenseFormat())` — returns a dense
    `Matrix{ComplexF64}` built by Kronecker product construction (see below).
  - `matrix_repr(H, SparseFormat())` — returns a `SparseMatrixCSC{ComplexF64}`
    assembled via sparse Kronecker products; suitable for Krylov/Lanczos solvers.

For each on-site term ``h_i \\, O_i`` the dense contribution is

```math
I_{d^{i-1}} \\otimes O_i \\otimes I_{d^{L-i}},
```

and for each two-site term ``J_{ij} \\, O_i \\otimes O_j`` (``j > i``) it is

```math
I_{d^{i-1}} \\otimes O_i \\otimes I_{d^{j-i-1}} \\otimes O_j \\otimes I_{d^{L-j}}.
```

!!! warning "Commuting statistics only"

    Kronecker product construction is only correct for bosonic (commuting) DoFs such
    as [`Spin`](@ref) and [`HardCoreBoson`](@ref).  For fermionic DoFs the
    Jordan–Wigner string is not inserted automatically; restrict `matrix_repr` to
    commuting DoFs until the planned basis-change helper is implemented.

# Examples

```jldoctest
julia> H = Heisenberg(Chain(2));

julia> M = matrix_repr(H);

julia> size(M)
(4, 4)

julia> isapprox(M, M'; atol=1e-12)
true

julia> Ms = matrix_repr(H, SparseFormat());

julia> size(Ms)
(4, 4)
```
"""
matrix_repr(H::LatticeOperator) = matrix_repr(H, DenseFormat())

function matrix_repr(H::LatticeOperator, ::DenseFormat)
    # Reject periodic geometries: wrap bond (L,1) has i>j, making d^(j-i-1) negative.
    if any(bt -> bt.i > bt.j, H.bond)
        throw(ArgumentError(
            "matrix_repr(DenseFormat) does not support periodic boundary conditions: " *
            "bond $(first(bt for bt in H.bond if bt.i > bt.j)) has i > j.  " *
            "Use an open-boundary Chain or SparseFormat instead.",
        ))
    end
    L = H.geom.L
    d = local_dim(H.dof)
    N = d^L
    N ≤ 2^20 || throw(ArgumentError(
        "Hilbert space dimension $N = $(d)^$L exceeds the safety limit 2^20. " *
        "Use SparseFormat or an MPS algorithm for large systems.",
    ))
    mat = zeros(ComplexF64, N, N)
    Id(n) = Matrix{ComplexF64}(I, n, n)

    for lt in H.onsite
        i     = lt.site
        left  = d^(i - 1)
        right = d^(L - i)
        mat .+= lt.coupling .* kron(kron(Id(left), ComplexF64.(lt.op)), Id(right))
    end

    for bt in H.bond
        i      = bt.i
        j      = bt.j
        left   = d^(i - 1)
        middle = d^(j - i - 1)
        right  = d^(L - j)
        op2 = middle == 1 ?
            kron(ComplexF64.(bt.op_i), ComplexF64.(bt.op_j)) :
            kron(kron(ComplexF64.(bt.op_i), Id(middle)), ComplexF64.(bt.op_j))
        mat .+= bt.coupling .* kron(kron(Id(left), op2), Id(right))
    end

    return mat
end
