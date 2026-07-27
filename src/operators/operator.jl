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
uniform_coupling(n::Int, x) = fill(x, n)   # `fill(x, n)` creates a length-n vector with all elements equal to x. used to broadcast a scalar coupling to all bonds

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
struct OneSiteTerm{O}   # parametric struct: `{O}` is a type parameter for the operator matrix type; this allows `op` to be any matrix type (Matrix, Hermitian, Diagonal, etc.) without boxing; in Python this would just be `self.op: Any`
    site::Int         # 1-indexed site number; `Int` = Julia's default integer (64-bit on 64-bit systems)
    op::O             # the d×d operator matrix acting at this site; type is determined by the constructor (statically typed)
    coupling::Float64 # the prefactor h_i for this term; `Float64` = 64-bit float = Python `float`
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
struct TwoSiteTerm{O1,O2}   # two type parameters O1 and O2 for the left and right operator types respectively; they may differ (e.g. O1=Sp matrix, O2=Sm matrix for the flip-flop term)
    i::Int            # left site index (1-indexed)
    j::Int            # right site index; always j > i for open boundary chains (otherwise MPO FSM breaks)
    op_i::O1          # d×d operator matrix at site i; type O1 captured from constructor
    op_j::O2          # d×d operator matrix at site j; type O2 may differ from O1 (e.g. S+ and S-)
    coupling::Float64 # bond coupling J_ij; positive = ferromagnetic, negative = antiferromagnetic (by convention)
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
  - `G <: AbstractLayout`   — the lattice geometry (sets site and bond lists).
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
struct LatticeOperator{D<:AbstractDoF,G<:AbstractLayout,LT,BT}   # four type parameters; `D<:AbstractDoF` constrains D to subtypes of AbstractDoF (like Python `D: AbstractDoF`); Julia encodes the full type information at compile time, so different models (XXZ vs Ising) produce different concrete types
    dof::D               # the degree of freedom (e.g. SpinHalf()); determines the local operator algebra and dimension d
    geom::G              # the lattice geometry (e.g. Chain(L)); determines site count and bond connectivity
    onsite::Vector{LT}   # list of on-site terms; `Vector{LT}` is a typed vector of OneSiteTerm instances
    bond::Vector{BT}     # list of two-site bond terms; typed vector of TwoSiteTerm instances
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
const Hamiltonian = LatticeOperator   # `const` makes this binding immutable (re-assigning would warn); `=` creates a type alias: Hamiltonian and LatticeOperator are the SAME type (Python has no native type alias syntax; closest is `Hamiltonian = LatticeOperator` but Julia's is enforced at the type level)

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

  - `LatticeOperator` with `dof = SpinHalf()`, suitable for `MPO(H)` or [`matrix_repr`](@ref).

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
function XXZ(g::Chain; J=1.0, Jz=1.0, h=0.0)   # named constructor for the XXZ model; `;` marks keyword arguments (J, Jz, h); all have default values so XXZ(g) uses Heisenberg couplings
    nb = length(bonds(g))   # number of bonds = L-1 for open boundary chain; `bonds(g)` returns a list of (i,j) pairs; `length(...)` = Python `len(...)`
    Jxy = J isa Number ? uniform_coupling(nb, Float64(J)) : Float64.(J)   # if J is a scalar, replicate it to all bonds; `isa Number` checks if J is any numeric type. `Float64(J)` converts scalar to Float64; `Float64.(J)` broadcasts element-wise conversion on a vector 
    Jzv = Jz isa Number ? uniform_coupling(nb, Float64(Jz)) : Float64.(Jz)   # same for Jz: scalar → uniform vector, vector → Float64 conversion
    hv = h isa Number ? uniform_coupling(g.L, Float64(h)) : Float64.(h)   # magnetic field: if scalar, replicate to all L sites (not just bonds); `g.L` is the site count
    ops = algebra_generators(SpinHalf())   # get the spin-½ operator algebra: returns a NamedTuple with fields Sx, Sy, Sz, Sp, Sm, I; physics: S+ = [[0,1],[0,0]], S- = [[0,0],[1,0]], Sz = [[1/2,0],[0,-1/2]]

    onsite = [OneSiteTerm(i, ops.Sz, -hv[i]) for i in sites(g)]   # create one on-site term per site: coupling = -h_i, operator = Sz; `sites(g)` returns 1:g.L; the minus sign implements the convention -h·Sz; comprehension creates a Vector{OneSiteTerm}

    bond = vcat(   # `vcat(a, b, c)` concatenates arrays/vectors vertically. creates the combined list of all bond terms
        [
            TwoSiteTerm(i, j, ops.Sp, ops.Sm, 0.5Jxy[b]) for
            (b, (i, j)) in enumerate(bonds(g))
        ],   # S+_i S-_j terms: the (J/2)·flip-flop part; `enumerate(bonds(g))` returns (index, (i,j)) pairs — same as Python's enumerate; `0.5Jxy[b]` = J/2 for bond b (Julia: numeric literal × variable = multiply, no `*` needed)
        [
            TwoSiteTerm(i, j, ops.Sm, ops.Sp, 0.5Jxy[b]) for
            (b, (i, j)) in enumerate(bonds(g))
        ],   # S-_i S+_j terms: the other half of the flip-flop; together S+S- + S-S+ = 2(Sx⊗Sx + Sy⊗Sy) which is the transverse exchange
        [TwoSiteTerm(i, j, ops.Sz, ops.Sz, Jzv[b]) for (b, (i, j)) in enumerate(bonds(g))],   # Sz_i Sz_j terms: the Ising (longitudinal) part with full coupling Jz (no factor of 1/2)
    )

    LatticeOperator(SpinHalf(), g, onsite, bond)   # construct the LatticeOperator; Julia infers the type parameters D, G, LT, BT from the argument types automatically
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
Heisenberg(g::Chain; J=1.0, h=0.0) = XXZ(g; J=J, Jz=J, h=h)   # isotropic case: set Jz=J so S+S-+S-S+ + 2*Jz*SzSz = J·(SxSx+SySy+SzSz) = J·S⃗·S⃗; one-liner delegates to XXZ with Jz=J

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
function Ising(g::Chain; J=1.0, h=0.0)   # transverse-field Ising: field couples to Sx (quantum fluctuations), Ising interaction along z
    nb = length(bonds(g))   # number of bonds L-1
    Jxy = J isa Number ? uniform_coupling(nb, Float64(J)) : Float64.(J)   # longitudinal (Ising) coupling per bond
    hv = h isa Number ? uniform_coupling(g.L, Float64(h)) : Float64.(h)   # transverse field per site; note: field is on all L sites, not just L-1 bonds
    ops = algebra_generators(SpinHalf())   # get spin-½ operators

    onsite = [OneSiteTerm(i, ops.Sx, -hv[i]) for i in sites(g)]   # transverse field: -h·Sx at each site; note Sx NOT Sz (hence "transverse"); this is what makes the Ising model quantum (classical Ising has no transverse field)
    bond = [
        TwoSiteTerm(i, j, ops.Sz, ops.Sz, Jxy[b]) for (b, (i, j)) in enumerate(bonds(g))
    ]   # longitudinal Ising interaction J·Sz⊗Sz on each NN bond; single term type unlike XXZ

    LatticeOperator(SpinHalf(), g, onsite, bond)   # construct the LatticeOperator
end

# ----------------------------------------------------------------------------------------
# Observable constructors (§5 / §7)
# ----------------------------------------------------------------------------------------

"""
    total_magnetization(g::AbstractLayout; dof=SpinHalf()) -> LatticeOperator

Build the total magnetization operator ``M = \\sum_i S^z_i``.

This is a conserved quantity of the XXZ and Heisenberg models (it commutes with
the Hamiltonian when ``h = 0``).  Measuring it after a variational optimisation
is a useful sanity check: the ground state of an antiferromagnet at zero field
on an even-``L`` chain should have ``\\langle M \\rangle = 0``.

# Arguments

  - `g::AbstractLayout` — lattice geometry.
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
function total_magnetization(g::AbstractLayout; dof=SpinHalf())   # observable constructor; accepts any geometry type (Chain, ring, etc.) via `AbstractLayout`; `dof=SpinHalf()` default keyword arg
    ops = algebra_generators(dof)   # get operator algebra for the given DoF
    LatticeOperator(dof, g, [OneSiteTerm(i, ops.Sz, 1.0) for i in sites(g)], TwoSiteTerm[])   # build the operator: +1·Sz at every site, empty bond list; `TwoSiteTerm[]` is a typed empty vector. `sites(g)` returns the site indices 1:L
end

"""
    staggered_magnetization(g::AbstractLayout; dof=SpinHalf()) -> LatticeOperator

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

  - `g::AbstractLayout` — lattice geometry.
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
function staggered_magnetization(g::AbstractLayout; dof=SpinHalf())   # Néel order parameter: alternating ±1 couplings
    ops = algebra_generators(dof)   # get operator algebra
    LatticeOperator(
        dof, g, [OneSiteTerm(i, ops.Sz, (-1.0)^i) for i in sites(g)], TwoSiteTerm[]
    )   # `(-1.0)^i` is (-1)^i: +1 for even i, -1 for odd i; physics: staggered pattern detects antiferromagnetic order where ↑↓↑↓ gives ∑(-)^i Sz_i = L/2 > 0
end

"""
    op_at_site(g::AbstractLayout, dof::AbstractDoF, label::Symbol, site::Int) -> LatticeOperator

Build a single-site observable: the operator named `label` of `dof` at `site`,
embedded in the full many-body Hilbert space defined by geometry `g`.

The geometry `g` determines the total Hilbert-space dimension ``d^L`` that
`matrix_repr` will construct.  Without it the Kronecker-product chain cannot know
how many identity factors to add on either side of the operator at `site`.

Use this to measure any on-site quantity — ``S^z_i``, ``n_i``, ``c_i``, etc.
— after a variational calculation.  The returned `LatticeOperator` can be passed to
[`expect`](@ref) by first converting it to a `FiniteMPO` via `MPO`.

# Arguments

  - `g::AbstractLayout` — the full lattice geometry (e.g. `Chain(L)`).
  - `dof::AbstractDoF`    — the local degree of freedom (determines the operator algebra).
  - `label::Symbol`       — name of the operator to retrieve from [`algebra_generators`](@ref),
    e.g. `:Sz`, `:n`, `:cup`.
  - `site::Int`           — which lattice site the operator acts on (1-indexed).

# Returns

  - `LatticeOperator` on geometry `g` with a single `OneSiteTerm` (coupling = 1)
    and no bond terms.

# Examples

```jldoctest
julia> g = Chain(4);

julia> O = op_at_site(g, SpinHalf(), :Sz, 2);

julia> O.onsite[1].site
2

julia> O.onsite[1].coupling
1.0

julia> size(matrix_repr(O))
(16, 16)
```
"""
function op_at_site(g::AbstractLayout, dof::AbstractDoF, label::Symbol, site::Int)   # `label::Symbol` requires a Julia Symbol (`:Sz`, `:Sx`, etc.); Symbols are interned strings used as identifiers 
    op = getproperty(algebra_generators(dof), label)   # `getproperty(obj, :field)` dynamically accesses a field by name. `algebra_generators(dof)` returns a NamedTuple; `label` is a Symbol like `:Sz`; this lets us write `op_at_site(g, dof, :Sz, 2)` without hardcoding the operator
    LatticeOperator(dof, g, [OneSiteTerm(site, op, 1.0)], TwoSiteTerm[])   # single on-site term with coupling 1.0; `TwoSiteTerm[]` empty bond list
end

"""
    two_site_op(g::AbstractLayout, dof::AbstractDoF,
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
    in the current version; use [`matrix_repr`](@ref) on small systems or handle
    JW strings by hand for fermionic models.

# Arguments

  - `g::AbstractLayout` — lattice geometry (used to construct the `LatticeOperator`
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

julia> O = two_site_op(g, SpinHalf(), :Sz, 1, :Sz, 4);

julia> length(O.bond)
1

julia> O.bond[1].i, O.bond[1].j
(1, 4)
```
"""
function two_site_op(
    g::AbstractLayout,
    dof::AbstractDoF,
    opA::Symbol,
    iA::Int,
    opB::Symbol,
    iB::Int,   # 6 positional arguments; Symbol for operator names allows dynamic lookup in algebra_generators
)
    ops = algebra_generators(dof)   # get full operator algebra
    Amat = getproperty(ops, opA)   # look up operator A by name. e.g. opA=:Sz → Amat = ops.Sz
    Bmat = getproperty(ops, opB)   # look up operator B by name
    LatticeOperator(dof, g, OneSiteTerm[], [TwoSiteTerm(iA, iB, Amat, Bmat, 1.0)])   # single bond term with coupling 1.0; `OneSiteTerm[]` empty on-site list; note: sites iA, iB need NOT be NN — the MPO FSM handles non-adjacent pairs via channel carry
end

"""
    identity_operator(g::AbstractLayout, dof::AbstractDoF) -> LatticeOperator

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
function identity_operator(g::AbstractLayout, dof::AbstractDoF)   # no terms needed: ⟨ψ|I|ψ⟩ = ‖ψ‖²; the MPO constructor handles the empty-terms special case by building a χ=1 all-identity MPO
    LatticeOperator(dof, g, OneSiteTerm[], TwoSiteTerm[])   # both lists empty; `OneSiteTerm[]` and `TwoSiteTerm[]` are typed empty vectors needed so Julia can infer the LT and BT type parameters correctly
end

# ----------------------------------------------------------------------------------------
# LatticeOperator arithmetic
# ----------------------------------------------------------------------------------------

"""
    Base.:*(c::Real, H::LatticeOperator) -> LatticeOperator
    Base.:*(H::LatticeOperator, c::Real) -> LatticeOperator

Scale all couplings of `H` by scalar `c`.
"""
function Base.:*(c::Real, H::LatticeOperator)   # `Base.:*` extends Julia's built-in multiplication operator for a new type. putting it in `Base` makes `c * H` work with the standard `*` syntax; `Base.:*` is the operator function in Julia's Base module
    onsite = [OneSiteTerm(lt.site, lt.op, c * lt.coupling) for lt in H.onsite]   # create new OneSiteTerm list with scaled couplings; immutable structs can't be modified, so we create new ones; the operator matrix `lt.op` is shared (not copied) for efficiency
    bond = [TwoSiteTerm(bt.i, bt.j, bt.op_i, bt.op_j, c * bt.coupling) for bt in H.bond]   # same for bond terms: scale couplings, keep operator matrices
    LatticeOperator(H.dof, H.geom, onsite, bond)   # construct new LatticeOperator; dof and geom are shared references (not copied)
end
Base.:*(H::LatticeOperator, c::Real) = c * H   # commutative case: H * c = c * H; one-liner delegates to the above; this makes `H * 2.0` work as well as `2.0 * H`

"""
    Base.:+(A::LatticeOperator, B::LatticeOperator) -> LatticeOperator

Merge the term lists of two operators (both must share the same DoF and geometry).
"""
function Base.:+(A::LatticeOperator, B::LatticeOperator)   # extends `+` for LatticeOperator. adding two operators merges their term lists
    onsite = vcat(A.onsite, B.onsite)   # concatenate on-site term lists; `vcat` merges vectors. the result is a new Vector containing all terms from both
    bond = vcat(A.bond, B.bond)   # concatenate bond term lists
    LatticeOperator(A.dof, A.geom, onsite, bond)   # construct merged operator; uses A's dof and geom (they should match B's)
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
matrix_repr(H::LatticeOperator) = matrix_repr(H, DenseFormat())   # default method: no format argument → dense; delegates to the 2-argument version; this is Julia's way of providing default dispatch.

function matrix_repr(H::LatticeOperator, ::DenseFormat)   # dense Kronecker-product construction; builds the full D×D matrix explicitly; only feasible for small systems (D ≤ 2^20)
    # Reject periodic geometries: wrap bond (L,1) has i>j, making d^(j-i-1) negative.
    if any(bt -> bt.i > bt.j, H.bond)   # `any(predicate, collection)` returns true if any element satisfies the predicate. periodic boundary conditions create a wrap bond (L,1) where i>j, which breaks the power-of-d formula below
        throw(
            ArgumentError(
                "matrix_repr(DenseFormat) does not support periodic boundary conditions: " *
                "bond $(first(bt for bt in H.bond if bt.i > bt.j)) has i > j.  " *   # `first(...)` finds the first matching element. embedded in string interpolation `$(...)`
                "Use an open-boundary Chain or SparseFormat instead.",
            ),
        )
    end
    L = H.geom.L   # number of sites
    d = local_dim(H.dof)   # local dimension
    N = d^L   # total Hilbert space dimension
    N ≤ 2^20 || throw(
        ArgumentError(   # same safety guard as sparse version; `2^20` is a literal integer exponentiation
            "Hilbert space dimension $N = $(d)^$L exceeds the safety limit 2^20. " *
            "Use SparseFormat or an MPS algorithm for large systems.",
        ),
    )
    mat = zeros(ComplexF64, N, N)   # allocate dense N×N zero matrix. will accumulate all operator contributions
    Id(n) = Matrix{ComplexF64}(I, n, n)   # local helper function: creates an n×n identity matrix; defined inside the outer function.uses `I` (lazy identity) materialised as a concrete Matrix

    for lt in H.onsite   # iterate over on-site terms
        i = lt.site   # site index (1-indexed)
        left = d^(i - 1)   # dimension of identity block to the LEFT of site i: d^{i-1}; physics: sites 1..i-1 are not acted on
        right = d^(L - i)   # dimension of identity block to the RIGHT of site i: d^{L-i}; physics: sites i+1..L are not acted on
        mat .+= lt.coupling .* kron(kron(Id(left), ComplexF64.(lt.op)), Id(right))   # build I_{d^{i-1}} ⊗ O_i ⊗ I_{d^{L-i}}; nested kron calls build left-to-right; `.+=` broadcasts in-place addition
    end

    for bt in H.bond   # iterate over bond terms
        i = bt.i   # left site
        j = bt.j   # right site
        left = d^(i - 1)   # left identity block dimension
        middle = d^(j - i - 1)   # intermediate identity block between sites i and j: d^{j-i-1}; for NN bonds (j=i+1), middle=d^0=1 (no intermediate sites)
        right = d^(L - j)   # right identity block dimension
        op2 = if middle == 1   # check if there are any intermediate sites between i and j
            kron(ComplexF64.(bt.op_i), ComplexF64.(bt.op_j))   # NN case (j=i+1): directly kron O_i ⊗ O_j (no intermediate identity needed)
        else   # NN case (j=i+1): directly kron O_i ⊗ O_j (no intermediate identity needed)
            kron(kron(ComplexF64.(bt.op_i), Id(middle)), ComplexF64.(bt.op_j))   # non-NN case: O_i ⊗ I_{middle} ⊗ O_j (insert identity for the sites between i and j)
        end   # non-NN case: O_i ⊗ I_{middle} ⊗ O_j (insert identity for the sites between i and j)
        mat .+= bt.coupling .* kron(kron(Id(left), op2), Id(right))   # embed into full Hilbert space: I_{left} ⊗ op2 ⊗ I_{right}
    end

    return mat   # return the assembled dense complex Hamiltonian matrix
end
