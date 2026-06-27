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
    uniform(n, x) -> Vector

Return a length-`n` vector with every entry equal to `x`.
"""
uniform(n::Int, x) = fill(x, n)

# ─────────────────────────────────────────────────────────────────────────────
# Term types
# ─────────────────────────────────────────────────────────────────────────────

"""
    LocalTerm{O}

A single-site contribution `coupling · op` at `site`.
"""
struct LocalTerm{O}
    site::Int
    op::O
    coupling::Float64
end

"""
    BondTerm{O1,O2}

A two-site contribution `coupling · (op_i ⊗ op_j)` at sites `i` and `j`.
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

A linear operator as a sum of on-site (`LocalTerm`) and two-site (`BondTerm`)
contributions over a DoF `D` and geometry `G`.

`Hamiltonian` is an alias: the role (dynamics generator vs. observable) is
determined by how the instance is passed to `solve` or `expect`, not by type.
"""
struct Operator{D<:AbstractDoF, G<:AbstractGeometry, LT, BT}
    dof::D
    geom::G
    onsite::Vector{LT}
    bond::Vector{BT}
end

const Hamiltonian = Operator

# ─────────────────────────────────────────────────────────────────────────────
# Named constructors — spin models (§7)
# ─────────────────────────────────────────────────────────────────────────────

"""
    XXZ(g::Chain; J=1.0, Jz=1.0, h=0.0) -> Operator

XXZ spin-½ chain: H = ½J Σ (S⁺ᵢS⁻ⱼ + S⁻ᵢS⁺ⱼ) + Jz Σ SᶻᵢSᶻⱼ − h Σ Sᶻᵢ.

Course convention (Ex 3/6): the field term is `−hᵢSᶻ`.
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

Isotropic Heisenberg chain: XXZ with `Jz = J`.
"""
Heisenberg(g::Chain; J=1.0, h=0.0) = XXZ(g; J=J, Jz=J, h=h)

"""
    Ising(g::Chain; J=1.0, h=0.0) -> Operator

Transverse-field Ising chain: H = J Σ SᶻᵢSᶻⱼ − h Σ Sˣᵢ.
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
    total_magnetization(g; dof=SpinHalf()) -> Operator

M = Σᵢ Sᶻᵢ  (spin-½ default).
"""
function total_magnetization(g::AbstractGeometry; dof=SpinHalf())
    ops = operators(dof)
    Operator(dof, g, [LocalTerm(i, ops.Sz, 1.0) for i in sites(g)], BondTerm[])
end

"""
    staggered_magnetization(g; dof=SpinHalf()) -> Operator

Mˢ = Σᵢ (−1)ⁱ Sᶻᵢ  (Néel order parameter).
"""
function staggered_magnetization(g::AbstractGeometry; dof=SpinHalf())
    ops = operators(dof)
    Operator(dof, g, [LocalTerm(i, ops.Sz, (-1.0)^i) for i in sites(g)], BondTerm[])
end

"""
    local_op(dof, sym, site) -> Operator

Single-site observable: the named operator `sym` of `dof` at `site`.
"""
function local_op(dof::AbstractDoF, sym::Symbol, site::Int)
    op  = getproperty(operators(dof), sym)
    # Minimal geometry: a chain containing just this site (no bonds needed)
    g   = Chain(site)
    Operator(dof, g, [LocalTerm(site, op, 1.0)], BondTerm[])
end

"""
    two_point(g, dof, opA, iA, opB, iB) -> Operator

The operator Aᵢ Bⱼ — a single two-site term whose expectation gives ⟨AᵢBⱼ⟩.
"""
function two_point(g::AbstractGeometry, dof::AbstractDoF,
                   opA::Symbol, iA::Int, opB::Symbol, iB::Int)
    ops  = operators(dof)
    Amat = getproperty(ops, opA)
    Bmat = getproperty(ops, opB)
    Operator(dof, g, LocalTerm[], [BondTerm(iA, iB, Amat, Bmat, 1.0)])
end

"""
    identity_operator(g, dof) -> Operator

The identity operator I = I₁⊗I₂⊗…⊗I_L. Stored as an empty term list;
`MPO(identity_operator(...))` returns a bond-dim-1 all-identity MPO.
"""
function identity_operator(g::AbstractGeometry, dof::AbstractDoF)
    Operator(dof, g, LocalTerm[], BondTerm[])
end

# ─────────────────────────────────────────────────────────────────────────────
# Dense matrix from term list — used for testing and small-system ED
# ─────────────────────────────────────────────────────────────────────────────

"""
    dense_matrix(H::Operator) -> Matrix{ComplexF64}

Build the full d^L × d^L dense Hamiltonian matrix from the term list.

Only valid for commuting-statistics DoFs (spins, hard-core bosons) or when
Jordan–Wigner strings are already embedded in the operators. For fermionic
DoFs use `basis_change(H, SpinHalf())` first.
"""
function dense_matrix(H::Operator)
    L = H.geom.L
    d = local_dim(H.dof)
    N = d^L
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
