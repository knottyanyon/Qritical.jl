# ==== Orthogonality-centre hierarchy ==========================================
# This file defines TYPE-LEVEL metadata for tracking WHERE the orthogonality
# centre (OC) sits inside an MPS chain. These are NOT computational types —
# they carry no algorithm logic. Think of them as Python Enum-like labels,
# but richer because they can carry data (e.g. which bond, which leg).

"""
    OrthoCenter

Abstract supertype recording **where a centre of orthogonality sits** in a tree tensor network (here, an MPS chain).

Following Evenbly's *center of orthogonality* ([evenbly_2019a](@cite), Def. 3.3): a location is a centre of orthogonality when **every branch attached to it forms an isometry** between the branch's open indices and the index that connects the branch to the centre. For an MPS this specialises to the familiar statement that all site tensors on the left are left-orthogonal (``A^\\dagger A = I``) and all on the right are right-orthogonal (``B B^\\dagger = I``) — a left/right pair of isometric branches.

Concrete subtypes distinguish the two forms the theory treats separately:

  - [`SiteCenter`](@ref) — the centre is a **tensor** ([evenbly_2019a](@cite), Def. 3.3, directly). This form is *not* gauge-unique: a unitary change of gauge on the adjacent bonds leaves it a centre of orthogonality.
  - [`BondCenter`](@ref) — the centre is a **link** carrying a diagonal
    ``\\Sigma`` (the "link-centred" extension of [evenbly_2019b](@cite)). The extra constraint that ``\\Sigma`` be diagonal with positive entries in descending order fully fixes the gauge, so this form *is* unique (up to signs / degenerate subspaces) and is exactly the SVD across that cut.

!!! note "These types are descriptive, not operational"

    `OrthoCenter` values are **metadata** — they record *where* a decomposition's centre lives (see [`BondCenter`](@ref)'s use in `FullSVD`, `ReducedSVD` and [`SchmidtSpectrum`](@ref)). They do **not** drive the left/right-canonical conversion itself; that logic lives in `canonicalize.jl` and is tracked by `CanonicalForm`. Wiring `.center` into that tracker is deferred future work.
"""
# In Julia `abstract type Foo end` is analogous to Python's `class Foo(ABC): pass`. No instances of OrthoCenter can be created — it only exists as a shared supertype for dispatch and type-checking (e.g. `x isa OrthoCenter` works like `isinstance(x, OrthoCenter)`).
abstract type OrthoCenter end   # root of the OC hierarchy; no instances can be created; `x isa OrthoCenter` tests subtype membership

"""
    BondCenter(bond)

**Link-centred** centre of orthogonality: the centre is a diagonal matrix ``\\Sigma`` sitting on a bond, used to mark *which bond* the decomposition's centre lives on. It does not itself perform any re-gauging.

# Fields

  - `bond :: Bond` — the link on which the centre sits; `.bond.left` and
    `.bond.right` are ``\\Sigma``'s own (both `Upper`) faces, whose partners are
    the right leg of ``U`` and the left leg of ``V^\\dagger`` respectively.

# See also

[`SiteCenter`](@ref), [`Bond`](@ref)
"""
# `struct BondCenter <: OrthoCenter` in Julia is like:
#   @dataclass class BondCenter(OrthoCenter): bond: Bond
# The `<:` operator means "is a subtype of" (Python's `(OrthoCenter)` in the class definition).
# By default, Julia structs are IMMUTABLE — once created you cannot change their fields.
# (Use `mutable struct` if you want mutability, analogous to a regular Python object.)
struct BondCenter <: OrthoCenter   # concrete subtype of OrthoCenter; immutable; `<: OrthoCenter` = Python `class BondCenter(OrthoCenter)`
    # `bond::Bond` declares a field named `bond` with type `Bond`.
    # In Python this would be a type-annotated attribute: bond: Bond
    # Physics: `bond` records which bond (left-right pair of indices) is the OC link.
    # The singular-value matrix Σ sits exactly on this bond in the decomposition.
    bond::Bond   # which bond is the gauge centre; `.bond.left` and `.bond.right` are the same TIx objects as Σ's legs — no label matching needed downstream
end

"""
    SiteCenter(leg)

**Tensor-centred** centre of orthogonality ([evenbly_2019a](@cite), Def. 3.3): the centre is a single site tensor, isometric on neither side, whose left and right branches are each isometries onto the bond that connects them to it. This form arises after the gauge is swept to a specific site.

Unlike [`BondCenter`](@ref), this form does **not** fully fix the gauge: a unitary change of gauge on the adjacent bonds leaves the site a valid centre of orthogonality. It is the SVD-unique link matrix that removes that remaining freedom.

Like [`BondCenter`](@ref), this type is a **metadata record** of *where* the centre sits; it does not itself re-gauge the network.

# Fields

  - `leg :: TIx{Upper}` — the physical leg of the site tensor that is the current centre. Physical legs are `Upper` (contravariant coefficients), and on the centre site every leg is `Upper` — both bond arrows point in.

# See also

[`BondCenter`](@ref), [`OrthoCenter`](@ref)
"""
struct SiteCenter <: OrthoCenter   # concrete subtype of OrthoCenter; records that the OC is at a single site tensor (not a bond)
    # `TIx{Upper}` is a PARAMETRIC TYPE — Julia's equivalent of a Python generic like `TIx[Upper]`.
    # `TIx` is a "tensor index" type, and `{Upper}` specialises it to the Upper (contravariant) variance.
    # Physics: at the orthogonality centre, both virtual bond arrows point INTO the site,
    # so the physical leg is Upper (contravariant) just like at any site — but here
    # *all three legs* are Upper, which distinguishes the centre from left/right-canonical sites.
    leg::TIx{Upper}   # the physical leg of the centre site tensor; `TIx{Upper}` = a typed index with contravariant (incoming arrow) variance; labels which physical degree of freedom is at the OC
end
