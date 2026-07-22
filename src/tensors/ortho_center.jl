# ==== Orthogonality-centre hierarchy ==========================================

"""
    OrthoCenter

Abstract supertype recording **where a centre of orthogonality sits** in a
tree tensor network (here, an MPS chain).

Following Evenbly's *center of orthogonality* ([evenbly_2019a](@cite),
Def. 3.3): a location is a centre of orthogonality when **every branch attached
to it forms an isometry** between the branch's open indices and the index that
connects the branch to the centre. For an MPS this specialises to the familiar
statement that all site tensors on the left are left-orthogonal
(``A^\\dagger A = I``) and all on the right are right-orthogonal
(``B B^\\dagger = I``) — a left/right pair of isometric branches.

Concrete subtypes distinguish the two forms the theory treats separately:

  - [`SiteCenter`](@ref) — the centre is a **tensor** ([evenbly_2019a](@cite),
    Def. 3.3, directly). This form is *not* gauge-unique: a unitary change of
    gauge on the adjacent bonds leaves it a centre of orthogonality.
  - [`BondCenter`](@ref) — the centre is a **link** carrying a diagonal
    ``\\Sigma`` (the "link-centred" extension of [evenbly_2019b](@cite)). The
    extra constraint that ``\\Sigma`` be diagonal with positive entries in
    descending order fully fixes the gauge, so this form *is* unique (up to
    signs / degenerate subspaces) and is exactly the SVD across that cut.

!!! note "These types are descriptive, not operational"

    `OrthoCenter` values are **metadata** — they record *where* a decomposition's
    centre lives (see [`BondCenter`](@ref)'s use in `FullSVD`, `ReducedSVD` and
    [`SchmidtSpectrum`](@ref)). They do **not** drive the left/right-canonical
    conversion itself; that logic lives in `canonicalize.jl` and is tracked by
    `CanonicalForm`. Wiring `.center` into that tracker is deferred future work.
"""
abstract type OrthoCenter end

"""
    BondCenter(bond)

**Link-centred** centre of orthogonality ([evenbly_2019b](@cite)): a diagonal
link matrix ``\\Sigma`` sits on one bond, the branches on either side of it are
isometries onto ``\\Sigma``'s two legs, and ``\\Sigma`` is diagonal with positive
entries in descending order. That descending-diagonal constraint fully fixes
the gauge, making this form unique (up to signs / degenerate subspaces) and
equivalent to the SVD ``A = U \\Sigma V^\\dagger`` across the cut.

Conversion to left-canonical shifts the centre rightward (absorbing ``\\Sigma``
into ``V^\\dagger``); conversion to right-canonical shifts it leftward (absorbing
``\\Sigma`` into ``U``).

This type is a **metadata record**: it is stored on `FullSVD`, `ReducedSVD` and
[`SchmidtSpectrum`](@ref) to mark *which bond* the decomposition's centre lives
on. It does not itself perform any re-gauging.

# Fields

  - `bond :: Bond` — the link on which the centre sits; `.bond.left` and
    `.bond.right` are ``\\Sigma``'s own (both `Upper`) faces, whose partners are
    the right leg of ``U`` and the left leg of ``V^\\dagger`` respectively.

# See also

[`SiteCenter`](@ref), [`Bond`](@ref)
"""
struct BondCenter <: OrthoCenter
    bond::Bond
end

"""
    SiteCenter(leg)

**Tensor-centred** centre of orthogonality ([evenbly_2019a](@cite), Def. 3.3):
the centre is a single site tensor, isometric on neither side, whose left and
right
branches are each isometries onto the bond that connects them to it. This form
arises after the gauge is swept to a specific site.

Unlike [`BondCenter`](@ref), this form does **not** fully fix the gauge: a
unitary change of gauge on the adjacent bonds leaves the site a valid centre of
orthogonality. It is the SVD-unique link matrix that removes that remaining
freedom.

Like [`BondCenter`](@ref), this type is a **metadata record** of *where* the
centre sits; it does not itself re-gauge the network.

# Fields

  - `leg :: TIx{Upper}` — the physical leg of the site tensor that is the
    current centre. Physical legs are `Upper` (contravariant coefficients), and
    on the centre site every leg is `Upper` — both bond arrows point in.

# See also

[`BondCenter`](@ref), [`OrthoCenter`](@ref)
"""
struct SiteCenter <: OrthoCenter
    leg::TIx{Upper}
end
