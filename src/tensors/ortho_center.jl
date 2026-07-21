# ==== Orthogonality-centre hierarchy ==========================================

"""
    OrthoCenter

Abstract supertype for the **orthogonality centre** of an MPS: the unique
tensor (or bond) for which all tensors to the left are left-orthogonal and all
tensors to the right are right-orthogonal.

Concrete subtypes:
- [`BondCenter`](@ref) — centre sits on a bond (both faces of one link).
- [`SiteCenter`](@ref) — centre sits on a single physical site tensor.

The subtype encodes whether a canonical-form conversion needs to swap the
arrow direction on a bond (`BondCenter`) or absorb singular values into a
site tensor (`SiteCenter`). Dispatch on `OrthoCenter` is exhaustive over
these two cases.
"""
abstract type OrthoCenter end

"""
    BondCenter(bond)

Orthogonality centre located on a **bond**: both faces of one link carry the
gauge freedom.  Conversion to left-canonical shifts the centre rightward
(absorbing ``\\Sigma`` into ``V^\\dagger``); conversion to right-canonical shifts
it leftward (absorbing ``\\Sigma`` into ``U``).

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

Orthogonality centre located on a **site**: the centre tensor carries a
physical leg and is neither left- nor right-orthogonal.  This form arises
after `move_center!` sweeps the gauge to a specific site.

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
