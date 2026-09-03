#=META
source:
  author: Bavithra
  coauthor: N/A
  reviewer:
docstrings:
  author: Bavithra
  coauthor:
  reviewer:
refs:
credits: N/A
=#

using TensorKit

# SECTION - LegRole: the general role a TIx/MulTIx plays

"""
    LegRole

Abstract root for the role a [`TIx`](@ref)/[`MulTIx`](@ref) plays within a tensor network.
Every index is fundamentally just a leg - even one built from only trivial sectors is a
perfectly good leg, it simply carries no extra structure info worth asking about. Concrete
subtypes name the roles Qritical.jl currently distinguishes: [`PhysicalLeg`](@ref) (an ordinary
site index) and [`VirtualLeg`](@ref) (a bond index, living across an entanglement cut).
"""
abstract type LegRole end

"""
    PhysicalLeg <: LegRole

An ordinary physical (site) leg. Carries no [`EntanglementStructure`](@ref) info: there is no
cut associated with a single physical leg, so no Schmidt spectrum to extract from it alone.
"""
struct PhysicalLeg <: LegRole end

"""
    VirtualLeg <: LegRole

A virtual (bond) leg, living across an entanglement cut - e.g. the leg connecting two sites of
an MPS. Carries [`EntanglementStructure`](@ref) info: its singular value spectrum across that
cut is a meaningful, extractable quantity.
"""
struct VirtualLeg <: LegRole end

# SECTION - StructureInfo: the general question "does this leg carry X info?"

"""
    StructureInfo

Abstract root of the "does this leg carry X info?" trait family (the Holy traits pattern).
"Carrying structure info" means more than
just describing the leg: it means something extra is *accessible* on it. A leg whose `space`
carries real sector data means TensorKit's own symmetry machinery can be used to simplify a
calculation on it ([`SymmetryStructure`](@ref)); a leg living across an entanglement cut means
its singular value spectrum can be extracted ([`EntanglementStructure`](@ref)). Each kind of
structure info is queried the same way on both [`TIx`](@ref) and [`MulTIx`](@ref).
"""
abstract type StructureInfo end

# SECTION - SymmetryStructure: decided by the space's TensorKit.sectortype

"""
    SymmetryStructure <: StructureInfo

Whether a leg's `space` carries real sector data. [`CarriesSymmetryInfo`](@ref) if
`TensorKit.sectortype(space) !== TensorKit.Trivial`, [`NoSymmetryInfo`](@ref) otherwise (the
trivial/ungraded case, e.g. a plain `ComplexSpace`). Computed via [`symmetry_structure`](@ref);
queried as a `Bool` via [`carries_symmetry_info`](@ref).
"""
abstract type SymmetryStructure <: StructureInfo end

"""
    CarriesSymmetryInfo <: SymmetryStructure

The leg's `space` carries real sector data (e.g. a `GradedSpace`), so TensorKit's symmetry
machinery is available to simplify calculations on it.
"""
struct CarriesSymmetryInfo <: SymmetryStructure end

"""
    NoSymmetryInfo <: SymmetryStructure

The leg's `space` is trivial/ungraded (e.g. a plain `ComplexSpace`): there is no sector
structure to exploit.
"""
struct NoSymmetryInfo <: SymmetryStructure end

"""
    symmetry_structure(::Type{<:TensorKit.ElementarySpace}) -> SymmetryStructure

Which [`SymmetryStructure`](@ref) a `TensorKit.ElementarySpace` type carries, decided by
`TensorKit.sectortype`.

# Examples

```jldoctest
julia> symmetry_structure(ComplexSpace)
NoSymmetryInfo()
```
"""
function symmetry_structure(::Type{S}) where {S<:TensorKit.ElementarySpace}
    return if TensorKit.sectortype(S) === TensorKit.Trivial
        NoSymmetryInfo()
    else
        CarriesSymmetryInfo()
    end
end

"""
    carries_symmetry_info(x) -> Bool

`true` if `x` carries [`CarriesSymmetryInfo`](@ref), delegating to [`symmetry_structure`](@ref).
"""
carries_symmetry_info(x) = _carries_symmetry(symmetry_structure(x))
_carries_symmetry(::CarriesSymmetryInfo) = true
_carries_symmetry(::NoSymmetryInfo) = false

# SECTION - EntanglementStructure: decided by LegRole

"""
    EntanglementStructure <: StructureInfo

Whether a leg sits across an entanglement cut, i.e. whether its singular value spectrum is a
meaningful quantity to extract. [`CarriesEntanglementInfo`](@ref) for a [`VirtualLeg`](@ref),
[`NoEntanglementInfo`](@ref) for a [`PhysicalLeg`](@ref). Computed via
[`entanglement_structure`](@ref); queried as a `Bool` via [`carries_entanglement_info`](@ref).
"""
abstract type EntanglementStructure <: StructureInfo end

"""
    CarriesEntanglementInfo <: EntanglementStructure

The leg lives across an entanglement cut ([`VirtualLeg`](@ref)): its Schmidt spectrum is
extractable.
"""
struct CarriesEntanglementInfo <: EntanglementStructure end

"""
    NoEntanglementInfo <: EntanglementStructure

The leg is an ordinary [`PhysicalLeg`](@ref): no cut, no Schmidt spectrum to extract from it
alone.
"""
struct NoEntanglementInfo <: EntanglementStructure end

"""
    entanglement_structure(::Type{<:LegRole}) -> EntanglementStructure

Which [`EntanglementStructure`](@ref) a [`LegRole`](@ref) type carries: [`NoEntanglementInfo`](@ref)
for [`PhysicalLeg`](@ref), [`CarriesEntanglementInfo`](@ref) for [`VirtualLeg`](@ref).
"""
entanglement_structure(::Type{<:PhysicalLeg}) = NoEntanglementInfo()
entanglement_structure(::Type{<:VirtualLeg}) = CarriesEntanglementInfo()

"""
    carries_entanglement_info(x) -> Bool

`true` if `x` carries [`CarriesEntanglementInfo`](@ref), delegating to
[`entanglement_structure`](@ref).
"""
carries_entanglement_info(x) = _carries_entanglement(entanglement_structure(x))
_carries_entanglement(::CarriesEntanglementInfo) = true
_carries_entanglement(::NoEntanglementInfo) = false
