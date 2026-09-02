#=META
source:
  author: Bavithra
  coauthor: Claude Opus 5
  reviewer:
docstrings:
  author: Bavithra
  coauthor: Claude Opus 5
  reviewer:
refs:
credits: N/A
=#

# This file is deliberately independent of `MPState`/`QProcess`: it defines the gauge-form and
# boundary-condition *vocabulary* only (singleton types + the coarse GaugeFreedom trait), in the
# same spirit `LinearAlgebra.Symmetric`/`Hermitian`/`UpperTriangular` are used as type-level tags
# for dispatch. A future `MPO` reuses these exact same tags as its own type parameter(s) and
# registers its own `is_canonical(::LeftCanonical, mpo::MPO)`-style methods elsewhere (in whichever
# file defines `MPO`) - nothing here needs to know `MPO` exists. `MPState`'s own `is_canonical`
# methods live in `canonical_decompositions.jl`, next to `MPState` itself, for the same reason.

# SECTION -  GaugeForm: the singleton gauge-shape tags

"""
    GaugeForm

Abstract root of the gauge-form singleton tags - what canonical shape a matrix-product object
(an [`MPState`](@ref) today, an `MPO` later) currently holds. Used as a type parameter (the same
Julia idiom `LinearAlgebra.Symmetric`/`Hermitian`/`UpperTriangular` use for tagging structural
guarantees at the type level) rather than a runtime-inspected field, so dispatch on the gauge
shape - e.g. `is_canonical(::LeftCanonical, x)` - is resolved at compile time.

Concrete subtypes: [`LeftCanonical`](@ref), [`RightCanonical`](@ref), [`MixedCanonical`](@ref),
[`VidalGauge`](@ref), [`UnknownGauge`](@ref).
"""
abstract type GaugeForm end

"""
Gauge tag: left-isometric throughout except possibly the last site. See [`GaugeForm`](@ref).
"""
struct LeftCanonical <: GaugeForm end

"""
Gauge tag: right-isometric throughout except possibly the first site. See [`GaugeForm`](@ref).
"""
struct RightCanonical <: GaugeForm end

"""
Gauge tag: left-isometric up to some site, right-isometric after it, with an orthogonality centre in between. See [`GaugeForm`](@ref).
"""
struct MixedCanonical <: GaugeForm end

"""
Gauge tag: Vidal's Γ-λ representation. See [`GaugeForm`](@ref).
"""
struct VidalGauge <: GaugeForm end

"""
Gauge tag: no isometry condition is guaranteed - gauge freedom has not been constrained. See [`GaugeForm`](@ref) and [`GaugeFreedom`](@ref).
"""
struct UnknownGauge <: GaugeForm end

# SECTION -  GaugeFreedom: the coarser "has the gauge been fixed at all" trait

"""
    GaugeFreedom

Whether a [`GaugeForm`](@ref) tag represents a gauge that has actually been constrained to a
choice ([`Fixed`](@ref)) or not ([`Free`](@ref)) - the same "Holy traits" style
`SimStudy.RecordingTrait`/`Active`/`Inactive` already uses for an analogous coarse yes/no split
layered on top of finer-grained concrete types. Every [`GaugeForm`](@ref) is `Fixed` except
[`UnknownGauge`](@ref), which is `Free`.

Callable on either the type or an instance: `GaugeFreedom(LeftCanonical)` and
`GaugeFreedom(LeftCanonical())` both work.
"""
abstract type GaugeFreedom end

"""
The gauge has been constrained to a specific choice. See [`GaugeFreedom`](@ref).
"""
struct Fixed <: GaugeFreedom end

"""
Gauge freedom has not been constrained - no isometry condition is guaranteed. See [`GaugeFreedom`](@ref).
"""
struct Free <: GaugeFreedom end

GaugeFreedom(::Type{<:GaugeForm}) = Fixed()
GaugeFreedom(::Type{UnknownGauge}) = Free()
GaugeFreedom(::G) where {G<:GaugeForm} = GaugeFreedom(G)

# SECTION -  BoundaryCondition: Finite / Infinite

"""
    BoundaryCondition

Abstract root of the finite/infinite boundary-condition singleton tags - a second, independent
type parameter alongside [`GaugeForm`](@ref) for matrix-product objects, so methods (canonicalization,
`is_canonical`, eventually transfer-matrix/iMPS-specific routines) can be registered per boundary
condition the same way they're registered per gauge shape.

Concrete subtypes: [`Finite`](@ref), [`Infinite`](@ref). Only `Finite` has real utility functions
today (`to_mps`/`canonicalize`/`to_vidal` all produce/require it); `Infinite` exists purely as a
plug-in point for future infinite-MPS/MPO work, with no behavior implemented yet.
"""
abstract type BoundaryCondition end

"""
A finite chain of `L` sites with two open boundaries. See [`BoundaryCondition`](@ref).
"""
struct Finite <: BoundaryCondition end

"""
An infinite, translation-invariant chain (no utility functions implemented yet - a future extension point). See [`BoundaryCondition`](@ref).
"""
struct Infinite <: BoundaryCondition end
