#=META
source:
  author: Bavithra
  coauthor: Claude Opus 5
  reviewer:
docstrings:
  author: Bavithra
  coauthor: Claude Opus 5
  reviewer:
refs: coecke_kissinger_2016a
credits: mpskit-validation (internal validation package) - canonicalize_utils.jl
=#

using TensorKit

# SECTION -  TensorTrain: the unified MPState/MPOperator chain

"""
    TensorTrain{G<:GaugeForm,S<:BoundarySupport,P}

A 1D tensor-train chain built from [`QProcess`](@ref) site tensors, unifying what were previously
two separate types: `MPState` (`P=1`, one physical leg per site) and `MPOperator` (`P=2`, a
bra- and a ket-physical leg per site). `MPState`/`MPOperator` remain available as type aliases
(see below) - nothing about their existing call sites changes.

Bulk `MPState` sites (`P=1`) have two output legs `(vL, σ)` (bond, physical) and one input leg
`vR` - the "site tensor as a morphism `vR → (vL, σ)`" convention established during Spec A's
design. Bulk `MPOperator` sites (`P=2`) have output legs `(vL, σ_ket)` and input legs
`(vR, σ_bra)` - `vR ⊗ σ_bra → vL ⊗ σ_ket`, the "output = ket, input = bra" convention the
categorical layer already uses for `adjoint`. Only when the whole chain is sealed at both
boundaries (trivial dimension-1 bonds) does the composite reduce to a genuine [`State`](@ref)
(or, for `MPOperator`, a genuine operator process with no remaining external bond legs).

`P` is a plain `Int` type parameter, not a `LegRole`/trait type - it only ever drives dispatch
through [`_site_roles`](@ref)`(Val(P))`, the sole point where `MPState`- and `MPOperator`-specific
leg wiring differs. Every other method here (`is_canonical`, `canonicalize`,
`_validate_boundary`, `_is_right_isometric`) is written once against `TensorTrain`, unconditional
on `P`, because [`is_isometry`](@ref)/[`adjoint`](@ref) are already generic over leg count. A
site's *boundary* leg count (dropping a bond via the trivial monoidal-unit space) is not part of
`P` either - `QProcess`'s own type never encodes arity, so `Vector{QProcess}` already tolerates
that per-site heterogeneity with no extra machinery.

The other two type parameters are singleton trait tags, not runtime-inspected fields - the same
idiom `LinearAlgebra.Symmetric`/`Hermitian`/`UpperTriangular` use: `G` (a [`GaugeForm`](@ref))
records which canonical shape currently holds, `S` (a [`BoundarySupport`](@ref)) records finite
vs. infinite extent. Dispatch on either - `is_canonical(::LeftCanonical, chain)`, a future
`canonicalize(::InfiniteSupport, chain, config)` - resolves at compile time rather than via a
runtime type check.

# Fields

  - `sites                :: Vector{QProcess}`    - `L` site tensors
  - `llim                 :: Int`                 - sites `1,...,llim-1` are left-isometric
  - `rlim                 :: Int`                 - sites `rlim,...,L` are right-isometric
  - `orthogonality_center :: Union{Int,Nothing}`  - site holding the orthogonality centre
  - `ε                    :: Float64`             - accumulated truncation error (quadrature)

`llim`/`rlim` keep exactly the meaning `CanonicalForm(llim,rlim)` had before this became a type
parameter: the orthogonality-centre region occupies sites `llim,...,rlim-1`; `llim=0`/`rlim=L+1`
are out-of-bounds sentinels making the corresponding range empty without special-casing (fully
left-canonical is `(L, L+1)`, fully right-canonical is `(0, 1)`, mixed-canonical at site `k` is
`(k-1, k+1)`). They are unused (`0, 0`) for [`VidalGauge`](@ref)/[`UnknownGauge`](@ref).

# Constructors

The inner constructor validates via [`_validate_boundary`](@ref), dispatched on `S` - e.g. a
`FiniteSupport` chain requires `orthogonality_center` (if not `nothing`) to fall within
`1:length(sites)`; an `InfiniteSupport` chain has no invariants checked yet (a plug-in point for
future iMPS/iMPO-specific checks). This is exactly `MPState`'s original inner-constructor pattern,
carried over unchanged: `llim`/`rlim`/`orthogonality_center` mean different things - and need
different validity checks - depending on `S`, and dispatching the constructor on `S` keeps that
difference entirely inside `_validate_boundary`'s two methods rather than duplicated per `P`.
"""
struct TensorTrain{G<:GaugeForm,S<:BoundarySupport,P}
    sites::Vector{QProcess}
    llim::Int
    rlim::Int
    orthogonality_center::Union{Int,Nothing}
    ε::Float64

    function TensorTrain{G,S,P}(
        sites::AbstractVector{<:QProcess},
        llim::Int,
        rlim::Int,
        orthogonality_center::Union{Int,Nothing},
        ε::Float64,
    ) where {G<:GaugeForm,S<:BoundarySupport,P}
        _validate_boundary(S(), sites, llim, rlim, orthogonality_center)
        return new{G,S,P}(sites, llim, rlim, orthogonality_center, ε)
    end
end

"""
    MPState{G<:GaugeForm,S<:BoundarySupport}

Alias `TensorTrain{G,S,1}` - a matrix product state (one physical leg per bulk site). See
[`TensorTrain`](@ref).
"""
const MPState{G,S} = TensorTrain{G,S,1}

"""
    MPOperator{G<:GaugeForm,S<:BoundarySupport}

Alias `TensorTrain{G,S,2}` - a matrix product operator (a bra- and a ket-physical leg per bulk
site). See [`TensorTrain`](@ref).
"""
const MPOperator{G,S} = TensorTrain{G,S,2}

"""
    _validate_boundary(s::BoundarySupport, sites, llim, rlim, orthogonality_center)

Boundary-support-dispatched validation hook for [`TensorTrain`](@ref)'s inner constructor - only
[`FiniteSupport`](@ref) has real invariants checked today ([`InfiniteSupport`](@ref) is a pure
plug-in point for future iMPS/iMPO-specific checks, not yet implemented).
"""
function _validate_boundary(::FiniteSupport, sites, llim, rlim, orthogonality_center)
    isnothing(orthogonality_center) ||
        (1 <= orthogonality_center <= length(sites)) ||
        throw(
            ArgumentError(
                "orthogonality_center must be within 1:length(sites) for a FiniteSupport TensorTrain.",
            ),
        )
    return nothing
end
_validate_boundary(::InfiniteSupport, sites, llim, rlim, orthogonality_center) = nothing

function MPState(
    sites::AbstractVector{<:QProcess},
    ::G,
    llim::Int,
    rlim::Int,
    center::Union{Int,Nothing},
    ε::Float64,
) where {G<:GaugeForm}
    return MPState{G,FiniteSupport}(sites, llim, rlim, center, ε)
end
function MPState(
    sites::AbstractVector{<:QProcess},
    ::G,
    ::S,
    llim::Int,
    rlim::Int,
    center::Union{Int,Nothing},
    ε::Float64,
) where {G<:GaugeForm,S<:BoundarySupport}
    return MPState{G,S}(sites, llim, rlim, center, ε)
end

function MPOperator(
    sites::AbstractVector{<:QProcess},
    ::G,
    llim::Int,
    rlim::Int,
    center::Union{Int,Nothing},
    ε::Float64,
) where {G<:GaugeForm}
    return MPOperator{G,FiniteSupport}(sites, llim, rlim, center, ε)
end
function MPOperator(
    sites::AbstractVector{<:QProcess},
    ::G,
    ::S,
    llim::Int,
    rlim::Int,
    center::Union{Int,Nothing},
    ε::Float64,
) where {G<:GaugeForm,S<:BoundarySupport}
    return MPOperator{G,S}(sites, llim, rlim, center, ε)
end

"""
    _site_roles(::Val{1}) -> (output_roles, input_roles)
    _site_roles(::Val{2}) -> (output_roles, input_roles)

The sole per-`P` specialization point for [`TensorTrain`](@ref): the `output_roles`/`input_roles`
passed to `QProcess(A; output_roles, input_roles)` when re-wrapping a raw factorized tensor into
a site. `Val(1)` (`MPState`) puts the physical leg on the output side alone
(`vR → (vL,σ)`); `Val(2)` (`MPOperator`) puts one physical leg on each side
(`vR⊗σ_bra → vL⊗σ_ket`).
"""
_site_roles(::Val{1}) = ((VirtualLeg(), PhysicalLeg()), VirtualLeg())
_site_roles(::Val{2}) = ((VirtualLeg(), PhysicalLeg()), (VirtualLeg(), PhysicalLeg()))

"""
    _finalize_site(::Val{P}, A) where {P}
    _finalize_site(chain::TensorTrain{G,S,P}, A) where {G,S,P}

Wrap a raw factorized tensor `A` into a site `QProcess`, using the leg roles [`_site_roles`](@ref)
reports for `P`. The `chain`-taking form reads `P` off an existing chain's type, for use inside
[`canonicalize`](@ref); the `Val`-taking form is used directly by construction entry points
(`to_mps`, `to_mpo`) that don't yet have a chain to read `P` from.
"""
function _finalize_site(::Val{P}, A::TensorKit.AbstractTensorMap) where {P}
    roles = _site_roles(Val(P))
    return QProcess(A; output_roles=roles[1], input_roles=roles[2])
end
function _finalize_site(::TensorTrain{G,S,P}, A::TensorKit.AbstractTensorMap) where {G,S,P}
    return _finalize_site(Val(P), A)
end

"""
    _isolate_bond_in(::Val{P}, t) -> AbstractTensorMap
    _restore_site_shape(::Val{P}, t) -> AbstractTensorMap

Per-`P` regroup pair used by [`canonicalize`](@ref)'s [`RightCanonicalize`](@ref)/
[`SiteCanonicalize`](@ref) loops to fold a bond-cut `remainder` into the *previous* (already
separated) site tensor. `_isolate_bond_in` regroups a legit bulk site tensor so its bond-in leg
`vR` is isolated alone as domain (pushing any extra domain leg, `σ_bra` for `P=2`, over to
codomain) - the same non-contiguous grouping `step(::LeftRight, ::Val{2}, ...)` uses internally,
needed here because `remainder`'s codomain is always a single leg regardless of `P`, so composing
it against a site whose domain has more than one leg (`P>1`) would otherwise fail to typecheck.
`_restore_site_shape` reorders back to the standard bulk-site leg order afterward (bond leg first
in the domain, then any extras). Both are the identity for `P=1` (no extra domain leg exists to
move), so [`MPState`](@ref)'s behavior is unchanged.
"""
_isolate_bond_in(::Val{1}, t) = t
_isolate_bond_in(::Val{2}, t) = TensorKit.permute(t, ((1, 2, 4), (3,)))
_restore_site_shape(::Val{1}, t) = t
_restore_site_shape(::Val{2}, t) = TensorKit.permute(t, ((1, 2), (4, 3)))

"""
    _fold_right(::Val{P}, t, remainder) -> AbstractTensorMap

Fold a `RightLeft` bond-cut `remainder` into the neighbouring (already separated) site tensor
`t`, producing a fresh legit bulk site tensor for the next cut. See [`_isolate_bond_in`](@ref)/
[`_restore_site_shape`](@ref) for why `t` needs regrouping first when `P>1`.
"""
function _fold_right(::Val{P}, t, remainder) where {P}
    return _restore_site_shape(Val(P), _isolate_bond_in(Val(P), t) * remainder)
end

# SECTION -  to_mps: dense State -> MPState via iterated factorization

"""
    to_mps(ψ::State; bond_cutoff=nothing, form::Symbol=:left,
           access=HasEntanglementSpectrum(), collector=SimStudy.NoOpCollector(),
           accumulator=SimStudy.NoOpErrorAccumulator()) -> MPState

Decompose a full quantum state `ψ` (an `L`-physical-leg [`State`](@ref)) into an [`MPState`](@ref)
via [`orthonormalize`](@ref).

# Keywords

$(Glossaries.Keyword{@__MODULE__}()([:bond_cutoff]))

  - `form` - `:left` for a left-canonical sweep ([`LeftCanonical`](@ref), `(llim,rlim)=(L,L+1)`,
    orthogonality centre at site `L`) or `:right` for a right-canonical sweep ([`RightCanonical`](@ref),
    `(0,1)`, centre at site `1`).
  - `access`/`collector`/`accumulator` - forwarded to [`orthonormalize`](@ref)/[`advance_bond!`](@ref);
    see [`SimStudy`](@ref) for the collector/accumulator trait pattern.
"""
function to_mps(
    ψ::State;
    bond_cutoff::Union{Int,Nothing}=nothing,
    form::Symbol=:left,
    access::AccessEntanglementSpectrumData=HasEntanglementSpectrum(),
    collector::AbstractCollector=NoOpCollector(),
    accumulator::AbstractErrorAccumulator=NoOpErrorAccumulator(),
)
    L = length(outputs(ψ))
    ψtensor = tensor(ψ)
    if form === :left
        As, _ = orthonormalize(
            ψtensor, L, LeftRight(); access, bond_cutoff, collector, accumulator
        )
        return MPState(
            _finalize_site.(Val(1), As),
            LeftCanonical(),
            L,
            L + 1,
            L,
            something(finalize!(accumulator), 0.0),
        )
    elseif form === :right
        As, _ = orthonormalize(
            ψtensor, L, RightLeft(); access, bond_cutoff, collector, accumulator
        )
        return MPState(
            _finalize_site.(Val(1), As),
            RightCanonical(),
            0,
            1,
            1,
            something(finalize!(accumulator), 0.0),
        )
    else
        throw(ArgumentError("to_mps: form must be :left or :right, got $form"))
    end
end

# SECTION -  canonicalize: re-gauge an existing MPState

"""
    CanonicalizeConfig

Supertype for canonicalization configurations passed to [`canonicalize`](@ref) - "what to do",
distinct from the [`GaugeForm`](@ref) tags in `gauge.jl` ("what state the result is in"; the two
vocabularies are related but not 1:1, see [`SiteCanonicalize`](@ref)). Concrete subtypes:
[`LeftCanonicalize`](@ref), [`RightCanonicalize`](@ref), [`SiteCanonicalize`](@ref).

!!! note "No separate bond-canonical config"

    Unlike the legacy `mps.jl`, there is no distinct "bond canonical" config storing the
    orthogonality centre as a separate diagonal singular-value tensor - only
    [`SiteCanonicalize`](@ref) (the centre held as a full-rank site tensor `AC`), which is
    strictly more informative and can always be further split by the caller if the diagonal form
    is wanted.
"""
abstract type CanonicalizeConfig end

"""
Config: sweep left-to-right, producing gauge form [`LeftCanonical`](@ref).
"""
struct LeftCanonicalize <: CanonicalizeConfig
    bond_cutoff::Union{Int,Nothing}
end
LeftCanonicalize() = LeftCanonicalize(nothing)

"""
Config: sweep right-to-left, producing gauge form [`RightCanonical`](@ref).
"""
struct RightCanonicalize <: CanonicalizeConfig
    bond_cutoff::Union{Int,Nothing}
end
RightCanonicalize() = RightCanonicalize(nothing)

"""
    SiteCanonicalize(k, bond_cutoff=nothing)

Config: mixed canonical with orthogonality centre at site `k`, producing gauge form
[`MixedCanonical`](@ref) (`(llim,rlim)=(k-1,k+1)`). "Site-canonicalize to site `k`" is the action;
[`MixedCanonical`](@ref) is the resulting shape - the two names need not (and here don't) match
1:1 with `LeftCanonicalize`/`LeftCanonical`'s naming.
"""
struct SiteCanonicalize <: CanonicalizeConfig
    k::Int
    bond_cutoff::Union{Int,Nothing}
end
SiteCanonicalize(k::Int) = SiteCanonicalize(k, nothing)

"""
    canonicalize(chain::TensorTrain, config::CanonicalizeConfig) -> TensorTrain

Re-gauge `chain` according to `config` by sweeping [`advance_bond!`](@ref) over its existing site
tensors. `chain` is not mutated; a new [`TensorTrain`](@ref) (same `S`/`P` as `chain`) is returned
with `ε` accumulated onto `chain.ε` (re-gauging cannot undo an approximation made earlier). Works
identically for `MPState` (`P=1`) and `MPOperator` (`P=2`): [`advance_bond!`](@ref)'s SVD sweep is
already leg-count-agnostic, and [`_finalize_site`](@ref) re-wraps each result according to
`chain`'s own `P`.
"""
function canonicalize(chain::TensorTrain{G,S,P}, config::LeftCanonicalize) where {G,S,P}
    L = length(chain.sites)
    accumulator = QuadratureTruncationErrorAccumulator()
    AL = Vector{TensorKit.AbstractTensorMap}(undef, L)
    T = tensor(chain.sites[1])
    for i in 1:(L - 1)
        site_tensor, remainder = advance_bond!(
            LeftRight(), Val(P), T, i; bond_cutoff=config.bond_cutoff, accumulator
        )
        AL[i] = site_tensor
        Am = _regroup_first_out(tensor(chain.sites[i + 1]))
        T = _regroup_bulk_site(remainder * Am)
    end
    AL[L] = T
    ε = hypot(chain.ε, something(finalize!(accumulator), 0.0))
    return TensorTrain{LeftCanonical,S,P}(
        [_finalize_site(chain, A) for A in AL], L, L + 1, L, ε
    )
end

function canonicalize(chain::TensorTrain{G,S,P}, config::RightCanonicalize) where {G,S,P}
    L = length(chain.sites)
    accumulator = QuadratureTruncationErrorAccumulator()
    AR = Vector{TensorKit.AbstractTensorMap}(undef, L)
    T = tensor(chain.sites[L])
    for i in L:-1:2
        site_tensor, remainder = advance_bond!(
            RightLeft(), Val(P), T, i; bond_cutoff=config.bond_cutoff, accumulator
        )
        AR[i] = site_tensor
        T = _fold_right(Val(P), tensor(chain.sites[i - 1]), remainder)
    end
    AR[1] = T
    ε = hypot(chain.ε, something(finalize!(accumulator), 0.0))
    return TensorTrain{RightCanonical,S,P}(
        [_finalize_site(chain, A) for A in AR], 0, 1, 1, ε
    )
end

function canonicalize(chain::TensorTrain{G,S,P}, config::SiteCanonicalize) where {G,S,P}
    L = length(chain.sites)
    k = config.k
    accumulator = QuadratureTruncationErrorAccumulator()

    AL = Vector{TensorKit.AbstractTensorMap}(undef, max(k - 1, 0))
    T = tensor(chain.sites[1])
    for i in 1:(k - 2)
        site_tensor, remainder = advance_bond!(
            LeftRight(), Val(P), T, i; bond_cutoff=config.bond_cutoff, accumulator
        )
        AL[i] = site_tensor
        Am = _regroup_first_out(tensor(chain.sites[i + 1]))
        T = _regroup_bulk_site(remainder * Am)
    end
    local L_rem
    if k > 1
        site_tensor, L_rem = advance_bond!(
            LeftRight(), Val(P), T, k - 1; bond_cutoff=config.bond_cutoff, accumulator
        )
        AL[k - 1] = site_tensor
    else
        V = TensorKit.space(tensor(chain.sites[1]), 1)
        L_rem = TensorKit.isomorphism(TensorKit.storagetype(tensor(chain.sites[1])), V, V)
    end

    AR = Vector{TensorKit.AbstractTensorMap}(undef, max(L - k, 0))
    T = tensor(chain.sites[L])
    for i in L:-1:(k + 2)
        site_tensor, remainder = advance_bond!(
            RightLeft(), Val(P), T, i; bond_cutoff=config.bond_cutoff, accumulator
        )
        AR[i - k] = site_tensor
        T = _fold_right(Val(P), tensor(chain.sites[i - 1]), remainder)
    end
    local R_rem
    if k < L
        site_tensor, R_rem = advance_bond!(
            RightLeft(), Val(P), T, k + 1; bond_cutoff=config.bond_cutoff, accumulator
        )
        AR[1] = site_tensor
    else
        Vr = TensorKit.space(tensor(chain.sites[L]), 3)'
        R_rem = TensorKit.isomorphism(TensorKit.storagetype(tensor(chain.sites[L])), Vr, Vr)
    end

    Am = _regroup_first_out(tensor(chain.sites[k]))
    AC = _fold_right(Val(P), _regroup_bulk_site(L_rem * Am), R_rem)

    sites = Vector{QProcess}(undef, L)
    for i in 1:(k - 1)
        sites[i] = _finalize_site(chain, AL[i])
    end
    sites[k] = _finalize_site(chain, AC)
    for i in (k + 1):L
        sites[i] = _finalize_site(chain, AR[i - k])
    end

    ε = hypot(chain.ε, something(finalize!(accumulator), 0.0))
    return TensorTrain{MixedCanonical,S,P}(sites, k - 1, k + 1, k, ε)
end

# SECTION -  is_canonical: checked via Spec A's is_isometry, not a bespoke norm calculation

"""
    _regroup_first_out(t) -> AbstractTensorMap

Regroup `t`'s legs into `(leg 1) | (every other leg)` - arity-agnostic, so it applies unchanged
to a `P=1` bulk site (3 legs total, giving `(vL) | (σ, vR)`) and a `P=2` bulk site (4 legs total,
giving `(vL) | (σ_ket, vR, σ_bra)`).
"""
function _regroup_first_out(t)
    return TensorKit.permute(t, ((1,), Tuple(2:TensorKit.numind(t))))
end

"""
    _regroup_bulk_site(t) -> AbstractTensorMap

Regroup `t`'s legs into `(leg 1, leg 2) | (every remaining leg)` - the shape a freshly-composed
`remainder * Am` tensor needs to be re-grouped into to look like a bulk site tensor again
(`(vL, σ) | (vR)` for `P=1`, `(vL, σ_ket) | (vR, σ_bra)` for `P=2`).
"""
function _regroup_bulk_site(t)
    return TensorKit.permute(t, ((1, 2), Tuple(3:TensorKit.numind(t))))
end

"""
    _is_left_isometric(site::QProcess; isapprox_kwargs...) -> Bool

Left-isometry check for a bulk site tensor, testing isometry via the `(vL, ..., extras) | (vR)`
grouping (isolating the bond-out leg `vR` alone via [`_isolate_bond_in`](@ref)) rather than the
*stored* `(vL, ...) | (..., vR)` grouping. For `P=1` there is no extra domain leg (`_isolate_bond_in`
is the identity), so this is exactly `is_isometry(site)` on the stored grouping, matching
`MPState`'s original behavior unchanged. For `P=2`, the stored grouping (`(vL,σ_ket) | (vR,σ_bra)`)
and the isometry-guaranteeing grouping (`(vL,σ_ket,σ_bra) | (vR)`) genuinely differ - permuting
which legs count as domain vs codomain changes what "isometric" even means, so checking the wrong
one silently gives the wrong answer rather than erroring. [`canonicalize`](@ref)'s left-sweep SVD
cut guarantees isometry only in this second grouping, which is why this is checked here rather
than via a plain [`is_isometry`](@ref) call.
"""
function _is_left_isometric(site::QProcess; isapprox_kwargs...)
    t = tensor(site)
    t = _isolate_bond_in(Val(TensorKit.numin(t)), t)
    return TensorKit.isisometric(t; side=:left, isapprox_kwargs...)
end

"""
    _is_right_isometric(site::QProcess; isapprox_kwargs...) -> Bool

Right-isometry check for a site tensor stored in our fixed `(vL, ...) | (..., vR)` leg grouping
(codomain | domain). [`is_isometry`](@ref) alone cannot express this: it always checks
`dagger(p) ∘ p ≈ 1` against *whatever* `p`'s own domain/codomain split already is, and our
site tensors are grouped uniformly as `(vL, ...) | (..., vR)` for every site regardless of
canonical direction - the right-isometry condition `B B† ≈ 1` needs the *other* grouping,
`vL | (..., vR)`, via [`_regroup_first_out`](@ref), not merely a dagger. This bypasses the
`Processes.is_isometry` wrapper's fixed `side=:left` and calls `TensorKit.isisometric` directly
with `side=:right` on the regrouped tensor. Arity-agnostic (works for both `MPState`'s 3-leg and
`MPOperator`'s 4-leg sites) since [`_regroup_first_out`](@ref) is - unlike the left-isometric case,
`vL | (..., vR)` happens to be the correct isometry-guaranteeing grouping for *both* `P` values
(the leg being isolated, `vL`, is always at the boundary), so no `_is_left_isometric`-style
special case is needed here.
"""
function _is_right_isometric(site::QProcess; isapprox_kwargs...)
    t = _regroup_first_out(tensor(site))
    return TensorKit.isisometric(t; side=:right, isapprox_kwargs...)
end

"""
    is_canonical(::LeftCanonical, chain::TensorTrain; isapprox_kwargs...) -> Bool
    is_canonical(::RightCanonical, chain::TensorTrain; isapprox_kwargs...) -> Bool
    is_canonical(::MixedCanonical, chain::TensorTrain; isapprox_kwargs...) -> Bool
    is_canonical(::VidalGauge, chain::TensorTrain) -> Bool
    is_canonical(::UnknownGauge, chain::TensorTrain) -> Bool
    is_canonical(chain::TensorTrain) -> Bool

`true` iff every site tensor in `chain` satisfies its expected isometry condition - sites
`1,...,llim-1` checked via [`_is_left_isometric`](@ref) (left isometry, `dagger(A) ∘ A ≈ 1`, via
the `(vL, ..., extras) | (vR)` grouping - the *stored* grouping only for `P=1`), sites
`rlim,...,L` via [`_is_right_isometric`](@ref) (right isometry, `B B† ≈ 1`, via the
`vL | (..., vR)` grouping). The centre site(s) are not checked. Written once against
[`TensorTrain`](@ref), unconditional on `P` - `MPState` and `MPOperator` share these methods
verbatim, since [`_is_left_isometric`](@ref)/[`_is_right_isometric`](@ref) are already
arity-agnostic.

Registered as three **separate** methods per [`GaugeForm`](@ref) tag (`LeftCanonical`/
`RightCanonical`/`MixedCanonical`), even though the left/right bodies are short, rather than one
`Union`-dispatched method - so each can independently grow different behavior later without
breaking the others. `is_canonical(::VidalGauge, ::TensorTrain) = true` and
`is_canonical(::UnknownGauge, ::TensorTrain) = false` mirror the legacy `VidalForm`/`ArbitraryForm`
branches exactly. The no-tag-argument method dispatches off `chain`'s own stored `GaugeForm` type
parameter, so `is_canonical(chain)` is the usual call site.

This is the payoff of Spec A's categorical layer: canonical-form checking is a real equation, not
a bespoke norm calculation.
"""
function is_canonical(::LeftCanonical, chain::TensorTrain; isapprox_kwargs...)
    return all(
        i -> _is_left_isometric(chain.sites[i]; isapprox_kwargs...), 1:(chain.llim - 1)
    )
end
function is_canonical(::RightCanonical, chain::TensorTrain; isapprox_kwargs...)
    return all(
        i -> _is_right_isometric(chain.sites[i]; isapprox_kwargs...),
        chain.rlim:length(chain.sites),
    )
end
function is_canonical(::MixedCanonical, chain::TensorTrain; isapprox_kwargs...)
    return all(
        i -> _is_left_isometric(chain.sites[i]; isapprox_kwargs...), 1:(chain.llim - 1)
    ) && all(
        i -> _is_right_isometric(chain.sites[i]; isapprox_kwargs...),
        chain.rlim:length(chain.sites),
    )
end
is_canonical(::VidalGauge, ::TensorTrain) = true
is_canonical(::UnknownGauge, ::TensorTrain) = false
function is_canonical(chain::TensorTrain{G}; kwargs...) where {G<:GaugeForm}
    return is_canonical(G(), chain; kwargs...)
end

"""
    is_gauge_fixed(chain::TensorTrain{G}) where {G<:GaugeForm} -> Bool

The coarse "has this been gauge-fixed at all" check ([`GaugeFreedom`](@ref)): `true` for every
gauge-form tag except [`UnknownGauge`](@ref). Written once against [`TensorTrain`](@ref), shared
by `MPState` and `MPOperator`.
"""
is_gauge_fixed(::TensorTrain{G}) where {G<:GaugeForm} = GaugeFreedom(G) isa Fixed

# SECTION -  to_vidal: canonical MPState -> Vidal (Gamma, lambda) form

"""
    to_vidal(chain::MPState) -> (Γs, λs)

Convert a fully left-canonical `chain` (gauge form [`LeftCanonical`](@ref) with `chain.llim == length(chain.sites)`) into Vidal form. An additional right-to-left sweep over the
already-left-canonical tensors extracts, at each bond, the gauge-transform tensor
[`advance_bond!`](@ref) produces as its `remainder` *before* folding it into the neighbouring
site - this is exactly the bond's `C` gauge tensor. Diagonalizing each via
`TensorKit.svd_compact` exposes the genuine (gauge-invariant) Schmidt values `λ`; each `Γ` is then
`AL` rotated into that Schmidt eigenbasis on both bonds, divided by the right bond's `λ`.

!!! note "Left-canonical input only"

    Only conversion from a fully left-canonical `chain` is implemented; convert a right-canonical
    chain via `canonicalize(chain, LeftCanonicalize())` first if needed.
"""
function to_vidal(chain::MPState)
    (chain isa MPState{LeftCanonical} && chain.llim == length(chain.sites)) || throw(
        ArgumentError(
            "to_vidal requires a fully left-canonical MPState; call canonicalize(chain, LeftCanonicalize()) first.",
        ),
    )

    L = length(chain.sites)
    Xs = Vector{TensorKit.AbstractTensorMap}(undef, L - 1)
    λs = Vector{TensorKit.AbstractTensorMap}(undef, L - 1)
    T = tensor(chain.sites[L])
    for i in L:-1:2
        _, remainder = advance_bond!(RightLeft(), T, i)
        X, S, _ = TensorKit.svd_compact(remainder)
        Xs[i - 1] = X
        λs[i - 1] = S
        T = tensor(chain.sites[i - 1]) * remainder
    end

    Γs = Vector{QProcess}(undef, L)
    for i in 1:L
        t = tensor(chain.sites[i])
        Xleft = if i == 1
            TensorKit.isomorphism(
                TensorKit.storagetype(t), TensorKit.space(t, 1), TensorKit.space(t, 1)
            )
        else
            Xs[i - 1]
        end
        Xright = if i == L
            TensorKit.isomorphism(
                TensorKit.storagetype(t), TensorKit.space(t, 3)', TensorKit.space(t, 3)'
            )
        else
            Xs[i]
        end
        Am = _regroup_first_out(t)
        AL_tilde = _regroup_bulk_site(Xleft' * Am) * Xright
        Γt = i < L ? AL_tilde * inv(λs[i]) : AL_tilde
        Γs[i] = _finalize_site(chain, Γt)
    end
    return Γs, λs
end
