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

# SECTION -  MPState: the finished chain

"""
    MPState{G<:GaugeForm,B<:BoundaryCondition}

A matrix product state built from [`QProcess`](@ref) site tensors, each with two output legs
`(vL, σ)` (bond, physical) and one input leg `vR` - the "site tensor as a morphism
`vR → (vL, σ)`" convention established during Spec A's design: only when the whole chain is
sealed at both boundaries (trivial dimension-1 bonds) does the composite reduce to a genuine
[`State`](@ref) with no remaining external inputs.

The two type parameters are singleton trait tags, not runtime-inspected fields - the same idiom
`LinearAlgebra.Symmetric`/`Hermitian`/`UpperTriangular` use: `G` (a [`GaugeForm`](@ref)) records
which canonical shape currently holds, `B` (a [`BoundaryCondition`](@ref)) records finite vs.
infinite. Both are reused as-is by a future `MPO` (see `gauge.jl`'s module note), and dispatch on
either - `is_canonical(::LeftCanonical, mps)`, a future `canonicalize(::Infinite, mps, config)` -
resolves at compile time rather than via a runtime type check.

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

The inner constructor validates via [`_validate_boundary`](@ref), dispatched on `B` - e.g. a
`Finite` chain requires `orthogonality_center` (if not `nothing`) to fall within `1:length(sites)`;
an `Infinite` chain has no invariants checked yet (a plug-in point for future iMPS-specific
checks). Two outer convenience constructors take a `GaugeForm` instance (and optionally a
`BoundaryCondition` instance, defaulting to [`Finite`](@ref) - "for now we only have utility
functions for finite MPS") instead of requiring the more verbose `MPState{LeftCanonical,Finite}(...)`
spelling directly.
"""
struct MPState{G<:GaugeForm,B<:BoundaryCondition}
    sites::Vector{QProcess}
    llim::Int
    rlim::Int
    orthogonality_center::Union{Int,Nothing}
    ε::Float64

    function MPState{G,B}(
        sites::AbstractVector{<:QProcess},
        llim::Int,
        rlim::Int,
        orthogonality_center::Union{Int,Nothing},
        ε::Float64,
    ) where {G<:GaugeForm,B<:BoundaryCondition}
        _validate_boundary(B(), sites, llim, rlim, orthogonality_center)
        return new{G,B}(sites, llim, rlim, orthogonality_center, ε)
    end
end

"""
    _validate_boundary(b::BoundaryCondition, sites, llim, rlim, orthogonality_center)

Boundary-condition-dispatched validation hook for [`MPState`](@ref)'s inner constructor - only
[`Finite`](@ref) has real invariants checked today ([`Infinite`](@ref) is a pure plug-in point for
future iMPS-specific checks, not yet implemented).
"""
function _validate_boundary(::Finite, sites, llim, rlim, orthogonality_center)
    isnothing(orthogonality_center) ||
        (1 <= orthogonality_center <= length(sites)) ||
        throw(
            ArgumentError(
                "orthogonality_center must be within 1:length(sites) for a Finite MPState."
            ),
        )
    return nothing
end
_validate_boundary(::Infinite, sites, llim, rlim, orthogonality_center) = nothing

function MPState(
    sites::AbstractVector{<:QProcess},
    ::G,
    llim::Int,
    rlim::Int,
    center::Union{Int,Nothing},
    ε::Float64,
) where {G<:GaugeForm}
    return MPState{G,Finite}(sites, llim, rlim, center, ε)
end
function MPState(
    sites::AbstractVector{<:QProcess},
    ::G,
    ::B,
    llim::Int,
    rlim::Int,
    center::Union{Int,Nothing},
    ε::Float64,
) where {G<:GaugeForm,B<:BoundaryCondition}
    return MPState{G,B}(sites, llim, rlim, center, ε)
end

function _finalize_site(A::TensorKit.AbstractTensorMap)
    return QProcess(A; output_roles=(VirtualLeg(), PhysicalLeg()), input_roles=VirtualLeg())
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
            _finalize_site.(As),
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
            _finalize_site.(As),
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
    canonicalize(chain::MPState, config::CanonicalizeConfig) -> MPState

Re-gauge `chain` according to `config` by sweeping [`advance_bond!`](@ref) over its existing site
tensors. `chain` is not mutated; a new [`MPState`](@ref) is returned with `ε` accumulated onto
`chain.ε` (re-gauging cannot undo an approximation made earlier).
"""
function canonicalize(chain::MPState, config::LeftCanonicalize)
    L = length(chain.sites)
    accumulator = QuadratureTruncationErrorAccumulator()
    AL = Vector{TensorKit.AbstractTensorMap}(undef, L)
    T = tensor(chain.sites[1])
    for i in 1:(L - 1)
        site_tensor, remainder = advance_bond!(
            LeftRight(), T, i; bond_cutoff=config.bond_cutoff, accumulator
        )
        AL[i] = site_tensor
        Am = TensorKit.permute(tensor(chain.sites[i + 1]), ((1,), (2, 3)))
        T = TensorKit.permute(remainder * Am, ((1, 2), (3,)))
    end
    AL[L] = T
    ε = hypot(chain.ε, something(finalize!(accumulator), 0.0))
    return MPState(_finalize_site.(AL), LeftCanonical(), L, L + 1, L, ε)
end

function canonicalize(chain::MPState, config::RightCanonicalize)
    L = length(chain.sites)
    accumulator = QuadratureTruncationErrorAccumulator()
    AR = Vector{TensorKit.AbstractTensorMap}(undef, L)
    T = tensor(chain.sites[L])
    for i in L:-1:2
        site_tensor, remainder = advance_bond!(
            RightLeft(), T, i; bond_cutoff=config.bond_cutoff, accumulator
        )
        AR[i] = site_tensor
        T = tensor(chain.sites[i - 1]) * remainder
    end
    AR[1] = T
    ε = hypot(chain.ε, something(finalize!(accumulator), 0.0))
    return MPState(_finalize_site.(AR), RightCanonical(), 0, 1, 1, ε)
end

function canonicalize(chain::MPState, config::SiteCanonicalize)
    L = length(chain.sites)
    k = config.k
    accumulator = QuadratureTruncationErrorAccumulator()

    AL = Vector{TensorKit.AbstractTensorMap}(undef, max(k - 1, 0))
    T = tensor(chain.sites[1])
    for i in 1:(k - 2)
        site_tensor, remainder = advance_bond!(
            LeftRight(), T, i; bond_cutoff=config.bond_cutoff, accumulator
        )
        AL[i] = site_tensor
        Am = TensorKit.permute(tensor(chain.sites[i + 1]), ((1,), (2, 3)))
        T = TensorKit.permute(remainder * Am, ((1, 2), (3,)))
    end
    local L_rem
    if k > 1
        site_tensor, L_rem = advance_bond!(
            LeftRight(), T, k - 1; bond_cutoff=config.bond_cutoff, accumulator
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
            RightLeft(), T, i; bond_cutoff=config.bond_cutoff, accumulator
        )
        AR[i - k] = site_tensor
        T = tensor(chain.sites[i - 1]) * remainder
    end
    local R_rem
    if k < L
        site_tensor, R_rem = advance_bond!(
            RightLeft(), T, k + 1; bond_cutoff=config.bond_cutoff, accumulator
        )
        AR[1] = site_tensor
    else
        Vr = TensorKit.space(tensor(chain.sites[L]), 3)'
        R_rem = TensorKit.isomorphism(TensorKit.storagetype(tensor(chain.sites[L])), Vr, Vr)
    end

    Am = TensorKit.permute(tensor(chain.sites[k]), ((1,), (2, 3)))
    AC = TensorKit.permute(L_rem * Am, ((1, 2), (3,))) * R_rem

    sites = Vector{QProcess}(undef, L)
    for i in 1:(k - 1)
        sites[i] = _finalize_site(AL[i])
    end
    sites[k] = _finalize_site(AC)
    for i in (k + 1):L
        sites[i] = _finalize_site(AR[i - k])
    end

    ε = hypot(chain.ε, something(finalize!(accumulator), 0.0))
    return MPState(sites, MixedCanonical(), k - 1, k + 1, k, ε)
end

# SECTION -  is_canonical: checked via Spec A's is_isometry, not a bespoke norm calculation

"""
    _is_right_isometric(site::QProcess; isapprox_kwargs...) -> Bool

Right-isometry check for a site tensor stored in our fixed `(vL, σ) | vR` leg grouping
(codomain | domain). [`is_isometry`](@ref) alone cannot express this: it always checks
`dagger(p) ∘ p ≈ 1` against *whatever* `p`'s own domain/codomain split already is, and our
site tensors are grouped uniformly as `(vL, σ) | vR` for every site regardless of canonical
direction (so `is_isometry(site)` is exactly the left-isometry check, matching the stored
grouping) - the right-isometry condition `B B† ≈ 1` needs the *other* grouping, `vL | (σ, vR)`,
which requires an explicit leg regroup (`TensorKit.permute`) before checking, not merely a
dagger. This bypasses the `Processes.is_isometry` wrapper's fixed `side=:left` and calls
`TensorKit.isisometric` directly with `side=:right` on the regrouped tensor.
"""
function _is_right_isometric(site::QProcess; isapprox_kwargs...)
    t = TensorKit.permute(tensor(site), ((1,), (2, 3)))
    return TensorKit.isisometric(t; side=:right, isapprox_kwargs...)
end

"""
    is_canonical(::LeftCanonical, mps::MPState; isapprox_kwargs...) -> Bool
    is_canonical(::RightCanonical, mps::MPState; isapprox_kwargs...) -> Bool
    is_canonical(::MixedCanonical, mps::MPState; isapprox_kwargs...) -> Bool
    is_canonical(::VidalGauge, mps::MPState) -> Bool
    is_canonical(::UnknownGauge, mps::MPState) -> Bool
    is_canonical(mps::MPState) -> Bool

`true` iff every site tensor in `mps` satisfies its expected isometry condition - sites
`1,...,llim-1` checked via [`is_isometry`](@ref) (left isometry, `dagger(A) ∘ A ≈ 1`, which
matches our site tensors' stored `(vL, σ) | vR` leg grouping directly), sites `rlim,...,L` via
[`_is_right_isometric`](@ref) (right isometry, `B B† ≈ 1`, which needs the `vL | (σ, vR)`
grouping instead - an explicit regroup, not just a dagger). The centre site(s) are not checked.

Registered as three **separate** methods per [`GaugeForm`](@ref) tag (`LeftCanonical`/
`RightCanonical`/`MixedCanonical`), even though the left/right bodies are short, rather than one
`Union`-dispatched method - so each can independently grow different behavior later (e.g. once an
`MPO` gets its own gauge dispatch, its isometry check may differ from an `MPS`'s, and separate
methods are what makes that a non-breaking addition). `is_canonical(::VidalGauge, ::MPState) = true` and `is_canonical(::UnknownGauge, ::MPState) = false` mirror the legacy `VidalForm`/
`ArbitraryForm` branches exactly. The no-tag-argument method dispatches off `mps`'s own stored
`GaugeForm` type parameter, so `is_canonical(mps)` is the usual call site.

This is the payoff of Spec A's categorical layer: canonical-form checking is a real equation, not
a bespoke norm calculation.
"""
function is_canonical(::LeftCanonical, mps::MPState; isapprox_kwargs...)
    return all(i -> is_isometry(mps.sites[i]; isapprox_kwargs...), 1:(mps.llim - 1))
end
function is_canonical(::RightCanonical, mps::MPState; isapprox_kwargs...)
    return all(
        i -> _is_right_isometric(mps.sites[i]; isapprox_kwargs...),
        mps.rlim:length(mps.sites),
    )
end
function is_canonical(::MixedCanonical, mps::MPState; isapprox_kwargs...)
    return all(i -> is_isometry(mps.sites[i]; isapprox_kwargs...), 1:(mps.llim - 1)) && all(
        i -> _is_right_isometric(mps.sites[i]; isapprox_kwargs...),
        mps.rlim:length(mps.sites),
    )
end
is_canonical(::VidalGauge, ::MPState) = true
is_canonical(::UnknownGauge, ::MPState) = false
function is_canonical(mps::MPState{G}; kwargs...) where {G<:GaugeForm}
    return is_canonical(G(), mps; kwargs...)
end

"""
    is_gauge_fixed(mps::MPState{G}) where {G<:GaugeForm} -> Bool

The coarse "has this been gauge-fixed at all" check ([`GaugeFreedom`](@ref)): `true` for every
gauge-form tag except [`UnknownGauge`](@ref).
"""
is_gauge_fixed(::MPState{G}) where {G<:GaugeForm} = GaugeFreedom(G) isa Fixed

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
        Am = TensorKit.permute(t, ((1,), (2, 3)))
        AL_tilde = TensorKit.permute(Xleft' * Am, ((1, 2), (3,))) * Xright
        Γt = i < L ? AL_tilde * inv(λs[i]) : AL_tilde
        Γs[i] = _finalize_site(Γt)
    end
    return Γs, λs
end
