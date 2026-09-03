#=META
source:
  author: Bavithra
  reviewer:
docstrings:
  author: Claude Sonnet 5
  coauthor: 
  reviewer:
refs:
credits: mpskit-validation (internal validation package) - decompositions.jl
=#

using TensorKit
using ..SimStudy:
    SimStudy,
    RecordingTrait,
    Active,
    step!,
    record!,
    AbstractCollector,
    NoOpCollector,
    AbstractErrorAccumulator,
    NoOpErrorAccumulator

# Only the SVD/QR/LQ factorization and canonicalization-sweep machinery is ported here;
# mpskit-validation/decompositions.jl's Trotterization/ProductFormula/OperatorSplitting types are
# TEBD-specific and out of scope for canonicalization - they belong in a future TEBD subroutine
# file, not here.

# SECTION -  Factorizer tags

abstract type MatrixFactorization end
abstract type SVDBased <: MatrixFactorization end
abstract type QRBased <: MatrixFactorization end
abstract type LQBased <: MatrixFactorization end

struct ExactDecomposition{M<:MatrixFactorization} end

const SVDFACTORIZER = ExactDecomposition{SVDBased}
const QRFACTORIZER = ExactDecomposition{QRBased}
const LQFACTORIZER = ExactDecomposition{LQBased}

# SECTION -  factorize_tensor / factorize_tensor! - trait-dispatched matrix factorization

"""
    factorize_tensor(A, ::SVDFACTORIZER; bond_cutoff=nothing, trunc_tol=nothing, kwargs...) -> U, S, Vd, ε
    factorize_tensor(A, ::QRFACTORIZER; kwargs...) -> Q, R
    factorize_tensor(A, ::LQFACTORIZER; kwargs...) -> L, Q

Factorize `A` according to the algorithm tagged by the second (singleton) argument, built
directly on `MatrixAlgebraKit`'s primitive decomposition functions.

For [`SVDFACTORIZER`](@ref), this calls `MatrixAlgebraKit.svd_trunc`.

# Keywords

$(Glossaries.Keyword{@__MODULE__}()([:bond_cutoff]))

  - `trunc_tol` - an absolute singular-value cutoff (`TensorKit.trunctol(; atol=trunc_tol)`),
    composed with `bond_cutoff`'s rank cutoff (`TensorKit.truncrank`) via `MatrixAlgebraKit`'s
    `TruncationStrategy` combinator `&` when both are given - a bond is truncated at whichever cap
    (rank or tolerance) is tighter. `nothing` (the default) omits the tolerance cutoff entirely.
    This is the standard Vidal/iTEBD-form treatment of a negligible Schmidt value (Schollwöck 2011):
    a singular value below `trunc_tol` is dropped from the kept bond dimension, not merely
    protected against later division by it - a `bond_cutoff` alone doesn't guarantee this, since a
    padded/near-zero singular value can persist within an otherwise-untruncated rank.

For [`QRFACTORIZER`](@ref), this calls `MatrixAlgebraKit.qr_compact` (`A = Q*R`, `Q` isometric on
columns - the left-canonicalization shape). For [`LQFACTORIZER`](@ref), this calls
`MatrixAlgebraKit.lq_compact` (`A = L*Q`, `Q` isometric on rows - the right-canonicalization
shape). Neither has truncation support.

`A` is a plain `TensorKit.AbstractTensorMap` - `svd_trunc`, `qr_compact`, and `lq_compact` are
all generic `MatrixAlgebraKit` functions with methods for `AbstractTensorMap`.

Named `factorize_tensor`, not `factorize`, to avoid colliding with `LinearAlgebra.factorize`.
"""
function factorize_tensor(
    A,
    ::SVDFACTORIZER;
    bond_cutoff::Union{Int,Nothing}=nothing,
    trunc_tol::Union{Real,Nothing}=nothing,
    kwargs...,
)
    trunc = _svd_truncation_strategy(bond_cutoff, trunc_tol)
    return TensorKit.svd_trunc(A; trunc, kwargs...)
end
factorize_tensor(A, ::QRFACTORIZER; kwargs...) = TensorKit.qr_compact(A; kwargs...)
factorize_tensor(A, ::LQFACTORIZER; kwargs...) = TensorKit.lq_compact(A; kwargs...)

# Composes a rank cutoff (`bond_cutoff`) and/or an absolute tolerance cutoff (`trunc_tol`) into
# the single `TruncationStrategy` `svd_trunc`/`svd_trunc!` expect, via `MatrixAlgebraKit`'s `&`
# combinator (a bond is truncated at whichever cap is tighter) - shared by `factorize_tensor` and
# `factorize_tensor!`'s `SVDFACTORIZER` methods.
function _svd_truncation_strategy(
    bond_cutoff::Union{Int,Nothing}, trunc_tol::Union{Real,Nothing}
)
    rank = isnothing(bond_cutoff) ? nothing : TensorKit.truncrank(bond_cutoff)
    tol = isnothing(trunc_tol) ? nothing : TensorKit.trunctol(; atol=trunc_tol)
    if rank === nothing
        return tol
    elseif tol === nothing
        return rank
    else
        return rank & tol
    end
end

"""
    factorize_tensor!(A, ::SVDFACTORIZER; bond_cutoff=nothing, trunc_tol=nothing, kwargs...) -> U, S, Vd, ε
    factorize_tensor!(A, ::QRFACTORIZER; kwargs...) -> Q, R
    factorize_tensor!(A, ::LQFACTORIZER; kwargs...) -> L, Q

Mutating counterpart of [`factorize_tensor`](@ref), following `MatrixAlgebraKit`'s own
`!`-suffix convention: `A` may be reused/destroyed as working storage. Always use the returned
factors, not `A` itself, afterwards.
"""
function factorize_tensor!(
    A,
    ::SVDFACTORIZER;
    bond_cutoff::Union{Int,Nothing}=nothing,
    trunc_tol::Union{Real,Nothing}=nothing,
    kwargs...,
)
    trunc = _svd_truncation_strategy(bond_cutoff, trunc_tol)
    return TensorKit.svd_trunc!(A; trunc, kwargs...)
end
factorize_tensor!(A, ::QRFACTORIZER; kwargs...) = TensorKit.qr_compact!(A; kwargs...)
factorize_tensor!(A, ::LQFACTORIZER; kwargs...) = TensorKit.lq_compact!(A; kwargs...)

# SECTION -  AccessEntanglementSpectrumData - capability trait selecting SVD vs QR/LQ

"""
    AccessEntanglementSpectrumData

Trait describing whether a factorization exposes entanglement spectrum data (the singular
values / Schmidt coefficients from an SVD). Two singleton subtypes: [`HasEntanglementSpectrum`](@ref),
[`NoEntanglementSpectrum`](@ref).
"""
abstract type AccessEntanglementSpectrumData end

"""
SVD exposes the entanglement spectrum; see [`AccessEntanglementSpectrumData`](@ref).
"""
struct HasEntanglementSpectrum <: AccessEntanglementSpectrumData end

"""
QR/LQ does not expose the entanglement spectrum; see [`AccessEntanglementSpectrumData`](@ref).
"""
struct NoEntanglementSpectrum <: AccessEntanglementSpectrumData end

"""
    factorize_tensor(A, ::HasEntanglementSpectrum; kwargs...)
    factorize_tensor(A, ::NoEntanglementSpectrum; kwargs...)

Capability-oriented alternative to selecting a factorizer by name: forwards to
`factorize_tensor(A, SVDFACTORIZER(); kwargs...)` / `factorize_tensor(A, QRFACTORIZER(); kwargs...)`
respectively. Choosing [`NoEntanglementSpectrum`](@ref) selects the cheaper QR path when a sweep
only needs a canonical gauge, not the Schmidt spectrum.
"""
function factorize_tensor(A, ::HasEntanglementSpectrum; kwargs...)
    return factorize_tensor(A, SVDFACTORIZER(); kwargs...)
end
function factorize_tensor(A, ::NoEntanglementSpectrum; kwargs...)
    return factorize_tensor(A, QRFACTORIZER(); kwargs...)
end

"""
Mutating counterpart of the `AccessEntanglementSpectrumData`-dispatched `factorize_tensor` methods.
"""
function factorize_tensor!(A, ::HasEntanglementSpectrum; kwargs...)
    return factorize_tensor!(A, SVDFACTORIZER(); kwargs...)
end
function factorize_tensor!(A, ::NoEntanglementSpectrum; kwargs...)
    return factorize_tensor!(A, QRFACTORIZER(); kwargs...)
end

# SECTION -  LeftRight / RightLeft - cursor-direction singletons

"""
    SweepDirection

Abstract root for the two cursor directions a canonicalization sweep can move in:
[`LeftRight`](@ref) (site index increases) and [`RightLeft`](@ref) (site index decreases).
"""
abstract type SweepDirection end

"""
Cursor direction: site index increases (`i -> i+1`). See [`SweepDirection`](@ref).
"""
struct LeftRight <: SweepDirection end

"""
Cursor direction: site index decreases (`i -> i-1`). See [`SweepDirection`](@ref).
"""
struct RightLeft <: SweepDirection end

Base.iterate(::LeftRight, i::Int) = (i, i + 1)
Base.iterate(::RightLeft, i::Int) = (i, i - 1)
Base.iterate(d::LeftRight) = iterate(d, 1)

# SECTION -  step / reabsorb / advance_bond! - the funnel every sweep goes through

"""
    step(::LeftRight, T; access=HasEntanglementSpectrum(), bond_cutoff=nothing) -> raw pieces
    step(::RightLeft, T; access=HasEntanglementSpectrum(), bond_cutoff=nothing) -> raw pieces

Regroup `T`'s legs for the current bond and factorize it, returning the *raw* factorization
output (`(Q, R)` / `(L, Q)` for the exact `NoEntanglementSpectrum` case, or `(U, S, Vd, ε)` for
the truncatable `HasEntanglementSpectrum`/SVD case). Pass the result to [`reabsorb`](@ref) to
turn it into `(site_tensor, remainder)`.
"""
function step(
    ::LeftRight,
    T;
    access::AccessEntanglementSpectrumData=HasEntanglementSpectrum(),
    bond_cutoff::Union{Int,Nothing}=nothing,
)
    T = TensorKit.permute(T, ((1, 2), Tuple(3:TensorKit.numind(T))))
    return factorize_tensor(T, access; bond_cutoff)
end
function step(
    ::RightLeft,
    T;
    access::AccessEntanglementSpectrumData=HasEntanglementSpectrum(),
    bond_cutoff::Union{Int,Nothing}=nothing,
)
    n = TensorKit.numind(T)
    T = TensorKit.permute(T, (Tuple(1:(n - 2)), Tuple((n - 1):n)))
    return factorize_tensor(T, access; bond_cutoff)
end

"""
    reabsorb(direction::SweepDirection, raw_pieces) -> (site_tensor, remainder)

Turn [`step`](@ref)'s raw factorization output into the site tensor to keep and the remainder
tensor to carry into the next bond.
"""
reabsorb(::LeftRight, (Q, R)::Tuple{Any,Any}) = (Q, R)
reabsorb(::RightLeft, (L_, Q)::Tuple{Any,Any}) = (TensorKit.permute(Q, ((1, 2), (3,))), L_)
reabsorb(::LeftRight, (U, S, Vd, ε)::Tuple{Any,Any,Any,Any}) = (U, S * Vd)
function reabsorb(::RightLeft, (U, S, Vd, ε)::Tuple{Any,Any,Any,Any})
    return (TensorKit.permute(Vd, ((1, 2), (3,))), U * S)
end

# SECTION -  step / reabsorb, Val(P)-dispatched - per-site cuts for TensorTrain's canonicalize

"""
    step(direction::SweepDirection, ::Val{P}, T; access=HasEntanglementSpectrum(), bond_cutoff=nothing)

Per-site-cut sibling of [`step`](@ref)'s direction-only methods, for re-gauging an *already
separated* [`TensorTrain`](@ref) site tensor (`P` physical legs) rather than peeling one leg off
a flat multi-site dense tensor. The direction-only `step` methods always slice a **contiguous**
prefix/suffix of `T`'s legs by a fixed count - correct for peeling a single already-vectorized leg
(what [`_sweep`](@ref)/[`orthonormalize`](@ref) do), but wrong here once `P>1`: the bond leg that
must be isolated and handed to the neighbouring site (`vR` for a [`LeftRight`](@ref) cut) sits at
position 3, with any extra physical leg (`σ_bra`, for `P=2`) at position 4 right after it -
isolating `vR` alone needs the **non-contiguous** permutation `(1,2,4) | (3,)`, not a slice.
[`RightLeft`](@ref) needs no `P`-specific case: isolating `vL` alone (`(1,) | (rest)`) is already
arity-agnostic.
"""
function step(
    ::LeftRight,
    ::Val{1},
    T;
    access::AccessEntanglementSpectrumData=HasEntanglementSpectrum(),
    bond_cutoff::Union{Int,Nothing}=nothing,
)
    T = TensorKit.permute(T, ((1, 2), (3,)))
    return factorize_tensor(T, access; bond_cutoff)
end
function step(
    ::LeftRight,
    ::Val{2},
    T;
    access::AccessEntanglementSpectrumData=HasEntanglementSpectrum(),
    bond_cutoff::Union{Int,Nothing}=nothing,
)
    T = TensorKit.permute(T, ((1, 2, 4), (3,)))
    return factorize_tensor(T, access; bond_cutoff)
end
function step(
    ::RightLeft,
    ::Val{P},
    T;
    access::AccessEntanglementSpectrumData=HasEntanglementSpectrum(),
    bond_cutoff::Union{Int,Nothing}=nothing,
) where {P}
    T = TensorKit.permute(T, ((1,), Tuple(2:(P + 2))))
    return factorize_tensor(T, access; bond_cutoff)
end

"""
    reabsorb(direction::SweepDirection, ::Val{P}, raw_pieces) -> (site_tensor, remainder)

`Val(P)`-dispatched sibling of [`reabsorb`](@ref), pairing with the `Val(P)`-dispatched
[`step`](@ref) methods above.
"""
reabsorb(::LeftRight, ::Val{1}, (U, S, Vd, ε)::Tuple{Any,Any,Any,Any}) = (U, S * Vd)
function reabsorb(::LeftRight, ::Val{2}, (U, S, Vd, ε)::Tuple{Any,Any,Any,Any})
    return (TensorKit.permute(U, ((1, 2), (4, 3))), S * Vd)
end
function reabsorb(::RightLeft, ::Val{P}, (U, S, Vd, ε)::Tuple{Any,Any,Any,Any}) where {P}
    return (TensorKit.permute(Vd, ((1, 2), Tuple(3:(P + 2)))), U * S)
end

"""
    advance_bond!(direction, T, bond; access=HasEntanglementSpectrum(), bond_cutoff=nothing,
                  collector=SimStudy.NoOpCollector(), accumulator=SimStudy.NoOpErrorAccumulator())
        -> (site_tensor, remainder)

The single funnel every canonicalization loop in `Subroutines` goes through: factorize `T` at
`bond` (via [`step`](@ref)), record the resulting truncation error into `accumulator` and
(optionally) the full [`SingValSpectrum`](@ref) into `collector` when `access isa HasEntanglementSpectrum`, then return [`reabsorb`](@ref)'s `(site_tensor, remainder)`.

`collector`/`accumulator` default to no-ops (`SimStudy.NoOpCollector`/`SimStudy.NoOpErrorAccumulator`),
so a plain re-gauging sweep pays nothing beyond `factorize_tensor`'s own cost. The per-bond
truncation error comes back for free as part of `factorize_tensor`'s own return tuple when
`access isa HasEntanglementSpectrum`, so `accumulator` always gets it; building the full
`SingValSpectrum` (reading off `S.data`, checking normalization) only happens when `collector`'s
own `SimStudy.RecordingTrait` is `SimStudy.Active` - a caller can therefore track truncation
error alone, cheaply, without a single `SingValSpectrum` ever being constructed.
"""
function advance_bond!(
    direction::SweepDirection,
    T,
    bond::Int;
    access::AccessEntanglementSpectrumData=HasEntanglementSpectrum(),
    bond_cutoff::Union{Int,Nothing}=nothing,
    collector::AbstractCollector=NoOpCollector(),
    accumulator::AbstractErrorAccumulator=NoOpErrorAccumulator(),
)
    raw = step(direction, T; access, bond_cutoff)
    if access isa HasEntanglementSpectrum
        _, S, _, ε = raw
        record!(accumulator, (; direction, bond, ε))

        if RecordingTrait(collector) isa Active
            svals = S.data
            normalized = isapprox(
                sum(abs2, svals), 1.0; atol=sqrt(eps(real(eltype(svals))))
            )
            step!(
                collector,
                (; direction, bond, spectrum=SingValSpectrum(svals, ε, normalized)),
            )
        end
    end
    return reabsorb(direction, raw)
end

"""
    advance_bond!(direction::SweepDirection, ::Val{P}, T, bond::Int; bond_cutoff=nothing,
                  collector=SimStudy.NoOpCollector(), accumulator=SimStudy.NoOpErrorAccumulator())
        -> (site_tensor, remainder)

`Val(P)`-dispatched sibling of [`advance_bond!`](@ref), for cutting an *already separated*
[`TensorTrain`](@ref) site tensor with `P` physical legs (used by `canonicalize`), rather than
peeling one leg off a flat multi-site dense tensor. Same factorize -> record!/step! -> reabsorb
funnel, calling the `Val(P)`-dispatched [`step`](@ref)/[`reabsorb`](@ref) above. Always uses
[`HasEntanglementSpectrum`](@ref) (a per-site canonicalization cut is always an SVD cut).
"""
function advance_bond!(
    direction::SweepDirection,
    ::Val{P},
    T,
    bond::Int;
    bond_cutoff::Union{Int,Nothing}=nothing,
    collector::AbstractCollector=NoOpCollector(),
    accumulator::AbstractErrorAccumulator=NoOpErrorAccumulator(),
) where {P}
    raw = step(direction, Val(P), T; access=HasEntanglementSpectrum(), bond_cutoff)
    _, S, _, ε = raw
    record!(accumulator, (; direction, bond, ε))
    if RecordingTrait(collector) isa Active
        svals = S.data
        normalized = isapprox(sum(abs2, svals), 1.0; atol=sqrt(eps(real(eltype(svals)))))
        step!(
            collector, (; direction, bond, spectrum=SingValSpectrum(svals, ε, normalized))
        )
    end
    return reabsorb(direction, Val(P), raw)
end

# SECTION -  orthogonalize / orthonormalize - full-sweep entry points

_start(::LeftRight, L) = 1
_start(::RightLeft, L) = L
_continue(::LeftRight, i, L) = i < L
_continue(::RightLeft, i, L) = i > 1
_boundary(::LeftRight, L) = L
_boundary(::RightLeft, L) = 1

function _sweep(
    ψtensor,
    L::Int,
    direction::SweepDirection;
    access::AccessEntanglementSpectrumData=HasEntanglementSpectrum(),
    bond_cutoff::Union{Int,Nothing}=nothing,
    collector::AbstractCollector=NoOpCollector(),
    accumulator::AbstractErrorAccumulator=NoOpErrorAccumulator(),
)
    T = TensorKit.insertleftunit(ψtensor, Val(1))
    T = TensorKit.insertrightunit(T, Val(TensorKit.numind(T)))
    As = Vector{TensorKit.TensorMap}(undef, L)
    i = _start(direction, L)
    while _continue(direction, i, L)
        site_tensor, remainder = advance_bond!(
            direction, T, i; access, bond_cutoff, collector, accumulator
        )
        As[i] = site_tensor
        T = remainder
        _, i = iterate(direction, i)
    end
    As[_boundary(direction, L)] = TensorKit.permute(T, ((1, 2), (3,)))
    return As
end

"""
    orthogonalize(ψtensor, L, direction::SweepDirection; access=HasEntanglementSpectrum(), bond_cutoff=nothing) -> As

Decompose the full `L`-site state tensor `ψtensor` into an MPS by sweeping in `direction`
([`LeftRight`](@ref) or [`RightLeft`](@ref)), factoring each bond via [`step`](@ref)/[`reabsorb`](@ref).
Exact: the boundary tensor's natural scale is left alone, so contracting `As` back together
reproduces `ψtensor` exactly (no truncation error aside from any requested `bond_cutoff`). See
[`orthonormalize`](@ref) for the normalized variant.
"""
function orthogonalize(ψtensor, L::Int, direction::SweepDirection; kwargs...)
    return _sweep(ψtensor, L, direction; kwargs...)
end

"""
    orthonormalize(ψtensor, L, direction::SweepDirection; access=HasEntanglementSpectrum(), bond_cutoff=nothing) -> (As, nrm)

Same sweep as [`orthogonalize`](@ref), but additionally measures the boundary tensor's norm and
divides it out, returning a properly unit-normalized MPS alongside the extracted normalization
factor `nrm`.
"""
function orthonormalize(ψtensor, L::Int, direction::SweepDirection; kwargs...)
    As = orthogonalize(ψtensor, L, direction; kwargs...)
    boundary = _boundary(direction, L)
    nrm = TensorKit.norm(As[boundary])
    As[boundary] = As[boundary] / nrm
    return As, nrm
end
