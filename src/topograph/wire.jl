# ==== Wire — owns the space ===================================================
# WireId lives in ids.jl, loaded before this file.

"""
    Wire{S}

One wire of a generalized topological graph: the object that carries a shared index between (at most) two tensors. Joyal's generalized topological graph attaches each wire's two ends *optionally* — a wire may be pinned (both ends attached), half-loose (one end), loose (neither), or a closed circle. `Wire` stores the two ends directly and **derives** which of these four states it is in via [`attachment`](@ref); the state itself is never stored, so it cannot drift out of sync with the ends.

# Fields

  - `id :: WireId`
  - `space :: S` — the shared dimension (or, once graded spaces land, an
    `ElementarySpace`). This is authoritative: a `Leg` at either end reads its
    dimension from here rather than storing its own copy.
  - `start :: Union{LegId,Nothing}` — the end at `γ(0)`; a leg attached here sees
    this wire as `Outgoing` (its tensor's codomain, `TIx{Lower}`).
  - `finish :: Union{LegId,Nothing}` — the end at `γ(1)`; a leg attached here
    sees this wire as `Incoming` (its tensor's domain, `TIx{Upper}`).
  - `closed :: Bool` — `true` for a circle: a closed loop with no nodes at all,
    contributing only a scalar factor `dim(space)` and invisible to `@tensor`.
  - `label :: Symbol` — a human-readable name (analogous to a QSpace itag).
  - `dof :: Union{Nothing,AbstractDoF}` — the physical degree of freedom carried
    by this wire, or `nothing` for a virtual bond.
  - `spectrum :: Union{Nothing,AbstractSpectrum}` — the Schmidt spectrum at this
    cut, or `nothing`. **Invariant: a non-`nothing` spectrum requires
    `dof === nothing`** — Schmidt spectra live on cuts, and cuts are virtual
    bonds, never physical legs. Enforced in the inner constructor.

See also: [`Leg`](@ref), [`attachment`](@ref).

# Examples

```jldoctest
julia> w = Wire(WireId(1), 4; label=:vL);

julia> w.space
4

julia> w.dof === nothing
true

julia> attachment(w)
Loose()
```
"""
mutable struct Wire{S}
    id::WireId
    space::S
    start::Union{LegId,Nothing}
    finish::Union{LegId,Nothing}
    closed::Bool
    label::Symbol
    dof::Union{Nothing,AbstractDoF}
    spectrum::Union{Nothing,AbstractSpectrum}

    function Wire{S}(
        id::WireId,
        space::S;
        start::Union{LegId,Nothing}=nothing,
        finish::Union{LegId,Nothing}=nothing,
        closed::Bool=false,
        label::Symbol=:_,
        dof::Union{Nothing,AbstractDoF}=nothing,
        spectrum::Union{Nothing,AbstractSpectrum}=nothing,
    ) where {S}
        # A Schmidt spectrum records the singular values of a cut; a cut is by definition a virtual bond, so a wire carrying a physical dof (a tensor leg with a local Hilbert space) can never also carry a spectrum. One check here rules out an entire class of bugs where a physical leg is mistaken for a cut, or vice versa.
        if dof !== nothing && spectrum !== nothing
            throw(
                ArgumentError(
                    "Wire cannot carry both a dof and a spectrum — a Schmidt " *
                    "spectrum lives only on a virtual bond (dof === nothing).",
                ),
            )
        end
        new{S}(id, space, start, finish, closed, label, dof, spectrum)
    end
end

# Outer constructor: infers S from the `space` argument so callers write `Wire(WireId(1), 4; label=:vL)` without spelling out the type parameter.
Wire(id::WireId, space::S; kwargs...) where {S} = Wire{S}(id, space; kwargs...)