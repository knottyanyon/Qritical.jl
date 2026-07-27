# SECTION - attach!, pin! and cut!

"""
    attach!(g::TensorNetwork, ℓ::LegId) -> TensorNetwork

Attach leg `ℓ` to its own wire, at whichever end matches `ℓ`'s [`orientation`](@ref): an [`Outgoing`](@ref) leg sets the wire's `start`, an [`Incoming`](@ref) leg sets its `finish`.

This is how a wire becomes half-loose: [`add_leg!`](@ref) only ever registers a leg and records which wire it belongs to, it does not touch that wire's `start`/`finish`. A leg with no partner (an MPS site's open physical leg, say) needs `attach!` on its own; a leg with a partner goes through [`pin!`](@ref) instead, which attaches both sides atomically. Throws `ArgumentError` if the matching end is already attached.

# Examples

```jldoctest
julia> g = TensorNetwork();

julia> n = add_node!(g);

julia> w = add_wire!(g, 2; label=:σ);

julia> A = QTensor(randn(2, 2), (upper(:σ, 2), lower(:τ, 2)));

julia> ℓ = add_leg!(g, A, n, w);

julia> attachment(g.wires[w])
Loose()

julia> attach!(g, ℓ);

julia> attachment(g.wires[w])
HalfLoose()
```
"""
function attach!(g::TensorNetwork, ℓ::LegId)
    leg = g.legs[ℓ]
    w = g.wires[leg.wire]
    _attach!(g, w, ℓ, leg_orientation(leg))
    return g
end

function _attach!(::TensorNetwork, w::Wire, ℓ::LegId, ::Outgoing)
    w.start === nothing ||
        throw(ArgumentError("wire $(w.id) is already attached at its start"))
    w.start = ℓ
end
function _attach!(::TensorNetwork, w::Wire, ℓ::LegId, ::Incoming)
    w.finish === nothing ||
        throw(ArgumentError("wire $(w.id) is already attached at its finish"))
    w.finish = ℓ
end

"""
    pin!(g::TensorNetwork, a::LegId, b::LegId) -> TensorNetwork

Pin legs `a` and `b` together, attaching both ends of the wire they already share.

!!! tip
    On a tensor network, this is what turns a shared wire into a contracted bond between two tensors.

`a` and `b` must already sit on the same wire (added there by two calls to [`add_leg!`](@ref) with the same `wire` argument). `pin!` never merges two different wires into one, it only ever sets the `start`/`finish` fields of the one wire both legs point to:

```text
before pin!, wire is Loose:

    node A ●(a)                          (b)●  node B


after pin!(g, a, b), wire is Pinned:

    node A ●───────▶ start   finish ◀───────●  node B
                (a, Outgoing)   (b, Incoming)
```

Which leg becomes `start` and which becomes `finish` is decided by dispatching on each leg's [`orientation`](@ref), never by an `if`: exactly one of `a`, `b` must be [`Outgoing`](@ref) and the other [`Incoming`](@ref). The outgoing one becomes the wire's start, the incoming one becomes its finish, this matches the rule that a start is always outgoing and a finish is always incoming.

!!! warning "Same-orientation pairs are rejected"

    Pinning two outgoing legs, or two incoming legs, together throws `ArgumentError`. A bond must pair exactly one `Upper` (incoming) index with one `Lower` (outgoing) index.

See also: [`cut!`](@ref), the inverse operation.
"""
function pin!(g::TensorNetwork, a::LegId, b::LegId)
    la = g.legs[a]
    lb = g.legs[b]
    _pin!(g, a, la, b, lb, leg_orientation(la), leg_orientation(lb))
    return g
end

function _pin!(
    g::TensorNetwork, a::LegId, la::Leg, b::LegId, lb::Leg, ::Outgoing, ::Incoming
)
    _set_ends!(g, a, la, b, lb)
end
function _pin!(
    g::TensorNetwork, a::LegId, la::Leg, b::LegId, lb::Leg, ::Incoming, ::Outgoing
)
    _pin!(g, b, lb, a, la, Outgoing(), Incoming())
end

# Less specific than the two methods above, so same-orientation pairs land here instead of hitting a MethodError that names an internal function and does not explain why.
function _pin!(
    ::TensorNetwork, ::LegId, ::Leg, ::LegId, ::Leg, o1::LegOrientation, o2::LegOrientation
)
    throw(
        ArgumentError(
            "cannot pin $o1 to $o2, a bond pairs exactly one Upper (Incoming) leg with one " *
            "Lower (Outgoing) leg",
        ),
    )
end

function _set_ends!(
    g::TensorNetwork, outgoing::LegId, lout::Leg, incoming::LegId, linc::Leg
)
    lout.wire == linc.wire ||
        throw(ArgumentError("cannot pin two legs that do not already sit on the same wire"))
    # Check both ends are free before attaching either: attach! alone would let a failure on the second call leave the wire partially pinned from the first, this keeps pin! atomic.
    w = g.wires[lout.wire]
    w.start === nothing ||
        throw(ArgumentError("wire $(w.id) is already attached at its start"))
    w.finish === nothing ||
        throw(ArgumentError("wire $(w.id) is already attached at its finish"))
    attach!(g, outgoing)
    attach!(g, incoming)
    return g
end

"""
    cut!(g::TensorNetwork, w::WireId) -> TensorNetwork

Detach both ends of wire `w`, the structural inverse of [`pin!`](@ref). The wire itself is not removed from `g`, only its `start`/`finish` fields are cleared, so `attachment(g.wires[w])` becomes `Loose()` afterwards regardless of what it was before.

!!! note
    `cut!` is idempotent: cutting an already-loose wire leaves it unchanged, there is nothing to detach.
"""
function cut!(g::TensorNetwork, w::WireId)
    wire = g.wires[w]
    wire.start = nothing
    wire.finish = nothing
    return g
end
