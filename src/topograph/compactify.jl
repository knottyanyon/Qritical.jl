# SECTION - compactify and boundary

"""
    compactify(g::TensorNetwork) -> TensorNetwork

Compactifying a graph means closing off every open end so nothing is left dangling. `g` is the network as given; `Ĝ` ("G-hat") is what compactifying it produces, following Joyal's own construction ([joyal_street_1991](@cite), p. 61): a new node is stitched onto each free wire end, purely to hold it in place, nothing else about `g` changes.

Concretely, each of the four attachment states is handled differently:

  - **Pinned**, both ends already attached: left alone.
  - **Half-loose**, one end already attached: adjoin one fresh outer node to the other end.
  - **Loose**, neither end attached: adjoin two fresh outer nodes, one to each end.
  - **Circle**, no ends at all to begin with: left alone.

!!! note "Why compactify at all"
    After compactifying, every wire that was half-loose or loose in `g` is `Pinned()` in `Ĝ`. That is what lets [`is_ordinary`](@ref) reduce to a single question, whether `Ĝ` has any circles left, rather than having to reason about half-loose and loose wires directly. It is also exactly what [`boundary`](@ref) reads off: the nodes compactify had to invent are precisely `g`'s open ends.

`g` itself is never mutated, `compactify` always returns a new network.

See also: [`boundary`](@ref), which reads off the nodes this function adds.

# Examples

```jldoctest
julia> g = TensorNetwork();

julia> n = add_node!(g);

julia> w = add_wire!(g, 2; label=:σ);

julia> A = QTensor(randn(2, 2), (upper(:σ, 2), lower(:τ, 2)));

julia> ℓ = add_leg!(g, A, n, w);

julia> attach!(g, ℓ);

julia> attachment(g.wires[w])
HalfLoose()

julia> ĝ = compactify(g);

julia> attachment(ĝ.wires[w])
Pinned()

julia> length(nodes(ĝ)) - length(nodes(g))
1
```
"""
function compactify(g::TensorNetwork)
    # Wire is mutable, so a shallow copy would still share the same Wire objects as g. Mutating
    # ĝ's wires below would then also mutate g's, corrupting the network this is meant to leave
    # untouched. A deepcopy gives ĝ its own Wire/Leg objects to modify freely.
    ĝ = deepcopy(g)
    for w in wires(ĝ)
        wire = ĝ.wires[w]
        a = attachment(wire)
        (a === Pinned() || a === Circle()) && continue
        label = wire.label
        dim = wire.space
        if a === HalfLoose() # only one end is attached so the new outer leg only needs attach!
            if wire.start !== nothing
                n = add_node!(ĝ)
                ℓ = _add_boundary_leg!(ĝ, n, w, TIx{Upper}(label, dim))
                attach!(ĝ, ℓ)
            else
                n = add_node!(ĝ)
                ℓ = _add_boundary_leg!(ĝ, n, w, TIx{Lower}(label, dim))
                attach!(ĝ, ℓ)
            end
        else   # BOTH ends free making is Loose() so needs pin! 
            n_start = add_node!(ĝ)
            n_finish = add_node!(ĝ)
            ℓ_start = _add_boundary_leg!(ĝ, n_start, w, TIx{Lower}(label, dim))
            ℓ_finish = _add_boundary_leg!(ĝ, n_finish, w, TIx{Upper}(label, dim))
            pin!(ĝ, ℓ_start, ℓ_finish)
        end
    end
    return ĝ
end

# Register a leg built from a bare TIx to get a fresh outer node that is not part of the tensor's existing indices. This is used be compactify to attach a wire's open end to a fresh outer node that has no backing QTensor. 
function _add_boundary_leg!(g::TensorNetwork, owner::NodeId, wire::WireId, ix::TIx)
    axis = length(g.node_legs[owner]) + 1
    ℓ = _make_leg(ix, wire, owner, axis)
    # Shares the actual table bookkeeping with add_leg! via _register_leg!
    return _register_leg!(g, owner, ℓ)
end

"""
    boundary(g::TensorNetwork) -> Vector{NodeId}

The outer nodes [`compactify`](@ref) would add to `g`, i.e. the nodes of `Ĝ` that are not already nodes of `g`. Empty exactly when `g` has no half-loose or loose wires, e.g. a fully contracted network.

# Examples

```jldoctest
julia> g = TensorNetwork();

julia> w = add_wire!(g, 3; label=:free);

julia> length(boundary(g))
2
```
"""
boundary(g::TensorNetwork) = setdiff(nodes(compactify(g)), nodes(g))

"""
    is_ordinary(g::TensorNetwork) -> Bool

`true` when `g` has no circles, checked on `Ĝ` (see [`compactify`](@ref)) rather than on `g` directly, since `compactify` is what turns half-loose and loose wires into pinned ones without touching circles, so any circle remaining in `Ĝ` was already a circle in `g`.
"""
function is_ordinary(g::TensorNetwork)
    !any(w -> attachment(w) === Circle(), values(compactify(g).wires))
end

"""
    is_closed(g::TensorNetwork) -> Bool

`true` when every wire in `g` is already [`Pinned`](@ref).

!!! tip "A fully contracted network is a bare scalar"
    Once every wire is pinned, nothing is left open: contracting the whole network leaves no free legs, so its value collapses to a single scalar.

Checked directly on `g`, not on `Ĝ`: an open physical leg (half-loose) must make this `false`, but compactifying `g` would attach a fresh outer node to exactly that leg and pin it there, hiding the very thing this function is meant to detect.
"""
is_closed(g::TensorNetwork) = all(w -> attachment(w) === Pinned(), values(g.wires))
