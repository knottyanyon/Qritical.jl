# SECTION - Leg owns the variance

"""
    Leg{V<:IxLoc}

One end of one [`Wire`](@ref), from the perspective of the tensor whose axis it occupies. Where `Wire` owns the shared data (space, spectrum), `Leg` owns everything that is per-end: the index variance (`V`, i.e. `Upper`/`Lower`), and the axis position in the backing array of the parent tensor.

!!! design-note "Design note"
    `Leg` is not one of Joyal's own definitions ([joyal_street_1991](@cite) works with nodes and wires only). It is a concept adapted from the usual condensed-matter convention of labelling a tensor's legs. It serves two purposes at once: it is the bookkeeping record this package uses internally, and it is the bridge that connects that physics convention to the mathematical language used to implement the generalized graph (string diagram) structure.

# Fields

  - `ix :: TIx{V}`, the variance-tagged index. `ix` is authoritative for variance.

    !!! note "Dimension is a cached view"
        The dimension carried by `ix` is a cached view of `wire.space`, it is not independent data. If the two ever disagree, `wire.space` is the one to trust.

  - `wire :: WireId`, the wire this leg is one end of. Storing the id here (rather than the `Wire` itself) is what makes `legs(g, n)` an O(deg) table lookup rather than a scan over every wire in the network ([cormen_2009](@cite), Ch. 22, on adjacency-list graph representations). This field is not redundant with `Wire.start`/`Wire.finish`: those point from the wire to its legs, `wire` points the other way, from the leg back to its wire.
  - `owner :: NodeId`, the node this leg belongs to.
  - `axis :: Int`, the position this leg occupies in the owning tensor's backing array (the polarisation position).

See also: [`Wire`](@ref).

# Examples

```jldoctest
julia> ℓ = Leg(TIx{Upper}(:σ, 2), WireId(1), NodeId(1), 1);

julia> ℓ.ix
TIx{Upper}(:σ, 2)

julia> ℓ.axis
1

julia> ℓ isa Leg{Upper}
true
```
"""
struct Leg{V<:IxLoc}
    ix::TIx{V}
    wire::WireId
    owner::NodeId
    axis::Int
end

# SECTION - make_leg

"""
    make_leg(A, k::Int, wire::WireId, owner::NodeId) -> Leg

Build the `Leg` for axis `k` of tensor `A`.

`A.indices` is declared as `NTuple{Valence,AbstractIx}`, so `A.indices[k]` is only known to be some `AbstractIx`, not concretely `TIx{Upper}` or `TIx{Lower}`. That means the variance parameter `V` cannot be resolved until runtime. `make_leg` accepts this and pays for it once: it hands `A.indices[k]` to `_make_leg`, whose signature `ix::TIx{V} where {V}` makes the compiler specialize on the concrete variance it is actually given. Everything `_make_leg` does afterwards runs against that concrete `V`, as if it had been known from the start.

This is the first of the three inference boundaries named in the topograph design. The alternative would have been a variance-free `PlainLeg` type; this function-barrier approach was chosen instead so that a future fused `MulTIx` leg can be added later as one more `_make_leg` method, without touching this one.

!!! warning "Elementary `TIx` only"
    `make_leg` only handles elementary `TIx` legs. A grouped tensor produced by `group_legs` has `MulTIx` indices, which do not carry a variance yet, so calling `make_leg` on one of those legs is expected to throw `MethodError` until `MulTIx` gains a variance.
"""
make_leg(A, k::Int, wire::WireId, owner::NodeId) = _make_leg(A.indices[k], wire, owner, k)
function _make_leg(ix::TIx{V}, wire::WireId, owner::NodeId, axis::Int) where {V}
    Leg{V}(ix, wire, owner, axis)
end
