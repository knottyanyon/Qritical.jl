# SECTION - is_progressive

"""
    is_progressive(g::TensorNetwork) -> Bool

!!! definition "Circuit"

    A **circuit** is a directed cycle. This is distinct from a **circle** ([`Circle`](@ref)), a closed component with no nodes at all, already ruled out by [`is_ordinary`](@ref) before `is_progressive` ever looks for a circuit.

!!! definition "Progressive"

    A generalized topological graph is **progressive** when it is oriented, ordinary (no circles), and contains no circuit.

!!! note "In practice"

      - A `TensorNetwork` is always oriented, every leg carries a variance, so the real work below is checking ordinariness and the absence of circuits.
      - OBC-MPS, MPO, and TTN are progressive.
      - A PBC-MPS trace `Tr(A^{j_1} \\cdots A^{j_n})`, and any closed `⟨ψ|ψ⟩` network, are not: tracing closes a bond back on itself, which is exactly a circuit.
      - Sweep algorithms need `is_progressive`; transfer-matrix algorithms do not.
"""
function is_progressive(g::TensorNetwork)
    is_ordinary(g) || return false
    return !has_circuit(compactify(g))
end

"""
    has_circuit(g::TensorNetwork) -> Bool

`true` if the directed graph formed by `g`'s pinned wires contains a directed cycle.

Each pinned wire is one directed edge, from the node owning its `start` (the `Outgoing` leg) to the node owning its `finish` (the `Incoming` leg), per [`Wire`](@ref)'s own convention.

!!! note "A trace is a self-loop"

    A trace is a pinned wire whose two legs belong to the same node, that makes it a self-loop in this directed graph, and a self-loop is itself a cycle of length one. `has_circuit` catches traces for free, with no special case needed.
"""
function has_circuit(g::TensorNetwork)
    # TODO(human): detect a directed cycle among g's Pinned wires.
    #
    # Build the edge set first: for each w in wires(g) with attachment(g.wires[w]) === Pinned(),
    # add a directed edge from g.legs[g.wires[w].start].owner to g.legs[g.wires[w].finish].owner.
    # (HalfLoose/Loose/Circle wires contribute no edge; is_progressive already filters those
    # out via is_ordinary before ever calling this on a non-compactified g.)
    #
    # Then decide whether that directed graph over nodes(g) contains a cycle. Consider: DFS
    # with a three-colour (white/gray/black) visited marking, vs. Kahn's algorithm (repeatedly
    # remove zero-indegree nodes; a cycle exists iff nodes remain when no more can be removed).
    # Either is fine, just make sure a self-loop (a trace) is caught, and that a graph with
    # several disconnected pieces is still checked in full, not just from one starting node.
    error("has_circuit: not yet implemented")
end
