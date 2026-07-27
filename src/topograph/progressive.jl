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

Implemented via Kahn's algorithm ([kahn_1962](@cite)): repeatedly strip nodes whose remaining indegree is zero. If every node can eventually be stripped, the graph is acyclic; whatever is left over once no more nodes can be stripped is exactly the set of nodes on (or reachable only through) a cycle.

!!! note "Why Kahn's algorithm and not DFS three-colouring"
    Both run in `O(V + E)`, neither is faster or more modern than the other, Kahn dates to 1962 ([kahn_1962](@cite)) and DFS-based cycle detection is from the same era of graph theory. General-purpose graph libraries (e.g. `Graphs.jl`, not a dependency of Qritical, its `is_cyclic`/`topological_sort` use DFS three-colouring) tend to prefer DFS because they already need DFS traversal machinery for many other algorithms (connected components, Tarjan's/Kosaraju's SCC, articulation points), so cycle detection can reuse that existing infrastructure rather than needing its own. There is no such shared machinery to reuse here, this is the only place in the topograph layer that needs a directed-graph traversal, so the choice came down to whichever framing made correctness easiest to see: "nothing left over after stripping" is a direct enough statement of "no cycle" that self-loops and disconnected pieces both fall out without special-casing.
"""
function has_circuit(g::TensorNetwork)
    # Kahn's algorithm: repeatedly strip nodes with zero remaining indegree. If every node can eventually be stripped, the graph is a DAG; anything left over once no more can be stripped means those nodes form (or sit on) a cycle. A self-loop node's indegree never reaches zero on its own, so a trace is caught with no special-casing, and starting the queue from ALL zero-indegree nodes (not just one) covers disconnected pieces as well.
    ns = nodes(g)
    out_edges = Dict{NodeId,Vector{NodeId}}(n => NodeId[] for n in ns)
    indegree = Dict{NodeId,Int}(n => 0 for n in ns)
    for w in wires(g)
        wire = g.wires[w]
        attachment(wire) === Pinned() || continue
        from = g.legs[wire.start].owner
        to = g.legs[wire.finish].owner
        push!(out_edges[from], to)
        indegree[to] += 1
    end

    queue = [n for n in ns if indegree[n] == 0]
    visited = 0
    while !isempty(queue)
        n = pop!(queue)
        visited += 1
        for m in out_edges[n]
            indegree[m] -= 1
            indegree[m] == 0 && push!(queue, m)
        end
    end
    return visited < length(ns)
end
