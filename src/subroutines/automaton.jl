#=META
source:
  author: Bavithra
  coauthor: Claude Opus 5
  reviewer:
docstrings:
  author: Bavithra
  coauthor: Claude Opus 5
  reviewer:
refs: schollwoeck_2011
credits: N/A
=#

# Hamiltonian-as-weighted-finite-state-automaton construction (Crosswhite & Bacon 2008,
# Schollwoeck's DMRG review sec. 6.1): automaton states become the MPO's virtual bond channels,
# each (state,state) transition carries an operator-valued weight that becomes one block of the
# bond's transfer matrix. Verified against Schollwoeck eq. 182-187 during design: the "one
# additional intermediate state per unit of interaction range per term" rule falls directly out of
# the state-liveness rule below. Deliberately standalone from `Hamiltonian` (`src/operations/`),
# which stays field-less for now - this file only ever sees an explicit `Vector{AutomatonTerm}`.
#
# TODO(perf, deferred): `build_topology`/`fill_topology` always use the fully general
# `Dict{Tuple{PenroseLabel,PenroseLabel},...}`-keyed construction below, even when every term in
# `terms` is nearest-neighbor/k-local for small k. An `InteractionRange` trait (`FiniteRange`,
# `NearestNeighbor`/`NextNearestNeighbor`/`KLocal{K}`) computed once from `terms` could dispatch a
# specialized `Vector`/integer-indexed fast path that skips `PenroseLabel` allocation and `Dict`
# lookups entirely for the common bounded-range case - deliberately not built yet (no profiling
# data shows the general path is actually a bottleneck; see
# docs/superpowers/specs/2026-09-02-hamiltonian-automaton-mpo-design.md sec. 5 for the full
# design). Add only once a real performance need is measured, not speculatively.

# SECTION -  AutomatonTerm - one term of a sum-of-multipartite-operators Hamiltonian

"""
    AutomatonTerm

One term of a Hamiltonian written as a sum of multipartite operators (`H = Σⱼ cⱼ · Oⱼ`): a
coefficient and an ordered list of `(site, local_operator)` pairs. Sites need not be contiguous -
a term like `c * Ŝᶻ₁ ⊗ Ŝᶻ₅` is `AutomatonTerm(c, [1 => Sz, 5 => Sz])`, with sites 2-4 implicitly
identity. This non-contiguity is what makes long-range Hamiltonians (power-law, dipolar, ...)
representable, not just nearest-neighbor chains.

# Fields

  - `coefficient :: Number`
  - `ops         :: Vector{Pair{Int,QProcess}}` - `(site, operator)` pairs; each `operator` a
    single-physical-leg endomorphism (one output leg, one input leg, both the same space) - e.g.
    built via `QProcess(pauli_z_tensormap; output_roles=PhysicalLeg(), input_roles=PhysicalLeg())`.
    Order/duplication of sites does not matter - [`build_automaton`](@ref) sorts internally.
"""
struct AutomatonTerm
    coefficient::Number
    ops::Vector{Pair{Int,QProcess}}
end

# SECTION -  HamiltonianAutomaton - the retained, inspectable automaton structure

"""
    HamiltonianAutomaton

The retained, inspectable result of [`build_automaton`](@ref) - kept separate from
[`materialize`](@ref)'s `TensorKit`/`QProcess` block-assembly so a caller (or a future
operator-entanglement-growth / per-term structural analysis pass) can query the automaton's own
combinatorial structure without ever materializing a tensor.

# Fields

  - `states      :: Vector{Vector{PenroseLabel}}` - `states[b+1]` is the live state set at bond
    `b` (`b = 0..L`), each state a distinct `PenroseLabel`.
  - `transitions :: Vector{Dict{Tuple{PenroseLabel,PenroseLabel},Pair{Number,QProcess}}}` -
    `transitions[i]` is the transition table at site `i` (`i = 1..L`), keyed by
    `(state_in, state_out)` (state_in lives at bond `i-1`, state_out at bond `i`), valued
    `(coefficient, operator)`.
  - `terms       :: Vector{AutomatonTerm}` - the original terms, kept for provenance.
"""
struct HamiltonianAutomaton
    states::Vector{Vector{PenroseLabel}}
    transitions::Vector{Dict{Tuple{PenroseLabel,PenroseLabel},Pair{Number,QProcess}}}
    terms::Vector{AutomatonTerm}
end

# SECTION -  AutomatonTopology - reusable structure, independent of coefficient/operator content

"""
    AutomatonTopology

The combinatorial *structure* of a [`HamiltonianAutomaton`](@ref) - which states exist per bond,
which `(state_in, state_out)` transition slots exist at all - kept separate from the numeric
`(coefficient, operator)` content that fills those slots. This structure depends only on `terms`'
**site pattern** (which sites each term touches), never on coefficient values or the operator
content itself, so it's exactly the part worth reusing across many coefficient realizations of the
same coupling pattern (disorder averaging, parameter sweeps): build it once via
[`build_topology`](@ref), then re-attach content per realization via [`fill_topology`](@ref)
instead of redoing the state-liveness combinatorics every time.

# Fields

  - `states     :: Vector{Vector{PenroseLabel}}` - same shape as
    [`HamiltonianAutomaton`](@ref)'s `states`.
  - `spans      :: Vector{Tuple{Int,Int}}` - per-term `(first_site, last_site)`.
  - `term_state :: Vector{PenroseLabel}` - per-term in-flight state identity.
"""
struct AutomatonTopology
    states::Vector{Vector{PenroseLabel}}
    spans::Vector{Tuple{Int,Int}}
    term_state::Vector{PenroseLabel}
end

"""
    build_topology(terms::Vector{AutomatonTerm}, L::Int) -> AutomatonTopology

Determine the state/transition *structure* only (per Schollwoeck sec. 6.1, see
[`build_automaton`](@ref) for the full construction this is half of) - no operator content, no
`TensorMap` work. Every bond always carries a `start` state and an `accept` state, plus one
dedicated in-flight state per term `j` at every bond strictly between its first and last touched
site - bond dimension at bond `b` is `2 + |{terms straddling b}|`.
"""
function build_topology(terms::Vector{AutomatonTerm}, L::Int)
    # TODO(perf, deferred): this is the dispatch point an InteractionRange fast path would hook
    # into - classify `terms` (e.g. max span <= k) and branch to a Vector/integer-indexed
    # construction for FiniteRange instead of the general PenroseLabel/Dict one below. See the
    # TODO at the top of this file for why it's not built yet.
    start = PenroseLabel(:start)
    accept = PenroseLabel(:accept)
    term_state = [PenroseLabel(:term, j) for j in eachindex(terms)]

    spans = Vector{Tuple{Int,Int}}(undef, length(terms))
    for (j, t) in enumerate(terms)
        sites_j = sort(unique(first.(t.ops)))
        spans[j] = (first(sites_j), last(sites_j))
    end

    states = Vector{Vector{PenroseLabel}}(undef, L + 1)
    for b in 0:L
        live = PenroseLabel[start, accept]
        for j in eachindex(terms)
            f, l = spans[j]
            f <= b < l && push!(live, term_state[j])
        end
        states[b + 1] = live
    end

    return AutomatonTopology(states, spans, term_state)
end

# SECTION -  fill_topology / build_automaton - attach numeric content

function _add_transition!(table, key, coefficient, op::QProcess)
    if haskey(table, key)
        existing_coefficient, existing_op = table[key]
        merged = existing_coefficient * tensor(existing_op) + coefficient * tensor(op)
        table[key] =
            1 => QProcess(merged; output_roles=PhysicalLeg(), input_roles=PhysicalLeg())
    else
        table[key] = coefficient => op
    end
    return table
end

"""
    fill_topology(topology::AutomatonTopology, terms::Vector{AutomatonTerm}, physical_spaces)
        -> HamiltonianAutomaton

Re-attach numeric `(coefficient, operator)` content to `topology`'s already-determined
state/transition structure, producing a full [`HamiltonianAutomaton`](@ref) without redoing the
state-liveness combinatorics [`build_topology`](@ref) already did.

**Caller contract**: `terms` must have the same site *pattern* (same spans, same term count) as
whatever was passed to [`build_topology`](@ref) to build `topology` - only coefficients and
operator content may differ. Checked cheaply (term count only, not a full span re-derivation -
re-checking spans would redo exactly the work this function exists to avoid); passing `terms` with
a genuinely different site pattern than `topology` was built from produces a silently wrong
automaton, not an error.

`physical_spaces` is either a single `TensorKit.ElementarySpace` (broadcast to every site) or a
`Vector` of `L` spaces (one per site) - see [`build_automaton`](@ref).
"""
function fill_topology(
    topology::AutomatonTopology,
    terms::Vector{AutomatonTerm},
    physical_spaces::Union{
        TensorKit.ElementarySpace,AbstractVector{<:TensorKit.ElementarySpace}
    },
)
    length(terms) == length(topology.spans) || throw(
        ArgumentError(
            "fill_topology: terms has $(length(terms)) term(s), topology was built for " *
            "$(length(topology.spans)) - fill_topology only re-attaches numeric content for " *
            "the same term site-pattern build_topology saw; build a fresh topology if the " *
            "pattern itself changed.",
        ),
    )

    L = length(topology.states) - 1
    spaces = physical_spaces isa AbstractVector ? physical_spaces : fill(physical_spaces, L)
    length(spaces) == L ||
        throw(ArgumentError("physical_spaces must have length L=$L, got $(length(spaces))"))

    start = PenroseLabel(:start)
    accept = PenroseLabel(:accept)

    term_ops = Vector{Dict{Int,QProcess}}(undef, length(terms))
    for (j, t) in enumerate(terms)
        term_ops[j] = Dict(site => op for (site, op) in t.ops)
    end

    transitions = [
        Dict{Tuple{PenroseLabel,PenroseLabel},Pair{Number,QProcess}}() for _ in 1:L
    ]
    identity_at(site) = identity_process(TIx(spaces[site], PhysicalLeg()))

    for i in 1:L
        _add_transition!(transitions[i], (start, start), 1, identity_at(i))
        _add_transition!(transitions[i], (accept, accept), 1, identity_at(i))

        for j in eachindex(terms)
            f, l = topology.spans[j]
            (i < f || i > l) && continue
            if f == l
                if i == f
                    _add_transition!(
                        transitions[i],
                        (start, accept),
                        terms[j].coefficient,
                        term_ops[j][i],
                    )
                end
                continue
            end
            if i == f
                _add_transition!(
                    transitions[i],
                    (start, topology.term_state[j]),
                    terms[j].coefficient,
                    term_ops[j][i],
                )
            elseif i == l
                _add_transition!(
                    transitions[i], (topology.term_state[j], accept), 1, term_ops[j][i]
                )
            else
                op = get(term_ops[j], i, nothing)
                op = isnothing(op) ? identity_at(i) : op
                _add_transition!(
                    transitions[i], (topology.term_state[j], topology.term_state[j]), 1, op
                )
            end
        end
    end

    return HamiltonianAutomaton(topology.states, transitions, terms)
end

"""
    build_automaton(terms::Vector{AutomatonTerm}, L::Int, physical_spaces) -> HamiltonianAutomaton

Build the finite-state-automaton state/transition structure for a Hamiltonian `H = Σⱼ cⱼ · Oⱼ`
on an `L`-site chain, per Schollwoeck sec. 6.1 - `build_topology(terms, L)` followed by
[`fill_topology`](@ref). Prefer calling [`build_topology`](@ref)/[`fill_topology`](@ref)
separately when re-materializing many coefficient realizations of the same term site-pattern
(disorder averaging, parameter sweeps): this convenience wrapper always redoes the structure
determination from scratch.

`physical_spaces` is either a single `TensorKit.ElementarySpace` (broadcast to every site, the
common uniform-chain case) or a `Vector` of `L` spaces (one per site) - needed to build the
identity pass-through operator at sites a term doesn't touch, and for the two permanent
`start->start`/`accept->accept` identity self-loops present at every site regardless of `terms`.
"""
function build_automaton(
    terms::Vector{AutomatonTerm},
    L::Int,
    physical_spaces::Union{
        TensorKit.ElementarySpace,AbstractVector{<:TensorKit.ElementarySpace}
    },
)
    return fill_topology(build_topology(terms, L), terms, physical_spaces)
end

# SECTION -  materialize - block-assembly into an actual MPOperator

"""
    materialize(automaton::HamiltonianAutomaton) -> MPOperator{UnknownGauge,FiniteSupport}

Turn `automaton`'s combinatorial state/transition structure into an actual `TensorKit`-backed
[`MPOperator`](@ref). Each bond `b` becomes a plain `TIx(dim, VirtualLeg())` of dimension
`length(automaton.states[b+1])`; each site tensor is built as a dense 4-index array (one axis per
leg: `vL, σ_ket, vR, σ_bra`, matching [`MPOperator`](@ref)'s `(vL,σ_ket) | (vR,σ_bra)` storage
convention) with every transition's `coefficient * tensor(operator)` written into its
`(channel_in, channel_out)` sub-block, then wrapped via `TensorKit.TensorMap` and the same
`_finalize_site` helper `to_mps`/`canonicalize` already use. Tagged `UnknownGauge` (freshly built,
not yet canonicalized) and `FiniteSupport`.
"""
function materialize(automaton::HamiltonianAutomaton)
    L = length(automaton.transitions)
    bond_ix = [TIx(length(automaton.states[b + 1]), VirtualLeg()) for b in 0:L]
    channel_index = [
        Dict(q => k for (k, q) in enumerate(automaton.states[b + 1])) for b in 0:L
    ]

    sites = Vector{QProcess}(undef, L)
    for i in 1:L
        table = automaton.transitions[i]
        isempty(table) && error(
            "materialize: site $i has no transitions - is L consistent with terms' spans?",
        )

        any_op = first(values(table)).second
        σ_ket = outputs(any_op)[1]
        σ_bra = inputs(any_op)[1]
        n_in = length(automaton.states[i])
        n_out = length(automaton.states[i + 1])

        arr = zeros(ComplexF64, n_in, dim(σ_ket), n_out, dim(σ_bra))
        idx_in = channel_index[i]
        idx_out = channel_index[i + 1]
        for ((state_in, state_out), (coefficient, op)) in table
            block = convert(Array, tensor(op))
            arr[idx_in[state_in], :, idx_out[state_out], :] .+= coefficient .* block
        end

        W = TensorKit.TensorMap(
            arr, (space(bond_ix[i]) ⊗ space(σ_ket)) ← (space(bond_ix[i + 1]) ⊗ space(σ_bra))
        )
        sites[i] = _finalize_site(Val(2), W)
    end

    return MPOperator{UnknownGauge,FiniteSupport}(sites, 0, 0, nothing, 0.0)
end

# SECTION -  split_commuting_groups - partition terms for Trotterization

# Design history: a general site-overlap-conflict-graph-coloring approach was worked through and
# found to have a real correctness subtlety worth recording, not just an implementation detail.
# Mixing nearest-neighbor bond terms with separate on-site field terms (e.g. TFIM) creates a
# genuine 3-clique in the conflict graph: a field term at site i shares a site with BOTH of its
# neighboring bond terms, and those two bond terms also share a site with each other. Since
# [Ŝˣᵢ, Ŝᶻᵢ⊗Ŝᶻᵢ₊₁] ≠ 0 in general, this isn't a modeling artifact - 3 colors are mathematically
# required whenever bond and field terms are kept as separate AutomatonTerms, unfixable by
# reordering. Real TEBD implementations avoid this by folding on-site terms into a neighboring
# bond term's operator sum BEFORE splitting - a distinct, upstream term-merging concern, not
# something this function does. Per explicit decision: defer the general graph-coloring solution
# (and the merging question) entirely; ship only the simple even/odd default now, via a strategy
# tag so the deferred path has a clearly-marked place to slot in later.

"""
    SplitStrategy

Abstract root of the term-splitting strategy tags for [`split_commuting_groups`](@ref). Concrete
subtypes: [`EvenOddSplit`](@ref) (default), [`GraphColoring`](@ref) (not yet implemented).
"""
abstract type SplitStrategy end

"""
2 groups by parity of each term's first touched site - the standard odd/even-bond split. Only
rigorously guarantees mutual commutativity within each group for a pure nearest-neighbor-bond
Hamiltonian (no on-site field terms mixed in, no long-range terms) - not validated for the general
case; see [`GraphColoring`](@ref). See [`SplitStrategy`](@ref).
"""
struct EvenOddSplit <: SplitStrategy end

"""
General site-overlap-conflict-graph-coloring strategy, correctly handling mixed bond/field-term
Hamiltonians and long-range terms. **Not yet implemented** - deferred. See [`SplitStrategy`](@ref).
"""
struct GraphColoring <: SplitStrategy end

"""
    split_commuting_groups(terms::Vector{AutomatonTerm}, strategy::SplitStrategy=EvenOddSplit())
        -> Vector{Vector{AutomatonTerm}}

Partition `terms` into groups of mutually commuting terms, for use as [`Trotterization`](@ref)'s
[`sequence`](@ref) input (e.g. `LieTrotter`/`SuzukiTrotter` applied to `[odd_group, even_group]`).
Dispatches on `strategy` - see [`EvenOddSplit`](@ref)/[`GraphColoring`](@ref) for what each one
actually guarantees.
"""
function split_commuting_groups(
    terms::Vector{AutomatonTerm}, strategy::SplitStrategy=EvenOddSplit()
)
    return split_commuting_groups(strategy, terms)
end

function split_commuting_groups(::EvenOddSplit, terms::Vector{AutomatonTerm})
    n = length(terms)
    n == 0 && return Vector{AutomatonTerm}[]

    groups = [AutomatonTerm[], AutomatonTerm[]]
    for t in terms
        first_site = minimum(first.(t.ops))
        push!(groups[isodd(first_site) ? 1 : 2], t)
    end
    return groups
end

function split_commuting_groups(::GraphColoring, terms::Vector{AutomatonTerm})
    return error("GraphColoring strategy is not yet implemented - use EvenOddSplit for now")
end
