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

# SECTION -  build_automaton - pure combinatorics, no TensorMap work

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
    build_automaton(terms::Vector{AutomatonTerm}, L::Int, physical_spaces) -> HamiltonianAutomaton

Build the finite-state-automaton state/transition structure for a Hamiltonian `H = Σⱼ cⱼ · Oⱼ`
on an `L`-site chain, per Schollwoeck sec. 6.1: every bond always carries a `start` state ("no
term has started yet here") and an `accept` state ("every term touching earlier sites has already
completed"), plus one dedicated in-flight state per term `j` at every bond strictly between its
first and last touched site. Bond dimension at bond `b` is therefore `2 + |{terms straddling b}|`

  - the range-dependent cost this construction is built around, not a nearest-neighbor convention.

`physical_spaces` is either a single `TensorKit.ElementarySpace` (broadcast to every site, the
common uniform-chain case) or a `Vector` of `L` spaces (one per site) - needed to build the
identity pass-through operator at sites a term doesn't touch, and for the two permanent
`start->start`/`accept->accept` identity self-loops present at every site regardless of `terms`.

Pure combinatorics only - no `TensorMap` allocation happens here; see [`materialize`](@ref) for
turning the result into an actual `MPOperator`.
"""
function build_automaton(
    terms::Vector{AutomatonTerm},
    L::Int,
    physical_spaces::Union{
        TensorKit.ElementarySpace,AbstractVector{<:TensorKit.ElementarySpace}
    },
)
    spaces = physical_spaces isa AbstractVector ? physical_spaces : fill(physical_spaces, L)
    length(spaces) == L ||
        throw(ArgumentError("physical_spaces must have length L=$L, got $(length(spaces))"))

    start = PenroseLabel(:start)
    accept = PenroseLabel(:accept)
    term_state = [PenroseLabel(:term, j) for j in eachindex(terms)]

    term_ops = Vector{Dict{Int,QProcess}}(undef, length(terms))
    spans = Vector{Tuple{Int,Int}}(undef, length(terms))
    for (j, t) in enumerate(terms)
        d = Dict(site => op for (site, op) in t.ops)
        term_ops[j] = d
        sites_j = sort(collect(keys(d)))
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

    transitions = [
        Dict{Tuple{PenroseLabel,PenroseLabel},Pair{Number,QProcess}}() for _ in 1:L
    ]
    identity_at(site) = identity_process(TIx(spaces[site], PhysicalLeg()))

    for i in 1:L
        _add_transition!(transitions[i], (start, start), 1, identity_at(i))
        _add_transition!(transitions[i], (accept, accept), 1, identity_at(i))

        for j in eachindex(terms)
            f, l = spans[j]
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
                    (start, term_state[j]),
                    terms[j].coefficient,
                    term_ops[j][i],
                )
            elseif i == l
                _add_transition!(transitions[i], (term_state[j], accept), 1, term_ops[j][i])
            else
                op = get(term_ops[j], i, nothing)
                op = isnothing(op) ? identity_at(i) : op
                _add_transition!(transitions[i], (term_state[j], term_state[j]), 1, op)
            end
        end
    end

    return HamiltonianAutomaton(states, transitions, terms)
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
