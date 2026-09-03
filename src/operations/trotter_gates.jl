#=META
source:
  author: Bavithra
  reviewer:
docstrings:
  author: Claude Sonnet 5
  coauthor: 
  reviewer:
refs: schollwoeck_2011
credits: N/A
=#

# Turns a Propagator's Hamiltonian into the actual exponentiated gate QProcess operators a TEBD
# sweep applies - the first place in the codebase that calls TensorKit.exp. Per Schollwoeck's
# TEBD construction: within one split_commuting_groups group, terms act on disjoint sites and so
# commute, meaning exp(Σ terms) = Π exp(term) exactly - each AutomatonTerm becomes its own small
# gate rather than combining a whole group into one tensor first. TEBD itself is defined only for
# nearest-neighbor (1- or 2-site contiguous) terms; non-contiguous terms raise a clear error
# rather than attempting general identity-embedding, which is outside what TEBD is.

# SECTION -  TrotterGateBlock / TrotterStep - exponentiated content vs. per-run binding

"""
    TrotterGateBlock{PF<:ProductFormula}

The already-exponentiated gates for one full application of the product formula `PF` - no `dt`
field: `dt` is consumed to *compute* each gate's exponent inside [`trotterize`](@ref) but is not
kept as redundant metadata here, since it is already baked into the gate tensors' numeric values.
`PF` is carried as a type parameter, not a field - `ProductFormula` subtypes ([`LieTrotter`](@ref),
[`SuzukiTrotter`](@ref), [`Suzuki4th`](@ref)) are stateless singletons, so storing an instance
would just duplicate what the type already encodes; recover it with `PF()` wherever a `pf`
instance is needed (e.g. calling back into `Subroutines.sequence`/`Subroutines.local_error_bound`).

# Fields

  - `gates :: Vector{Pair{UnitRange{Int},QProcess}}` - `(site_range, gate)` pairs, in application
    order (step order, then term order within each step). `site_range` is `s:s` for a 1-site gate
    or `s1:s2` for a 2-site gate, mirroring `AutomatonTerm.ops::Vector{Pair{Int,QProcess}}`'s own
    site+operator bundling convention - not a parallel `sites` array, which would risk silent
    index desync.
"""
struct TrotterGateBlock{PF<:ProductFormula}
    gates::Vector{Pair{UnitRange{Int},QProcess}}
end

"""
    TrotterStep{PF<:ProductFormula}

A [`TrotterGateBlock`](@ref) bound to the `dt`/repeat-count a driver needs to actually apply it:
the unit a future `TEBDAlgorithm` driver consumes and kicks off - "apply `block.gates`, in order,
`num_steps` times." Building `TrotterStep`s, contracting gates into an MPS, taking snapshots, and
accumulating error are all driver-loop concerns, not this file's - see [`trotter_error`](@ref)/
[`record_trotter_error!`](@ref) for the documented, opt-in plug-in point a driver uses for the
latter.

# Fields

  - `block      :: TrotterGateBlock{PF}` - the exponentiated gate content.
  - `dt         :: Float64`             - the (real- or imaginary-)time step the block was built at.
  - `num_steps  :: Int`                 - how many times a driver repeats this block to advance the
    simulation by `num_steps * dt`; maps directly onto
    `Subroutines.accumulate_trotter_error!`'s `num_steps` argument.
"""
struct TrotterStep{PF<:ProductFormula}
    block::TrotterGateBlock{PF}
    dt::Float64
    num_steps::Int
end

# SECTION -  _combine_term - one AutomatonTerm's ops -> one gate tensor

# Combine an AutomatonTerm's (site, operator) pairs into a single QProcess tensor ready to
# exponentiate. Requires contiguous sites (1 or 2 touched sites) - TEBD is fundamentally a
# nearest-neighbor-bond gate sweep (Schollwoeck sec. 7); non-contiguous/long-range terms are
# outside what TEBD is, not merely unimplemented, so this errors rather than attempting general
# identity-embedding across gap sites.
function _combine_term(term::AutomatonTerm)
    sorted = sort(term.ops; by=first)
    n = length(sorted)
    site_range = sorted[1].first:sorted[end].first
    if n == 1
        return site_range, sorted[1].second
    elseif n == 2
        s1, s2 = sorted[1].first, sorted[2].first
        s2 == s1 + 1 || error(
            "trotterize: AutomatonTerm touches non-adjacent sites $s1 and $s2 - TEBD gates " *
            "require contiguous (nearest-neighbor) terms; non-contiguous/long-range terms " *
            "fall outside what TEBD is (Schollwoeck's TEBD construction is defined only for " *
            "nearest-neighbor bond terms).",
        )
        return site_range, sorted[1].second ⊗ sorted[2].second
    else
        error(
            "trotterize: AutomatonTerm touches $n sites ($(first.(sorted))) - TEBD gates only " *
            "support 1- or 2-site (contiguous) terms.",
        )
    end
end

# SECTION -  trotterize - the missing link: Propagator -> exponentiated gate sequence

function _trotter_exponent(::Type{RealTime}, coeff, dt, term_coefficient)
    return -im * coeff * dt * term_coefficient
end
function _trotter_exponent(::Type{ImaginaryTime}, coeff, dt, term_coefficient)
    return -coeff * dt * term_coefficient
end

"""
    trotterize(propagator::Propagator{T}, pf::PF; num_steps::Int=1) where {T<:Time,PF<:ProductFormula}
        -> TrotterStep{PF}

Apply the product-formula splitting tagged by `pf` to `propagator`'s Hamiltonian, producing the
actual sequence of exponentiated gate `QProcess` operators a TEBD sweep applies - the missing link
between `Subroutines.sequence`'s opaque `(term, coefficient)` recipe and real tensors.

Works identically for `Propagator{RealTime}`/`Propagator{ImaginaryTime}`: grouping and `sequence`'s
ordering/coefficients are completely agnostic to `Time` kind, which only enters when computing
each gate's exponent (`-im*dt*...` vs `-dt*...`).

# Construction

 1. `groups = Subroutines.split_commuting_groups(propagator.hamiltonian.terms)` (default
    `EvenOddSplit`).
 2. `steps = Subroutines.sequence(pf, groups, propagator.dt)` - each *group* is one atomic "term"
    to the product formula.
 3. For each `(group, coeff)` step, for each `AutomatonTerm` in the group: combine its ops into one
    tensor (see [`TrotterGateBlock`](@ref)'s module note on contiguity), then
    `QProcess(TensorKit.exp(exponent * tensor), ...)`.
 4. Collect the gates into a [`TrotterGateBlock`](@ref), wrapped as a [`TrotterStep`](@ref) with
    `propagator.dt` and `num_steps`.

No error accumulation happens inside `trotterize` - see [`trotter_error`](@ref)/
[`record_trotter_error!`](@ref) for the separate, opt-in extension point.
"""
function trotterize(
    propagator::Propagator{T}, pf::PF; num_steps::Int=1
) where {T<:Time,PF<:ProductFormula}
    hamiltonian = propagator.hamiltonian
    dt = propagator.dt
    groups = split_commuting_groups(hamiltonian.terms)
    steps = sequence(pf, groups, dt)

    gates = Pair{UnitRange{Int},QProcess}[]
    for (group, coeff) in steps
        for term in group
            site_range, combined = _combine_term(term)
            exponent = _trotter_exponent(T, coeff, dt, term.coefficient)
            gate_tensor = exp(exponent * tensor(combined))
            push!(
                gates,
                site_range => QProcess(gate_tensor, outputs(combined), inputs(combined)),
            )
        end
    end

    return TrotterStep(TrotterGateBlock{PF}(gates), dt, num_steps)
end

# SECTION -  trotter_error / record_trotter_error! - opt-in error tracking

"""
    trotter_error(step::TrotterStep{PF}, hamiltonian::Hamiltonian, norm) where {PF} -> Float64

Single-shot local Trotter error bound for `step`, reusing `Subroutines.local_error_bound`. Not
called inside [`trotterize`](@ref) - a driver plugs this in explicitly after executing a step,
alongside snapshotting/observable evaluation. Recomputes `split_commuting_groups(hamiltonian.terms)`
itself (cheap relative to the exponentiations already paid for in `trotterize`) rather than taking
a separate `terms`/`groups` argument, so the bound is guaranteed to mirror exactly what `trotterize`
grouped internally - no argument for a caller to accidentally desync from `trotterize`'s own
grouping.

# Arguments

$(Glossaries.Argument{@__MODULE__}()([:trotter_hamiltonian, :trotter_norm]))
"""
function trotter_error(step::TrotterStep{PF}, hamiltonian::Hamiltonian, norm) where {PF}
    return local_error_bound(PF(), split_commuting_groups(hamiltonian.terms), step.dt, norm)
end

"""
    record_trotter_error!(acc::TrotterErrorAccumulator, step::TrotterStep{PF},
                           hamiltonian::Hamiltonian, norm) where {PF} -> Float64

Record `step`'s Trotter error into `acc` across `step.num_steps` repetitions, reusing
`Subroutines.accumulate_trotter_error!`. See [`trotter_error`](@ref) for why `hamiltonian` is
taken instead of a `terms`/`groups` argument.

# Arguments

$(Glossaries.Argument{@__MODULE__}()([:trotter_hamiltonian, :trotter_norm]))
"""
function record_trotter_error!(
    acc::TrotterErrorAccumulator, step::TrotterStep{PF}, hamiltonian::Hamiltonian, norm
) where {PF}
    return accumulate_trotter_error!(
        acc, PF(), split_commuting_groups(hamiltonian.terms), step.dt, step.num_steps, norm
    )
end

# SECTION -  JIT precompilation workload

PrecompileTools.@compile_workload begin
    V = TensorKit.ComplexSpace(2)
    sz = QProcess(
        TensorKit.TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    terms = [AutomatonTerm(-1.0, [1 => sz, 2 => sz])]
    H = Hamiltonian(terms, 2, V)
    for kind in (RealTime, ImaginaryTime), pf in (LieTrotter(), SuzukiTrotter())
        p = propagator(H, 0.1; kind=kind)
        trotterize(p, pf)
    end
end
