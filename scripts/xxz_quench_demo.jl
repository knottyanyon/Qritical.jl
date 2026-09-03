#=
XXZ quench demo: finite TEBD evolving a Néel state and a domain-wall state under the open-boundary
XXZ chain (J = Jxy = 0.7, Jz = 1.0, h = π/10), monitoring per-site ⟨Ŝᶻ⟩ and the bipartite
entanglement entropy at the chain's moving orthogonality center at every Trotter step.

Run: julia --project=. scripts/xxz_quench_demo.jl
=#

using Qritical
using TensorKit

# SECTION -  parameters (adjust here)

const L = 8
const V = ComplexSpace(2)
const Jxy = 0.7
const Jz = 1.0
const h = π / 10
const dt = 0.05
const num_steps = 40
const bond_cutoff = 32

# SECTION -  a small collector that accumulates every TEBDStepSnapshot into a Vector

struct SnapshotLog <: Qritical.AbstractCollector
    seen::Vector{Any}
end
SnapshotLog() = SnapshotLog(Any[])
function Qritical.step!(::Qritical.Active, c::SnapshotLog, ctx::NamedTuple)
    return push!(c.seen, ctx.snapshot)
end
Qritical.finalize!(::Qritical.Active, c::SnapshotLog) = c.seen

# SECTION -  run one quench and print a per-step table

function run_quench(name::String, ψ0)
    H = xxz_hamiltonian(L, V; Jxy=Jxy, Jz=Jz, h=h)

    sz = QProcess(
        TensorMap(ComplexF64[1 0; 0 -1], V ← V);
        output_roles=PhysicalLeg(),
        input_roles=PhysicalLeg(),
    )
    observables = Dict(i => LocalObservable(i, sz) for i in 1:L)

    collector = SnapshotLog()
    algorithm = TEBDAlgorithm(SuzukiTrotter(), bond_cutoff, num_steps; snapshot_every=1)
    evolve!(algorithm, H, ψ0, dt; observables=observables, collector=collector)

    println("\n=== $name ===")
    println("step | t     | entanglement_entropy | ⟨Sz_i⟩ (i=1..$L)")
    for snap in collector.seen
        t = snap.step * dt
        szs = [round(real(value(snap.observables[i])); digits=3) for i in 1:L]
        println(
            rpad(snap.step, 5),
            "| ",
            rpad(round(t; digits=3), 6),
            "| ",
            rpad(round(snap.entanglement_entropy; digits=4), 22),
            "| ",
            szs,
        )
    end
    return collector
end

function main()
    println(
        "XXZ quench demo: L=$L, Jxy=$Jxy, Jz=$Jz, h=$(round(h; digits=4)), dt=$dt, num_steps=$num_steps, bond_cutoff=$bond_cutoff",
    )

    run_quench("Néel state |↑↓↑↓...⟩", neel_state(L, V))
    run_quench("Domain-wall state |↑↑↑↑↓↓↓↓⟩", domain_wall_state(L, V))

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
