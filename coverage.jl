#!/usr/bin/env julia
# Run from repo root:  julia --project=. coverage.jl
#
# 1. Runs the test suite with coverage instrumentation.
# 2. Prints a per-file summary to the terminal.
# 3. Writes lcov.info — consumed by the VS Code "Coverage Gutters" extension.
# 4. Cleans up .cov files.

import Pkg
Pkg.activate(@__DIR__)

using Coverage
using Printf

const ROOT    = @__DIR__
const SRC     = joinpath(ROOT, "src")
const LCOV    = joinpath(ROOT, "lcov.info")
const RUNTESTS = joinpath(ROOT, "test", "runtests.jl")

# ── 1. Run tests with coverage instrumentation ────────────────────────────────
println("Running tests with --code-coverage=user …")
run(ignorestatus(`$(Base.julia_cmd()) --project=$ROOT --code-coverage=user $RUNTESTS`))

# ── 2. Process .cov files ─────────────────────────────────────────────────────
fc = process_folder(SRC)

# ── 3. Print per-file summary ─────────────────────────────────────────────────
println()
println("── Coverage ─────────────────────────────────────────────────────────")
for f in sort(fc; by=x -> x.filename)
    covered = count(c -> c !== nothing && c > 0, f.coverage)
    tracked = count(c -> c !== nothing, f.coverage)
    tracked == 0 && continue
    pct = 100 * covered / tracked
    bar = pct >= 90 ? "✓" : pct >= 70 ? "~" : "✗"
    rel = relpath(f.filename, ROOT)
    @printf("  %s %5.1f%%  %s  (%d/%d)\n", bar, pct, rel, covered, tracked)
end
c, t = get_summary(fc)
println("─────────────────────────────────────────────────────────────────────")
@printf("  Total: %.1f%%  (%d/%d lines covered)\n", 100 * c / t, c, t)

# ── 4. Write lcov.info ────────────────────────────────────────────────────────
Coverage.LCOV.writefile(LCOV, fc)
println("\nlcov.info written — install 'Coverage Gutters' in VS Code to see inline highlights.")

# ── 5. Clean up .cov sidecar files ───────────────────────────────────────────
clean_folder(SRC)
