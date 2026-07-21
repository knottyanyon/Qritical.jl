# Installation

```@raw html
<details class="page-info-drawer">
  <summary>
    <span class="pi-toggle-label">Page info</span>
    <span class="pi-pills">
      <span class="status-pill status-draft">draft</span>
      <span class="status-pill status-needs-rewrite">needs rewrite</span>
      <span class="status-pill status-needs-proofreading">needs proofreading</span>
    </span>
  </summary>
  <div class="page-info-body">
    <table class="page-info-table"><tbody>
      <tr><td class="pi-key">status</td><td class="pi-val"><span class="status-pill status-draft">draft</span> <span class="status-pill status-needs-rewrite">needs rewrite</span> <span class="status-pill status-needs-proofreading">needs proofreading</span></td></tr>
      <tr><td class="pi-key">last updated</td><td class="pi-val">2026-06-03</td></tr>
      <tr><td class="pi-key">written by</td><td class="pi-val">Bavithra Govintharajah</td></tr>
      <tr><td class="pi-key">edited by</td><td class="pi-val">Claude Sonnet 4.6 — initial draft</td></tr>
    </tbody></table>
  </div>
</details>
```

Qritical.jl is not yet registered in the General registry. Install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/knottyanyon/Qritical.jl")
```

Or, inside the Julia REPL package manager (press `]`):

```
pkg> add https://github.com/knottyanyon/Qritical.jl
```

## Requirements

- Julia ≥ 1.11
- Dependencies are resolved automatically by `Pkg`

## Verify

```julia
using Qritical
upper(:σ, 2)   # TIx{Upper}(:σ, 2)
```

## Development install

To work on Qritical itself, clone the repository and activate the local environment:

```bash
git clone https://github.com/knottyanyon/Qritical.jl
cd Qritical.jl
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Run the test suite:

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```
