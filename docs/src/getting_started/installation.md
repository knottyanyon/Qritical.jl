# Installation

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
