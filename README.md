<p align="center">
  <img src="docs/src/assets/banner.svg" alt="Qritical.jl" width="600"/>
</p>

<p align="center">
  <a href="https://knottyanyon.github.io/Qritical.jl/stable/"><img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Stable"/></a>
  <a href="https://knottyanyon.github.io/Qritical.jl/dev/"><img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Dev"/></a>
  <a href="https://github.com/knottyanyon/Qritical.jl/actions/workflows/CI.yml?query=branch%3Amain"><img src="https://github.com/knottyanyon/Qritical.jl/actions/workflows/CI.yml/badge.svg?branch=main" alt="Build Status"/></a>
  <a href="https://codecov.io/gh/knottyanyon/Qritical.jl"><img src="https://codecov.io/gh/knottyanyon/Qritical.jl/branch/main/graph/badge.svg" alt="Coverage"/></a>
  <a href="https://github.com/invenia/BlueStyle"><img src="https://img.shields.io/badge/code%20style-blue-4495d1.svg" alt="Code Style: Blue"/></a>
  <a href="https://github.com/JuliaTesting/Aqua.jl"><img src="https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg" alt="Aqua"/></a>
</p>

---

## Running tests

```
julia --project -e "using Pkg; Pkg.test()"
```

## Building documentation
To build and view the documentation locally during development

1. From the project root run:
   - only to build without serving: `julia --project=docs docs/make.jl`
   - to build and serve with live-reloading: `julia --project=docs -e 'using LiveServer; servedocs()'`

2. Then open `docs/build/index.html` in a browser