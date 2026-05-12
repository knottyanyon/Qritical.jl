FROM julia:1

# Fixed depot path — avoids $HOME ambiguity inside GitHub Actions containers
ENV JULIA_DEPOT_PATH=/julia-depot

WORKDIR /build

# Copy lockfiles only (triggers a layer rebuild only when deps change)
COPY Project.toml Manifest.toml ./
COPY docs/Project.toml ./docs/

# Minimal stub so Pkg can resolve Qritical as a local dev dep without the full source.
# The real source is checked out at CI time and only Qritical itself recompiles.
RUN mkdir -p src && printf 'module Qritical\nend\n' > src/Qritical.jl

# TODO: precompile all registry packages in the docs environment
# Write the Julia command(s) here that instantiate and precompile the docs env.
# Hint: you need Pkg.instantiate() to resolve the manifest, then Pkg.precompile()
# to compile every package into /julia-depot. Consider: should you also precompile
# the main project env, or only the docs env?
RUN julia --project=docs -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
