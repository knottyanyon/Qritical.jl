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

# Patch safe.directory into the bundled git's own system config so it can
# operate inside GitHub Actions containers where the workspace is owned by a
# different UID than the runner process.
RUN for git_bin in /julia-depot/artifacts/*/bin/git; do \
    git_etc="$(dirname $(dirname "$git_bin"))/etc"; \
    mkdir -p "$git_etc"; \
    printf '[safe]\n\tdirectory = *\n' >> "$git_etc/gitconfig"; \
done
