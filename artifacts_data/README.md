# artifacts_data/

- Staging area for content distributed via Julia's `Artifacts.jl`, kept out of the package's
  git tree/tarball and instead downloaded on demand by `Pkg.instantiate`.
- `tutorial_data/` — the artifact's actual payload (tarred as-is; its contents become the
  directory returned by `tutorial_data_dir()` in [`src/utils/io.jl`](../src/utils/io.jl)):
  - `A.txt` — delimited numeric matrix, loaded via `load_array`
  - `psi.jls`, `psi1.jls`, `psi2.jls` — serialized state-vector/wavefunction fixtures
  - `Bahkauv.png` — figure/diagram asset
- `tutorial_data.tar.gz` — the packaged tarball for the `tutorial_data` artifact, generated
  from `tutorial_data/`; sha256-hashed and attached as a GitHub release asset so
  `Artifacts.toml` can reference it as a download.
- Hosted as GitHub release tag `tutorial_data-v1`, asset `tutorial_data.tar.gz`. Naming
  convention: `<artifact-name>-v<n>`, so each artifact directory here gets its own
  independent version lineage (e.g. a future `benchmark_states/` would live under its own
  `benchmark_states-v1`, `benchmark_states-v2`, ... tags, never sharing a tag with
  `tutorial_data`).
- These files currently duplicate the still-tracked `data/` directory; once tutorials are
  repointed at `tutorial_data_dir()` and confirmed working, `data/` will be removed.
- **Updating an artifact's contents:** edit files under `<name>/`, rerun the
  `create_artifact`/`bind_artifact!(...; force=true)`/`archive_artifact` steps to get a new
  hash and tarball, upload it under a new tag (`<name>-v<n+1>`), then update `Artifacts.toml`'s
  download entry. Old tags/tarballs should be left in place so package versions pinned to the
  old hash keep resolving.
