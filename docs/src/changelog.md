# Changelog

All notable changes to Qritical.jl are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased] — v0.6.2

### Added

- Math environment blocks (`theorem`, `lemma`, `definition`, `proof`) rendered as styled
  admonition-like boxes, with JavaScript auto-numbering, cross-page anchor links, and
  foldable nested proofs.
- Author-year bracket citation style (`[Author, Year]`) matching physics convention,
  implemented as a custom `DocumenterCitations` style.
- Kitchen Sink developer page (`dev/kitchen_sink.md`) showcasing every available
  documentation component in one place.
- Brand-coloured gallery cards on the Tutorials landing page, one per topic group.
- GitHub edit links and repository button wired into the Documenter navbar.
- Version number shown in the sidebar via `pkgversion`.
- Favicon added to docs and README header polished.
- Build-time interactive widget system (`docs/interactives/`): Julia computes heavy
  numerics at doc-build time and serialises results into self-contained HTML iframes,
  so readers get live interactivity with no server or Pluto runtime.
- **Tutorial 01** — SVD image compression rank explorer: rank slider drives browser-side
  rank-*k* reconstruction via outer-product accumulation of pre-serialised singular triplets.
- **Tutorial 08** — TEBD Néel quench explorer: D and dt sliders update a live S(t) line
  chart and a ⟨S_z⟩ space–time heatmap, all simulation data pre-computed and embedded as JSON.

### Changed

- Tutorials reorganised from a flat exercise list into four topic groups:
  **Part 1 – Fundamentals**, **Part 2 – Tensor Trains**, **Part 3 – Dynamics**,
  **Part 4 – Misc**, each with its own gallery-card overview page.
- `src/` reorganised into thematic subdirectories (v0.6.2 internal refactor).
- Logo updated with a refined design; old nabla draft logo retired.
- Bibliography labels rendered bold-italic in the reference list.
- Docs site styled with brand colour palette, breadcrumb navigation, and a right-side
  page table-of-contents panel.

### Fixed

- Favicon declared with the correct `:ico` asset class so it renders in browsers
  (closes [#133](https://github.com/knottyanyon/Qritical.jl/issues/133)).
- Missing `Random` import in Tutorial 10 caused doc build failures
  (closes [#131](https://github.com/knottyanyon/Qritical.jl/issues/131)).
- RNG seed and assertion tolerance tightened in Tutorial 10 imaginary-time evolution
  example to ensure reproducible doc builds.

### Internal

- Removed unused Docker infrastructure from the repository.
- Added Codecov test-coverage upload to CI.
- Tightened Dependabot config: grouped dependency bumps, switched to monthly Julia
  update schedule.

---

## [v0.6.1-alpha] — 2026-07-15

First tagged release. Establishes the full documentation site and the Makie drawing
extension alongside the core tensor-network library.

### Added

**Tensor-network drawing (Makie extension)**

- `QriticalMakieExt`: optional Makie package extension for schematic tensor-network
  diagrams.
- `draw(::FiniteMPS)` renders a full MPS chain with bond wires, site tensors, and
  physical legs.
- `draw(::QTensor)` renders a single tensor with variance-coloured leg stubs
  (covariant = blue, contravariant = red).
- Tensors drawn by kind: general (square), diagonal (diamond), unitary (circle),
  isometry (trapezoid).
- Directed arrowheads on bond wires; explicit axis limits.
- LaTeX labels on legs and bonds; Julia brand colour palette.
- Drawing DSL verbs: `node!`, `wire!`, `stub!`, `region!`.

**Documentation site**

- Getting Started tutorial **GS-1**: Tensors & Indices (live-executed Literate notebook).
- Getting Started tutorial **GS-2**: SVD & Truncation (live-executed Literate notebook).
- Getting Started tutorial **GS-3**: Drawing Tensor Networks, including a TEBD
  brick-wall circuit example.
- Exercises 01–11 converted to Literate.jl notebooks; exercises 01–10 run live during
  docs build (exercise 11 rendered as static Julia code blocks).
- Exercise symbol-discovery system: each API docstring receives an auto-generated
  "Used in tutorials" section linking to the exercises where that symbol appears.
- Comprehensive glossary (55+ terms) with hover-tooltip previews and automatic
  `[`term`](/references/glossary/#term)` → link preprocessing across all Markdown source files.
- Mermaid diagram rendering via `DocumenterMermaid.jl`.
- Foldable output cells for long code-execution output.
- Visualisation API reference page.
- Quick Reference sections with anchor links on every API page.
- Full API reference reorganised into structured Types / Functions sections across
  18 topic pages.
- SHA-256 stamp-based stale checking for Literate source files; `DOCS_FAST=1` env
  variable skips tutorial execution for fast iteration on prose or API pages.
- Inline base64 figures extracted to separate image files, reducing
  `search_index.js` from >2 MB to <200 KB.

### Fixed

- Broken `@ref` cross-references across all API and exercise pages.
- `LiveServer` / `servedocs()` infinite rebuild loop.
- Plot save path in Exercise 11 anchored to the exercise directory.
- Missing docstrings that caused `makedocs` warnings on the Index Layer page.
- `@example` block execution warnings eliminated across all exercise files.
