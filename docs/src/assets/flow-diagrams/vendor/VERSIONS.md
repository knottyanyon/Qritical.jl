# Vendored dependencies

| File | Library | Version | Source | SHA-256 | Date fetched |
|---|---|---|---|---|---|
| dagre.min.js | @dagrejs/dagre | v2.0.0 | https://github.com/dagrejs/dagre/releases/tag/v2.0.0 (dist/dagre.min.js) | (see VERSIONS.sha256) | 2026-08-24 |
| rough.min.js | rough.js | v3.1.0 | https://github.com/rough-stuff/rough/releases/tag/v3.1.0 (dist/rough.umd.js) | (see VERSIONS.sha256) | 2026-08-24 |
| katex/katex.mjs, katex/katex.css, katex/fonts/* | KaTeX | v0.18.4 | https://github.com/KaTeX/KaTeX/releases/tag/v0.18.4 (katex.zip) | (see VERSIONS.sha256) | 2026-08-24 |
| mathjax/tex-svg.js | MathJax | 4.1.3 | https://github.com/mathjax/MathJax/releases/tag/4.1.3 | (see VERSIONS.sha256) | 2026-08-24 |

## Updating a dependency

1. Download the new release from the source above.
2. Compute `shasum -a 256 <file>` for every changed file.
3. Diff the new file against the old one (or review the upstream changelog).
4. Update this table's version/date columns and regenerate `VERSIONS.sha256` (Step 8 below).
5. Commit both files together in one reviewable commit.
