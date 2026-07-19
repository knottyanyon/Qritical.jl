# Page Metadata Drawer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a collapsible metadata drawer under each page title showing coloured status pills (draft / needs-rewrite / needs-proofreading / needs-touchups / needs-theory / proofread) plus optional attribution fields (last updated, written by, edited by), and a reader-facing disclaimer on the index page.

**Architecture:** A Julia preprocessor (`docs/page_meta_preprocessor.jl`) converts a `{page-info\n...\n}` shorthand block in any `.md` source file into a `@raw html` `<details>` element before `makedocs` runs — the same pipeline as the existing `glossary_linker.jl`. Status pill styles and drawer layout live in `docs/src/assets/custom.css`. No JS required: `<details>/<summary>` handles fold/unfold natively.

**Tech Stack:** Julia (regex, string manipulation), plain CSS (color-mix, flexbox), HTML5 `<details>/<summary>`, Documenter.jl `@raw html` blocks.

## Global Constraints

- All CSS appended to `docs/src/assets/custom.css` — no new CSS files.
- Follow the exact pattern of `glossary_linker.jl`: a `preprocessor(content)` function + a `preprocess_directory(src_dir)` wrapper; call the wrapper in `make.jl` just before `makedocs`.
- Colors: use only the Julia brand palette already in the CSS: red `#CB3C33`, orange `#D4680A`, amber `#B89010`, purple `#9558B2`, green `#389826`, slate `#9aa3ad`. Use `color-mix(in srgb, ...)` for tints (already used in `.tutorial-card`).
- Dark mode: target `html.theme--documenter-dark` (existing pattern throughout `custom.css`).
- Monospace font in the drawer body: `"SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace` — no external font load.
- Do **not** modify `.jl` tutorial source files in this plan; applying tags to individual pages is editorial work done after the infrastructure is in place.
- Test by running `DOCS_FAST=1 julia --project=docs docs/make.jl` and inspecting `docs/build/`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| **Create** | `docs/page_meta_preprocessor.jl` | Parse `{page-info}` blocks; emit `@raw html` drawer HTML |
| **Modify** | `docs/src/assets/custom.css` | Status pill styles + drawer layout (light + dark) |
| **Modify** | `docs/make.jl` | `include` preprocessor; call `preprocess_page_meta` before `makedocs` |
| **Modify** | `docs/src/index.md` | Add documentation-status disclaimer section |

---

## Task 1: CSS — status pills and drawer layout

**Files:**
- Modify: `docs/src/assets/custom.css` (append at end of file)

**Interfaces:**
- Produces: CSS classes consumed by the HTML the preprocessor emits in Task 2:
  - `.status-pill` (base)
  - `.status-draft`, `.status-needs-rewrite`, `.status-needs-proofreading`, `.status-needs-touchups`, `.status-needs-theory`, `.status-proofread`
  - `.page-info-drawer`, `.pi-toggle-label`, `.pi-pills`, `.page-info-body`, `.page-info-table`, `.pi-key`

- [ ] **Step 1: Append pill and drawer CSS to `docs/src/assets/custom.css`**

Add the following block at the very end of the file:

```css
/* ================================================================
   Page-info metadata drawer
   Collapsible <details> element placed after the page H1.
   Status pills appear in the always-visible <summary> line.

   Authoring: place a {page-info\n...\n} block in any .md source
   immediately after the # Title line; page_meta_preprocessor.jl
   converts it to the <details> HTML before makedocs runs.

   Status slugs and their colors (Julia brand palette):
     draft              — slate   #9aa3ad
     needs-rewrite      — red     #CB3C33
     needs-proofreading — orange  #D4680A
     needs-touchups     — amber   #B89010
     needs-theory       — purple  #9558B2
     proofread          — green   #389826
   ================================================================ */

/* ── Status pills ──────────────────────────────────────────────── */
.status-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.13em 0.55em;
    border-radius: 9999px;
    font-size: 0.685rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    white-space: nowrap;
    border: 1px solid transparent;
    line-height: 1.6;
}

/* Light mode */
.status-draft {
    background: color-mix(in srgb, #9aa3ad 14%, white);
    color: #4a5260;
    border-color: color-mix(in srgb, #9aa3ad 30%, white);
}
.status-needs-rewrite {
    background: color-mix(in srgb, #CB3C33 11%, white);
    color: #a02820;
    border-color: color-mix(in srgb, #CB3C33 26%, white);
}
.status-needs-proofreading {
    background: color-mix(in srgb, #D4680A 11%, white);
    color: #9c4c06;
    border-color: color-mix(in srgb, #D4680A 26%, white);
}
.status-needs-touchups {
    background: color-mix(in srgb, #B89010 11%, white);
    color: #7a5e00;
    border-color: color-mix(in srgb, #B89010 26%, white);
}
.status-needs-theory {
    background: color-mix(in srgb, #9558B2 11%, white);
    color: #6a3a8a;
    border-color: color-mix(in srgb, #9558B2 26%, white);
}
.status-proofread {
    background: color-mix(in srgb, #389826 11%, white);
    color: #256818;
    border-color: color-mix(in srgb, #389826 26%, white);
}

/* Dark mode */
html.theme--documenter-dark .status-draft {
    background: color-mix(in srgb, #9aa3ad 18%, #13131a);
    color: #9aa3ad;
    border-color: color-mix(in srgb, #9aa3ad 34%, #13131a);
}
html.theme--documenter-dark .status-needs-rewrite {
    background: color-mix(in srgb, #CB3C33 18%, #13131a);
    color: #e07878;
    border-color: color-mix(in srgb, #CB3C33 36%, #13131a);
}
html.theme--documenter-dark .status-needs-proofreading {
    background: color-mix(in srgb, #D4680A 18%, #13131a);
    color: #e89050;
    border-color: color-mix(in srgb, #D4680A 36%, #13131a);
}
html.theme--documenter-dark .status-needs-touchups {
    background: color-mix(in srgb, #B89010 18%, #13131a);
    color: #d4b030;
    border-color: color-mix(in srgb, #B89010 36%, #13131a);
}
html.theme--documenter-dark .status-needs-theory {
    background: color-mix(in srgb, #9558B2 18%, #13131a);
    color: #c080e0;
    border-color: color-mix(in srgb, #9558B2 36%, #13131a);
}
html.theme--documenter-dark .status-proofread {
    background: color-mix(in srgb, #389826 18%, #13131a);
    color: #65c553;
    border-color: color-mix(in srgb, #389826 36%, #13131a);
}

/* ── Drawer shell ──────────────────────────────────────────────── */
.page-info-drawer {
    margin: 0.45rem 0 1.6rem;
    border: 1px solid #e1e1e6;
    border-radius: 6px;
    overflow: hidden;
    font-size: 0.8rem;
}

.page-info-drawer > summary {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.38rem 0.8rem;
    cursor: pointer;
    user-select: none;
    list-style: none;
    background: #f5f5f7;
}

.page-info-drawer > summary::-webkit-details-marker { display: none; }

.page-info-drawer > summary::before {
    content: "▶";
    font-size: 0.52rem;
    color: #9aa3ad;
    flex-shrink: 0;
    margin-top: 0.05em;
}

.page-info-drawer[open] > summary::before { content: "▼"; }

.pi-toggle-label {
    font-size: 0.66rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6c757d;
    white-space: nowrap;
}

.pi-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.28rem;
    align-items: center;
}

/* ── Expanded body ─────────────────────────────────────────────── */
.page-info-body {
    padding: 0.55rem 0.9rem 0.7rem;
    border-top: 1px solid #e1e1e6;
}

.page-info-table {
    border-collapse: collapse;
    font-family: "SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace;
    font-size: 0.775rem;
    line-height: 1.65;
}

.page-info-table td { padding: 0.05rem 0; vertical-align: top; }

.pi-key {
    color: #6c757d;
    white-space: nowrap;
    padding-right: 2rem;
    width: 1%;
}

/* ── Dark mode ─────────────────────────────────────────────────── */
html.theme--documenter-dark .page-info-drawer       { border-color: #34353d; }
html.theme--documenter-dark .page-info-drawer > summary { background: #1c1d22; }
html.theme--documenter-dark .page-info-drawer > summary::before { color: #555d68; }
html.theme--documenter-dark .page-info-body         { border-top-color: #34353d; }
html.theme--documenter-dark .pi-toggle-label        { color: #9aa3ad; }
html.theme--documenter-dark .pi-key                 { color: #9aa3ad; }
```

- [ ] **Step 2: Verify CSS is syntactically valid**

```bash
# Quick parse check — npx is available on most systems
npx --yes csslint --errors=errors docs/src/assets/custom.css 2>&1 | tail -5
# Acceptable: warnings about vendor prefixes or color-mix. Zero "errors".
# If npx unavailable, skip — the build in Task 3 will catch parse failures.
```

- [ ] **Step 3: Commit CSS**

```bash
git add docs/src/assets/custom.css
git commit -m "feat(docs): add status pill and page-info drawer CSS"
```

---

## Task 2: page_meta_preprocessor.jl

**Files:**
- Create: `docs/page_meta_preprocessor.jl`

**Interfaces:**
- Consumes: `.md` files containing `{page-info\n  key: value\n}` blocks (any page)
- Produces:
  - `page_meta_preprocessor(content::String) → String` — single-string transform
  - `preprocess_page_meta(src_dir::String)` — directory walker, called in `make.jl` (Task 3)
- Valid keys: `status` (comma-separated slugs), `updated`, `written`, `edited`
- Valid status slugs: `draft`, `needs-rewrite`, `needs-proofreading`, `needs-touchups`, `needs-theory`, `proofread`

- [ ] **Step 1: Create `docs/page_meta_preprocessor.jl`**

```julia
# page_meta_preprocessor.jl
#
# Converts {page-info\n...\n} shorthand blocks in Documenter .md source files
# to @raw html <details> drawers.
#
# Authoring syntax (place after the # Title line in any .md file):
#
#   {page-info
#     status:  draft, needs-proofreading
#     updated: 2026-07-19
#     written: Bavithra Govintharajah
#     edited:  Claude Sonnet 4.6 — grammar (2026-07-19)
#   }
#
# In Literate .jl tutorial sources, prefix every line with "# ":
#
#   # {page-info
#   #   status: draft
#   # }
#
# Valid status slugs:
#   draft | needs-rewrite | needs-proofreading | needs-touchups | needs-theory | proofread

const PAGE_INFO_PATTERN = r"\{page-info\n(.*?)\n\}"s

const _STATUS_LABELS = Dict(
    "draft"              => "draft",
    "needs-rewrite"      => "needs rewrite",
    "needs-proofreading" => "needs proofreading",
    "needs-touchups"     => "needs touch-ups",
    "needs-theory"       => "needs theory",
    "proofread"          => "proofread",
)

function _parse_page_meta(block::AbstractString)::Dict{String,String}
    meta = Dict{String,String}()
    for line in split(block, '\n')
        m = match(r"^\s*([\w-]+)\s*:\s*(.+)$", line)
        m === nothing && continue
        meta[strip(m[1])] = strip(m[2])
    end
    return meta
end

function _status_pill(slug::AbstractString)::String
    label = get(_STATUS_LABELS, slug, slug)
    return """<span class="status-pill status-$(slug)">$(label)</span>"""
end

function _meta_to_html(meta::Dict{String,String})::String
    # Status pills — rendered in both summary and body
    pills_html = ""
    status_row = ""
    if haskey(meta, "status")
        slugs = filter!(!isempty, strip.(split(meta["status"], ',')))
        pills   = _status_pill.(slugs)
        pills_html = join(pills, "\n        ")
        status_row = "\n      <tr><td class=\"pi-key\">status</td>" *
                     "<td class=\"pi-val\">" * join(pills, " ") * "</td></tr>"
    end

    # Optional ordered metadata rows
    rows = String[status_row]
    for (key, display) in (("updated", "last updated"), ("written", "written by"), ("edited", "edited by"))
        haskey(meta, key) || continue
        push!(rows, "\n      <tr><td class=\"pi-key\">$(display)</td>" *
                    "<td class=\"pi-val\">$(meta[key])</td></tr>")
    end
    body_rows = join(filter(!isempty, rows))

    return """```@raw html
<details class="page-info-drawer">
  <summary>
    <span class="pi-toggle-label">Page info</span>
    <span class="pi-pills">
        $(pills_html)
    </span>
  </summary>
  <div class="page-info-body">
    <table class="page-info-table"><tbody>$(body_rows)
    </tbody></table>
  </div>
</details>
```"""
end

"""
    page_meta_preprocessor(content) -> String

Replace every `{page-info\\n...\\n}` block in `content` with its rendered
`@raw html` drawer equivalent.
"""
function page_meta_preprocessor(content::AbstractString)::String
    return replace(content, PAGE_INFO_PATTERN => function(m)
        cap = match(PAGE_INFO_PATTERN, m)
        cap === nothing && return m
        meta = _parse_page_meta(cap[1])
        return _meta_to_html(meta)
    end)
end

"""
    preprocess_page_meta(src_dir)

Walk `src_dir` recursively, applying `page_meta_preprocessor` to every `.md`
file that contains a `{page-info}` block. Files without a block are untouched.
"""
function preprocess_page_meta(src_dir::AbstractString)
    for (root, _, files) in walkdir(src_dir)
        for file in files
            endswith(file, ".md") || continue
            fpath   = joinpath(root, file)
            content = read(fpath, String)
            updated = page_meta_preprocessor(content)
            updated != content && write(fpath, updated)
        end
    end
end

export page_meta_preprocessor, preprocess_page_meta
```

- [ ] **Step 2: Smoke-test the preprocessor in the Julia REPL**

```julia
# Run from the project root
include("docs/page_meta_preprocessor.jl")

sample = """
# My Page

{page-info
  status: draft, needs-proofreading
  updated: 2026-07-19
  written: Bavithra Govintharajah
  edited:  Claude Sonnet 4.6 — grammar (2026-07-19)
}

First paragraph.
"""

result = page_meta_preprocessor(sample)
println(result)
```

Expected: the `{page-info}` block is replaced with a ` ```@raw html ` block containing `<details class="page-info-drawer">...</details>`. The surrounding text (`# My Page\n\n` and `\n\nFirst paragraph.\n`) is unchanged.

Also verify a file with no `{page-info}` block is returned unchanged:

```julia
plain = "# Title\n\nJust prose.\n"
@assert page_meta_preprocessor(plain) == plain
```

- [ ] **Step 3: Commit preprocessor**

```bash
git add docs/page_meta_preprocessor.jl
git commit -m "feat(docs): add page_meta_preprocessor for page-info drawer blocks"
```

---

## Task 3: Wire preprocessor into make.jl

**Files:**
- Modify: `docs/make.jl`

**Interfaces:**
- Consumes: `preprocess_page_meta` from `docs/page_meta_preprocessor.jl` (Task 2)
- Produces: `preprocess_page_meta` called on `src_dir` before `makedocs`

- [ ] **Step 1: Add `include` at the top of make.jl, after the existing `include` lines**

Find these two lines near the top of `make.jl`:

```julia
include(joinpath(@__DIR__, "symbol_docstring_injector.jl"))
include(joinpath(@__DIR__, "glossary_linker.jl"))
```

Add one line immediately after:

```julia
include(joinpath(@__DIR__, "symbol_docstring_injector.jl"))
include(joinpath(@__DIR__, "glossary_linker.jl"))
include(joinpath(@__DIR__, "page_meta_preprocessor.jl"))
```

- [ ] **Step 2: Call `preprocess_page_meta` just before `makedocs`**

Find these lines near the bottom of `make.jl` (just before `makedocs(...)`):

```julia
src_dir = normpath(joinpath(@__DIR__, "src"))
preprocess_glossary_links(src_dir)
```

Add the new call immediately after:

```julia
src_dir = normpath(joinpath(@__DIR__, "src"))
preprocess_glossary_links(src_dir)
preprocess_page_meta(src_dir)
```

- [ ] **Step 3: Run a fast docs build to confirm the pipeline wires up without errors**

```bash
cd /Users/bavithra/Documents/Uni/Courses/26_HTN/Qritical.jl
DOCS_FAST=1 julia --project=docs docs/make.jl 2>&1 | tail -20
```

Expected: build completes with no `ERROR` lines. Warnings about missing cross-refs are normal. The build output at `docs/build/` should be present.

- [ ] **Step 4: Commit the make.jl wiring**

```bash
git add docs/make.jl
git commit -m "feat(docs): wire page_meta_preprocessor into docs build pipeline"
```

---

## Task 4: Index page disclaimer

**Files:**
- Modify: `docs/src/index.md`

**Interfaces:**
- Consumes: nothing from other tasks (pure Markdown)
- Produces: a `!!! note` admonition on the index page visible to all readers

- [ ] **Step 1: Replace `docs/src/index.md` with the version below**

The existing file is nearly empty. Replace it entirely:

```markdown
```@meta
CurrentModule = Qritical
```

# Qritical

Documentation for [Qritical](https://github.com/knottyanyon/Qritical.jl).

## Documentation status

Pages across this site carry a small **Page info** drawer directly beneath
their title. Click the drawer to see review status, when the page was last
updated, and who wrote or edited it.

!!! note "Status tag guide"
    | Tag | Colour | Meaning |
    |-----|--------|---------|
    | **draft** | gray | Page is still being written; structure or content may be incomplete. |
    | **needs rewrite** | red | Content is substantially wrong or misleading — treat with significant caution. |
    | **needs proofreading** | orange | Content has not been reviewed; read with appropriate skepticism. |
    | **needs touch-ups** | amber | Minor phrasing or formatting improvements are pending; content is broadly correct. |
    | **needs theory** | purple | The physics or mathematical explanation is thin or may contain inaccuracies. |
    | **proofread** | green | The author has read and verified this page. |

    Pages with no drawer have not yet been assessed.
```

- [ ] **Step 2: Rebuild docs and inspect the index page**

```bash
DOCS_FAST=1 julia --project=docs docs/make.jl 2>&1 | grep -E "ERROR|WARNING|makedocs"
```

Then open `docs/build/index.html` in a browser (or run `open docs/build/index.html` on macOS) and confirm:
- The "Documentation status" heading is present.
- The admonition table renders with all six rows.

- [ ] **Step 3: Commit**

```bash
git add docs/src/index.md
git commit -m "docs: add documentation status guide to index page"
```

---

## Task 5: Apply to one page end-to-end (smoke test)

This task proves the full pipeline on a real page — pick `docs/src/getting_started/introduction.md` (a plain `.md` file, not Literate-generated, so no `.jl` source to touch).

**Files:**
- Modify: `docs/src/getting_started/introduction.md`

- [ ] **Step 1: Add a `{page-info}` block to `introduction.md`**

Open `docs/src/getting_started/introduction.md`. After the `# Introduction: Legs, Indices, and the Design Idea` heading, insert:

```markdown
# Introduction: Legs, Indices, and the Design Idea

{page-info
  status: needs-proofreading
  updated: 2026-07-19
  written: Bavithra Govintharajah
  edited:  Claude Sonnet 4.6 — initial draft
}

When you sketch a tensor network on paper...
```

- [ ] **Step 2: Run a fast build and check the generated HTML**

```bash
DOCS_FAST=1 julia --project=docs docs/make.jl 2>&1 | grep -E "ERROR|introduction"
```

Then inspect the generated file:

```bash
grep -A 10 'page-info-drawer' docs/build/getting_started/introduction/index.html | head -20
```

Expected: a `<details class="page-info-drawer">` block is present, containing a `<span class="status-pill status-needs-proofreading">` and the metadata table.

- [ ] **Step 3: Open in browser and verify visual output**

```bash
open docs/build/getting_started/introduction/index.html
```

Confirm:
- The drawer is **collapsed** by default, showing `▶ Page info` + the orange "needs proofreading" pill.
- Clicking the drawer **expands** it to reveal the monospace key-value table.
- The pill colour matches the CSS (orange tint background, darker orange text).
- Dark mode (toggle in Documenter top-right): drawer background and pill colours adapt correctly.

- [ ] **Step 4: Commit**

```bash
git add docs/src/getting_started/introduction.md
git commit -m "docs(intro): add page-info metadata drawer — needs proofreading"
```

---

## Applying tags to other pages (post-plan editorial work)

After the infrastructure is confirmed working, add `{page-info}` blocks to remaining pages using the following authoring rules:

**Plain `.md` files** — insert the block directly after the `# Title` line.

**Literate `.jl` tutorial sources** — use `# ` prefixed comments so Literate passes them through as markdown:

```julia
# # Ex 1. SVD, Schmidt Rank, and Truncation
#
# {page-info
#   status: draft, needs-proofreading
#   updated: 2026-07-19
#   written: Bavithra Govintharajah
# }
```

**All four fields are optional.** A status-only block is valid:

```
{page-info
  status: draft
}
```
