# flowdiagram_preprocessor.jl
#
# Converts {{flowdiagram: category/name}} tokens in Documenter .md source files
# into the @raw html embed markup that mounts an interactive diagram from
# qritical-numroutines-diagrams (synced into assets/flow-diagrams/ by that
# repo's sync-to-qritical.sh).
#
# Authoring syntax (anywhere in a .md file):
#
#   {{flowdiagram: numerical-routines/svd-truncation-decision}}
#
# `category` must be `numerical-routines` or `tensor-network-diagrams` — the
# same two directories qritical-numroutines-diagrams organizes diagrams into.
# `name` is the diagram file's base name (no `.js`), exactly as registered in
# that repo's preview/index.html.
#
# In Literate .jl tutorial sources, prefix the line with "# ":
#
#   # {{flowdiagram: numerical-routines/svd-truncation-decision}}

const FLOWDIAGRAM_PATTERN = r"\{\{flowdiagram:\s*([\w-]+)/([\w-]+)\s*\}\}"

const _FLOWDIAGRAM_CATEGORIES = ("numerical-routines", "tensor-network-diagrams")

"""
    _flowdiagram_html(category, name, depth) -> String

Render the @raw html block that mounts `assets/flow-diagrams/<category>/<name>.js`
into the page. `depth` is how many directories deep the current file sits below
`docs/src/`, used to compute the right number of `../` hops back to `assets/`.
"""
function _flowdiagram_html(
    category::AbstractString, name::AbstractString, depth::Int
)::String
    prefix = repeat("../", depth)
    dagre_script = if category == "numerical-routines"
        "<script src=\"$(prefix)assets/flow-diagrams/vendor/dagre.min.js\"></script>\n"
    else
        ""
    end

    return """```@raw html
<link rel="stylesheet" href="$(prefix)assets/flow-diagrams/flow-diagram.css">
<link rel="stylesheet" href="$(prefix)assets/flow-diagrams/vendor/katex/katex.css">
<script>
// Documenter's own MathJax3 config loads require.js, which defines a global
// define() that looks AMD-compatible. rough.min.js/dagre.min.js are UMD
// bundles that, when they see a global define(), register as anonymous AMD
// modules instead of attaching window.rough/window.dagre — so every
// flowdiagram mount() call would fail with "rough is not defined". Hiding
// define() while these two vendor scripts load forces the UMD branch that
// sets the global, then restores it for anything else on the page that
// needs it (e.g. Documenter's own MathJax bootstrapping).
window.__flowdiagram_saved_define = window.define;
window.define = undefined;
</script>
<script src="$(prefix)assets/flow-diagrams/vendor/rough.min.js"></script>
$(dagre_script)<script>
window.define = window.__flowdiagram_saved_define;
delete window.__flowdiagram_saved_define;
</script>
<div class="flow-diagram-wrapper">
  <div id="flowdiagram-$(category)-$(name)"></div>
</div>
<script type="module">
import('$(prefix)assets/flow-diagrams/$(category)/$(name).js')
  .then(m => m.mount(document.getElementById('flowdiagram-$(category)-$(name)')));
</script>
```"""
end

"""
    flowdiagram_preprocessor(content, depth) -> String

Replace every `{{flowdiagram: category/name}}` token in `content` with its
rendered embed markup. `depth` is the current file's directory depth below
`docs/src/` (0 for `index.md`, 1 for `getting_started/index.md`, etc.).
"""
function flowdiagram_preprocessor(content::AbstractString, depth::Int)::String
    return replace(
        content,
        FLOWDIAGRAM_PATTERN => function (m)
            cap = match(FLOWDIAGRAM_PATTERN, m)
            cap === nothing && return m
            category, name = cap[1], cap[2]
            if !(category in _FLOWDIAGRAM_CATEGORIES)
                error(
                    "flowdiagram_preprocessor: unknown category \"$category\" in token \"$m\" — " *
                    "expected one of $_FLOWDIAGRAM_CATEGORIES",
                )
            end
            return _flowdiagram_html(category, name, depth)
        end,
    )
end

"""
    preprocess_flowdiagrams(src_dir)

Walk `src_dir` recursively, applying `flowdiagram_preprocessor` to every `.md`
file that contains a `{{flowdiagram: ...}}` token. Files without a token are
untouched.
"""
function preprocess_flowdiagrams(src_dir::AbstractString)
    for (root, _, files) in walkdir(src_dir)
        dir_depth = length(splitpath(relpath(root, src_dir))) - (root == src_dir ? 1 : 0)
        for file in files
            endswith(file, ".md") || continue
            fpath = joinpath(root, file)
            content = read(fpath, String)
            # Documenter's default prettyurls builds "dir/name.md" as
            # "dir/name/index.html" — one directory deeper than the source
            # path — except "index.md", which stays at "dir/index.html".
            depth = dir_depth + (file == "index.md" ? 0 : 1)
            updated = flowdiagram_preprocessor(content, depth)
            updated != content && write(fpath, updated)
        end
    end
end

export flowdiagram_preprocessor, preprocess_flowdiagrams
