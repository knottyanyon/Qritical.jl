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
    "draft" => "draft",
    "needs-rewrite" => "needs rewrite",
    "needs-proofreading" => "needs proofreading",
    "needs-touchups" => "needs touch-ups",
    "needs-theory" => "needs theory",
    "proofread" => "proofread",
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
        pills = _status_pill.(slugs)
        pills_html = join(pills, "\n        ")
        status_row =
            "\n      <tr><td class=\"pi-key\">status</td>" *
            "<td class=\"pi-val\">" *
            join(pills, " ") *
            "</td></tr>"
    end

    # Optional ordered metadata rows
    rows = String[status_row]
    for (key, display) in
        (("updated", "last updated"), ("written", "written by"), ("edited", "edited by"))
        haskey(meta, key) || continue
        push!(
            rows,
            "\n      <tr><td class=\"pi-key\">$(display)</td>" *
            "<td class=\"pi-val\">$(meta[key])</td></tr>",
        )
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
    return replace(content, PAGE_INFO_PATTERN => function (m)
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
            fpath = joinpath(root, file)
            content = read(fpath, String)
            updated = page_meta_preprocessor(content)
            updated != content && write(fpath, updated)
        end
    end
end

export page_meta_preprocessor, preprocess_page_meta
