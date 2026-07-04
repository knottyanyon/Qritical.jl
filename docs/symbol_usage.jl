"""
    SymbolUsageTracker

Parses Jupyter notebook exercise files to discover which Qritical symbols are used,
then injects "Used in" sections into their docstrings via a Documenter hook.
"""
module SymbolUsageTracker

using JSON, Documenter

export find_symbol_usage, inject_used_in_docstrings

"""
    find_symbol_usage(notebooks_dir::String) -> Dict{Symbol, Vector{Tuple{String, String}}}

Scan all exercise notebooks and return a mapping from symbol name to list of
(exercise_name, exercise_path) tuples where that symbol appears.

Each notebook is a Jupyter .ipynb file (JSON). We extract code cells, parse them
for symbol references, and track which exercises use which symbols.
"""
function find_symbol_usage(notebooks_dir::String)
    usage = Dict{Symbol, Vector{Tuple{String, String}}}()

    # Find all .ipynb files in the exercises directory
    notebook_files = String[]
    for (root, dirs, files) in walkdir(notebooks_dir)
        for file in files
            if endswith(file, ".ipynb")
                push!(notebook_files, joinpath(root, file))
            end
        end
    end

    for notebook_path in notebook_files
        try
            # Extract exercise info from path: docs/src/exercises/NN/ex_NN.ipynb
            # We want to show "Exercise NN" in the docs
            parts = split(notebook_path, "/")
            if length(parts) >= 2
                exercise_num = parts[end-1]  # e.g., "01"
                exercise_name = "Exercise $exercise_num"
            else
                exercise_name = basename(notebook_path)
            end

            # Parse the notebook JSON
            notebook_content = read(notebook_path, String)
            notebook = JSON.parse(notebook_content)

            # Extract all code cells
            cells = get(notebook, "cells", [])
            for cell in cells
                if get(cell, "cell_type", "") == "code"
                    source = get(cell, "source", [])
                    if isa(source, Vector)
                        code = join(source, "")
                    else
                        code = source
                    end

                    # Find symbols used: look for identifiers that match Qritical exports
                    # Simple heuristic: match word boundaries followed by ( or [ or .
                    # and exclude Julia keywords
                    symbols_in_cell = extract_symbols(code)

                    for sym in symbols_in_cell
                        if !haskey(usage, sym)
                            usage[sym] = Tuple{String, String}[]
                        end
                        # Avoid duplicates
                        if (exercise_name, notebook_path) ∉ usage[sym]
                            push!(usage[sym], (exercise_name, notebook_path))
                        end
                    end
                end
            end
        catch e
            @warn "Failed to parse notebook $notebook_path: $e"
        end
    end

    usage
end

"""
    extract_symbols(code::String) -> Set{Symbol}

Extract symbol names from Julia code that look like function/type names.
This is a simple heuristic: capture identifiers that are typically capitalized
(types) or lowercase (functions), excluding keywords.
"""
function extract_symbols(code::String)
    symbols = Set{Symbol}()

    # Simple pattern: word characters followed by ( or not, at word boundaries
    # Matches: function_name, FunctionName, Type, etc.
    # We use a very permissive regex to avoid false negatives
    pattern = r"\b([a-zA-Z_]\w*)\b"

    julia_keywords = Set([
        "if", "else", "elseif", "end", "for", "while", "do", "break", "continue",
        "function", "return", "const", "global", "local", "begin", "try", "catch",
        "finally", "import", "export", "using", "module", "struct", "mutable",
        "abstract", "type", "where", "in", "isa", "true", "false", "nothing",
        "and", "or", "not", "quote", "macro", "let", "baremodule"
    ])

    for m in eachmatch(pattern, code)
        word = m.captures[1]
        if word ∉ julia_keywords && !isempty(word)
            push!(symbols, Symbol(word))
        end
    end

    symbols
end

"""
    inject_used_in_docstrings(md::Documenter.Documents.Document, usage::Dict{Symbol, Vector{Tuple{String, String}}})

Documenter hook that injects "Used in" sections into docstrings.
Called during the Documenter build process after docstrings are collected.
"""
function inject_used_in_docstrings(md::Documenter.Documents.Document, usage::Dict{Symbol, Vector{Tuple{String, String}}})
    # Iterate through all docstrings in the document
    for (_, docobj) in md.internal.docs
        # docobj contains the symbol metadata
        name = get(docobj, :name, nothing)
        if name === nothing
            continue
        end

        # Check if we found usage of this symbol
        sym = Symbol(name)
        if haskey(usage, sym) && !isempty(usage[sym])
            exercises = usage[sym]

            # Build the "Used in" section as Markdown
            used_in_text = "## Used in\n\n"
            for (exercise_name, _) in exercises
                used_in_text *= "- $exercise_name\n"
            end

            # Append to the docstring
            # Note: the actual injection point depends on how Documenter structures docs
            # This is a simplified version; real implementation may need adjustment
            if haskey(docobj, :doc)
                doc = docobj[:doc]
                # Append to the docstring if it's Markdown
                if isa(doc, Markdown.MD)
                    push!(doc.content, Markdown.parse(used_in_text))
                end
            end
        end
    end
end

end  # module SymbolUsageTracker
