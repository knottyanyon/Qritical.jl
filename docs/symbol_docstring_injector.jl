# Symbol usage injection for exercise references in docstrings
# This file discovers which Qritical symbols are used in exercise notebooks
# and injects "Used in exercises" sections into their docstrings.

"""
    find_notebook_symbol_usage(notebooks_dir::String) -> Dict{Symbol, Vector{String}}

Scan Jupyter notebooks and return a mapping of symbol -> list of exercise names.
Returns an empty dict if JSON is unavailable (symbol discovery is optional).
"""
function find_notebook_symbol_usage(notebooks_dir::String)
    usage = Dict{Symbol, Vector{String}}()

    # Check if JSON is available in the current scope
    if !isdefined(Main, :JSON)
        # Silently skip if JSON is not available — symbol discovery is optional
        return usage
    end

    if !isdir(notebooks_dir)
        return usage
    end

    for (root, dirs, files) in walkdir(notebooks_dir)
        for file in files
            if endswith(file, ".ipynb")
                notebook_path = joinpath(root, file)

                # Extract exercise number from path
                # e.g., docs/src/exercises/01/ex_01.ipynb -> Exercise 01
                relative = relpath(notebook_path, notebooks_dir)
                parts = split(relative, "/")
                if length(parts) >= 2 && startswith(parts[end], "ex_")
                    exercise_num = parts[1]
                    exercise_name = "Exercise $exercise_num"

                    try
                        # Parse notebook JSON
                        notebook_json = Main.JSON.parse(read(notebook_path, String))

                        # Extract code from all code cells
                        cells = get(notebook_json, "cells", [])
                        for cell in cells
                            if get(cell, "cell_type", "") == "code"
                                source = get(cell, "source", [])
                                code_text = if isa(source, Vector)
                                    join(source, "")
                                else
                                    String(source)
                                end

                                # Extract Qritical symbols from the code
                                symbols = extract_qritical_symbols(code_text)
                                for sym in symbols
                                    if !haskey(usage, sym)
                                        usage[sym] = String[]
                                    end
                                    if exercise_name ∉ usage[sym]
                                        push!(usage[sym], exercise_name)
                                    end
                                end
                            end
                        end
                    catch
                        # Silently skip notebooks that fail to parse — notebook parsing
                        # is optional for the docs build to succeed
                        continue
                    end
                end
            end
        end
    end

    usage
end

"""
    extract_qritical_symbols(code::String) -> Set{Symbol}

Extract identifiers from Julia code that are likely Qritical symbols.
"""
function extract_qritical_symbols(code::String)
    symbols = Set{Symbol}()

    # List of known Qritical exports to look for
    known_exports = Set([
        # Index layer
        "IxLoc", "Upper", "Lower", "TIx", "MulTIx", "AbstractIx",
        "dim", "label", "which_space", "upper", "lower", "uppers", "lowers",
        "flip", "uppers_range", "lowers_range",
        # Partitions
        "Partition", "Bipartition", "complement", "bipartition", "group_legs",
        "bond_label",
        # QTensor
        "QTensor", "dagger",
        # SVD
        "do_svd", "AbstractTrunc", "NoTrunc", "MaxBondDimTrunc", "ValCutoffTrunc",
        # Spectrum
        "SingValSpectrum", "EigValSpectrum", "schmidt_rank", "entanglement_entropy",
        # MPS
        "FiniteMPS", "to_mps", "add_mps", "overlap", "local_expectation", "two_site_op",
        "CanonicalizeConfig", "LeftCanonical", "RightCanonical", "BondCanonical",
        "canonicalize", "canonical_error", "is_canonical", "to_vidal", "to_canonical",
        # Geometry
        "Chain", "sites", "bonds",
        # DoF
        "Spin", "SpinHalf", "SpinOne", "SpinlessFermion", "Electron", "MajoranaFermion",
        "HardCoreBoson", "local_dim", "algebra_generators",
        # Operators
        "LatticeOperator", "Hamiltonian", "XXZ", "Heisenberg", "Ising",
        "uniform_coupling", "OneSiteTerm", "TwoSiteTerm", "op_at_site",
        "total_magnetization", "staggered_magnetization", "identity_operator",
        "matrix_repr", "DenseFormat", "SparseFormat",
        # MPO
        "FiniteMPO", "MPO", "expect", "apply_mpo",
        # TEBD
        "neel_state", "Quench", "TEBD", "Tracker", "NoTracker", "solve",
        "TimeAxis", "RealTime", "ImaginaryTime", "Propagator", "gate",
        "ConstantProtocol", "bond_hamiltonian", "apply_gate",
        # ED
        "ExactDiagonalization", "GroundState", "power_method", "PowerMethodResult",
        # Storage formats
        "StorageFormat",
    ])

    # Find all identifiers in the code
    pattern = r"\b([a-zA-Z_]\w*)\b"
    for m in eachmatch(pattern, code)
        word = m.captures[1]
        if word in known_exports
            push!(symbols, Symbol(word))
        end
    end

    symbols
end

"""
    create_exercise_page_mapping(exercises_dir::String) -> Dict{String, String}

Create a mapping from exercise names (e.g., "Exercise 01") to their doc page paths.
"""
function create_exercise_page_mapping(exercises_dir::String)
    pages = Dict{String, String}()

    for i in 1:11
        exercise_num = lpad(i, 2, '0')
        exercise_name = "Exercise $exercise_num"
        page_path = "exercises/$exercise_num/ex_$exercise_num"  # Documenter adds .md
        pages[exercise_name] = page_path
    end

    pages
end

"""
    inject_usage_into_module_docstrings!(mod::Module, usage::Dict{Symbol, Vector{String}},
                                         exercise_pages::Dict{String, String})

Inject "Used in exercises" sections with links into the docstrings of all exported symbols.
"""
function inject_usage_into_module_docstrings!(mod::Module, usage::Dict{Symbol, Vector{String}},
                                             exercise_pages::Dict{String, String})
    # Get all exported symbols from the module
    exports = names(mod; all=false)  # only exported names

    for sym in exports
        if haskey(usage, sym)
            exercises = usage[sym]
            try
                # Get the current docstring
                binding = Docs.Binding(mod, sym)
                doc = Docs.getdoc(binding)

                if doc !== nothing && isa(doc, Markdown.MD)
                    # Build "Used in exercises" section with links
                    used_in_lines = ["## Used in exercises", ""]
                    for exercise in sort(exercises)
                        if haskey(exercise_pages, exercise)
                            page_path = exercise_pages[exercise]
                            # Create a markdown link
                            push!(used_in_lines, "- [$exercise](@ref $page_path)")
                        else
                            push!(used_in_lines, "- $exercise")
                        end
                    end
                    used_in_text = join(used_in_lines, "\n")

                    # Append to docstring
                    append!(doc.content, Markdown.parse(used_in_text).content)

                    # Update the docstring in the module's docs
                    Docs.setdoc!(binding, doc)
                end
            catch e
                # Silently skip if we can't inject for a symbol
                # @warn "Failed to inject usage for $sym: $e"
            end
        end
    end
end
