using Serialization

"""
    load_state_from_file(filepath::String)

Load a `.jls` serialized file from disk and return the deserialized object.
Expects the stored object to be an Array or Matrix.
"""
function load_state_from_file(filepath::String)
    # Validate extension
    if !endswith(filepath, ".jls")
        msg = "File must have a .jls extension: $filepath"
        @error msg filepath = filepath
        throw(ArgumentError(msg))
    end

    # Validate file exists
    if !isfile(filepath)
        msg = "File not found: $filepath"
        @error msg filepath = filepath
        throw(SystemError(msg))
    end

    data = open(filepath, "r") do io
        deserialize(io)
    end

    # Validate it's an array type
    if !(data isa AbstractArray)
        throw(TypeError(:load_state_from_file, "AbstractArray", typeof(data)))
    end

    return data
end

"""
    load_state_from_file_all(dir::String; recursive::Bool=false)

Load all `.jls` files from a directory into a Dict mapping filename => array.
Set `recursive=true` to search subdirectories as well.
"""
function load_state_from_file_all(dir::String; recursive::Bool=false)
    if !isdir(dir)
        throw(SystemError("Directory not found: $dir"))
    end

    results = Dict{String,AbstractArray}()

    files = if recursive
        [joinpath(root, f) for (root, _, fs) in walkdir(dir) for f in fs if endswith(f, ".jls")]
    else
        filter(f -> endswith(f, ".jls"), readdir(dir; join=true))
    end

    for filepath in files
        try
            results[basename(filepath)] = load_state_from_file(filepath)
            println("✓ Loaded: $filepath")
        catch e
            @warn "Failed to load $filepath" exception = e
        end
    end

    return results
end

"""
    save_jls(data::AbstractArray, filepath::String)

Serialize an Array or Matrix to a `.jls` file.
"""
function save_jls(data::AbstractArray, filepath::String)
    if !endswith(filepath, ".jls")
        filepath *= ".jls"
    end
    open(filepath, "w") do io
        serialize(io, data)
    end
    return println("Saved to: $filepath")
end