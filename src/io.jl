using Serialization
using DelimitedFiles

# ==== Array I/O ===============================================================

"""
    load_array(path::String) -> AbstractArray

Load a numerical array from `path`, dispatching on the file extension.

| Extension | Backend | Notes |
|-----------|---------|-------|
| `.jls`    | `Serialization.deserialize` | Exact round-trip; Julia-native; preferred for fixtures |
| `.txt`    | `DelimitedFiles.readdlm` | Human-readable; precision limited to text repr |

An unsupported extension throws an `ArgumentError`.

# Example
```julia
using Serialization
serialize("data.jls", rand(4, 4))
A = load_array("data.jls")   # round-trips exactly
```

# See also
[`as_state`](@ref)
"""
function load_array(path::String)
    ext = lowercase(splitext(path)[2])
    if ext == ".jls"
        return open(deserialize, path, "r")
    elseif ext == ".txt"
        return readdlm(path)
    else
        throw(ArgumentError("load_array: unsupported extension \"$ext\" (supported: .jls, .txt)"))
    end
end
