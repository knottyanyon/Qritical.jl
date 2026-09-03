using Serialization   # Julia's built-in serialization library; `Serialization.serialize/deserialize` = exact binary round-trips for Julia objects
using DelimitedFiles  # Julia standard library for reading/writing delimited text files
using Artifacts       # Julia standard library resolving `Artifacts.toml`-bound data at ~/.julia/artifacts/

# SECTION -  Artifact data

"""
    tutorial_data_dir() -> String

Path to the local, on-disk copy of the `tutorial_data` artifact (tutorial/demo fixtures
distributed outside the git tree via `Artifacts.toml`). `Pkg.instantiate` downloads and
unpacks the artifact on first use; subsequent calls resolve to the same cached directory.
"""
tutorial_data_dir() = artifact"tutorial_data"   # `@artifact_str` macro resolves the name against the nearest Artifacts.toml and returns the local unpacked directory path

# SECTION -  Array I/O

"""
    load_array(path::String) -> AbstractArray

Load a numerical array from `path`, dispatching on the file extension.

| Extension | Backend                     | Notes                                                  |
|:--------- |:--------------------------- |:------------------------------------------------------ |
| `.jls`    | `Serialization.deserialize` | Exact round-trip; Julia-native; preferred for fixtures |
| `.txt`    | `DelimitedFiles.readdlm`    | Human-readable; precision limited to text repr         |

An unsupported extension throws an `ArgumentError`.

# Example

```julia
using Serialization
serialize(\"data.jls\", rand(4, 4))
A = load_array(\"data.jls\")   # round-trips exactly
```

# See also

[`as_state`](@ref)   # `path::String` = type annotation on the argument; Julia will reject non-String arguments at compile time
"""
function load_array(path::String)   # `path::String` = type annotation on the argument; Julia will reject non-String arguments at compile time 
    ext = lowercase(splitext(path)[2])   # `splitext(path)` = split "file.jls" into ("file", ".jls"); returns a 2-tuple; `[2]` = second element (1-indexed in Julia); `lowercase` normalises the extension 
    if ext == ".jls"   # `.jls` = Julia serialized format; exact binary round-trip
        return open(deserialize, path, "r")   # `open(f, path, mode)` = open file, call `f(io)`, close it; `deserialize(io)` = read the serialized Julia object; Python: `with open(path, 'rb') as f: pickle.load(f)`
    elseif ext == ".txt"   # `.txt` = delimited text format (default delimiter = whitespace)
        return readdlm(path)   # `readdlm(path)` = read a delimited text file into a Matrix{Float64} 
    else
        throw(
            ArgumentError(
                "load_array: unsupported extension \"$ext\" (supported: .jls, .txt)"
            ),
        )   # `throw(ArgumentError(...))` = raise ValueError equivalent; `\"$ext\"` = escaped double quotes inside string interpolation 
    end
end
