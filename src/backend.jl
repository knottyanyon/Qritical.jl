using ScopedValues

# ── Backend registry ──────────────────────────────────────────────────────────

const _VALID_BACKENDS = (:native, :tensorkit)
const _BACKEND = ScopedValue{Symbol}(:native)

# ── Context API ───────────────────────────────────────────────────────────────

"""
    current_backend() → Symbol

Return the active backend for the current scope. Defaults to `:native` outside
any `with_backend` block.

# Examples
```jldoctest
julia> current_backend()
:native
```
"""
current_backend() = _BACKEND[]

"""
    with_backend(f, backend::Symbol)

Execute `f` with `backend` as the active backend. Restores the previous backend
when `f` returns — including on exceptions. Valid backends: `:native`,
`:tensorkit`.

# Examples
```jldoctest
julia> with_backend(:native) do
           current_backend()
       end
:native

julia> current_backend()   # restored after the block
:native
```
"""
function with_backend(f, backend::Symbol)
    backend in _VALID_BACKENDS || throw(
        ArgumentError(
            "Unknown backend :$backend. Valid backends: $(join(_VALID_BACKENDS, ", "))."
        ),
    )
    with(f, _BACKEND => backend)
end

# ── Backend-aware IndexedTensor constructor ───────────────────────────────────

"""
    IndexedTensor(data, indices; backend=current_backend())

Construct an `IndexedTensor` routed to the active backend's storage type.
`:native` wraps a plain `Array`; `:tensorkit` is not yet implemented (see
GitHub issue #60).
"""
function IndexedTensor(
    data::Array{Element,Order},
    indices::NTuple{Order,AbstractIndex};
    backend::Symbol = current_backend(),
) where {Element,Order}
    if backend === :native
        return IndexedTensor{Element,Order,Array{Element,Order}}(data, indices)
    elseif backend === :tensorkit
        error(":tensorkit backend not yet implemented — see GitHub issue #60")
    else
        throw(
            ArgumentError(
                "Unknown backend :$backend. Valid backends: $(join(_VALID_BACKENDS, ", "))."
            ),
        )
    end
end
