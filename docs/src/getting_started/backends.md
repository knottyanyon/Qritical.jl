# Backend Modes

When I first started writing tensor network code for the course, plain Julia arrays were exactly what I needed — simple, readable, and easy to debug by hand. But as the lectures progressed and topics like canonical forms started coming up with exercises about XXZ spin chain model, I realised I wanted to keep that simple code working while also leaving a door open for something more powerful later.

The idea is this: many physical systems have symmetries (like conserved total spin) that make their tensors block-sparse — most entries are zero by structure, not by accident. Exploiting that can make contractions dramatically faster at large bond dimensions. Our tutor introduced `TensorOperations.jl` and the `@tensor` macro early on, and it turns out that `TensorKit.jl` — which is part of the same [QuantumKitHub](https://github.com/QuantumKitHub) ecosystem — already has all of that block-sparse, symmetry-aware machinery built in.

So rather than rewriting everything later, the package lets you choose a backend: `:native` for plain Julia arrays (the default, and where all learning happens), and `:tensorkit` as a future opt-in once that bridge is built. The same algorithm code works in both modes — only the backing store changes.

## The :native backend

```jldoctest
julia> current_backend()
:native
```

## Switching backends

To use a different backend, wrap your code in a `with_backend` block. Think of it like a temporary setting — whatever you do inside the block runs under that backend, and when the block ends everything goes back to how it was. Nothing leaks out.

```jldoctest
julia> with_backend(:native) do
           current_backend()
       end
:native

julia> with_backend(:tensorkit) do
           current_backend()
       end
:tensorkit
```

Passing an unrecognised name throws immediately — before your block even runs:

```jldoctest
julia> with_backend(:unknown) do end
ERROR: ArgumentError: Unknown backend :unknown. Use :native or :tensorkit.
[...]
```

