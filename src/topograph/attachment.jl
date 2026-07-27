
# SECTION - Attachment always derived, never stored

"""
    Attachment

Abstract supertype for the four different ways a wire's ends can be attached:

  - [`Pinned`](@ref), both ends attached
  - [`HalfLoose`](@ref), exactly one end attached
  - [`Loose`](@ref), neither end attached
  - [`Circle`](@ref), the wire is closed, with no ends at all

Each of these four is a singleton, a struct with no fields, so there is only ever one value of that type in existence. Using a distinct type for each case, rather than a `Symbol` or an `Int` tag, is the "Holy traits" pattern: it lets [`attachment`](@ref) dispatch on the answer as a type at compile time, instead of branching on a runtime value. None of the four are ever stored on `Wire` itself, `attachment(w)` always computes the answer fresh from `w`'s ends.
"""
abstract type Attachment end

"""
    Pinned <: Attachment

Both ends of the wire are attached.

```text
●───────▶ start   finish ◀───────●
```

!!! definition "Pinned"
    "Pinned" is Joyal's own word: an edge is pinned when its inclusion extends to both of its endpoints ([joyal_street_1991](@cite), p. 61).

!!! tip
    A contracted bond if the two ends belong to different nodes; a trace if they belong to the same node. `attachment` cannot and does not distinguish the two, that depends on `Leg.owner`, not on the wire itself.

# Examples

```jldoctest
julia> w = Wire(WireId(1), 4; label=:bond, start=LegId(1), finish=LegId(2));

julia> attachment(w)
Pinned()
```
"""
struct Pinned <: Attachment end

"""
    HalfLoose <: Attachment

Exactly one end of the wire is attached.

```text
●───────▶ start   finish (open)
```

!!! tip
    If you specialise this to a tensor network built as an MPS, a half-loose wire is exactly what a physical leg is: one end sits on the site tensor, the other end is left open.

# Examples

```jldoctest
julia> w = Wire(WireId(1), 4; label=:open, start=LegId(1));

julia> attachment(w)
HalfLoose()
```
"""
struct HalfLoose <: Attachment end

"""
    Loose <: Attachment

Neither end of the wire is attached.

```text
start (open)   finish (open)
```

!!! tip
    A bare identity wire. When a loose wire has to be written down as an explicit tensor rather than left implicit, e.g. for `ncon`, it is represented as a Kronecker delta `δ`: the "do nothing" tensor that relabels an index without acting on the state at all.

# Examples

```jldoctest
julia> w = Wire(WireId(1), 4; label=:free);

julia> attachment(w)
Loose()
```
"""
struct Loose <: Attachment end

"""
    Circle <: Attachment

The wire is closed: a component with no nodes at all.

```text
 ╭───╮
 ╰───╯   (no start, no finish, no owner)
```

!!! note
    A circle contributes only a scalar factor `dim(space)` to the network, and it is invisible to `@tensor`. `@tensor` (like Einstein summation generally) only ever acts on indices that appear as a labelled leg on some tensor, a circle has no legs and no owning tensor at all, so there is nothing for it to see or sum over. Its entire effect is that scalar prefactor, tracked separately rather than inside the `@tensor` expression itself.

# Examples

```jldoctest
julia> w = Wire(WireId(1), 4; label=:loop, closed=true);

julia> attachment(w)
Circle()
```
"""
struct Circle <: Attachment end

"""
    attachment(w::Wire) -> Attachment

Classify wire `w` by how many of its ends are attached:

```text
start   finish        Attachment
 set     set     ⟶    Pinned()
 set    empty    ⟶    HalfLoose()
empty    set     ⟶    HalfLoose()
empty   empty    ⟶    Loose()

        (closed flag overrides all four rows) ⟶ Circle()
```

!!! design-note "Design note"
    The answer is derived from `w.start`, `w.finish`, and `w.closed` every time, it is never read from a field stored on `Wire` itself. Storing it separately would mean every place that changes an end ([`pin!`](@ref), [`cut!`](@ref), [`attach!`](@ref)) would also have to remember to keep that field in sync, and any place that forgot would leave a wire whose recorded attachment silently disagreed with its actual ends. Deriving it removes that possibility entirely, there is only one source of truth: the ends themselves.

The closed flag is checked first: a closed wire is a [`Circle`](@ref) regardless of what `start`/`finish` happen to hold.
"""
function attachment(w::Wire)
    w.closed && return Circle()
    a = w.start !== nothing
    b = w.finish !== nothing
    (a && b) && return Pinned()
    (a || b) && return HalfLoose()
    return Loose()
end
