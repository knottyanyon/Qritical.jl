# Architecture Decision Records

I haven't really seen people use ADRs in the Julia ecosystem — it seems more
like a practice from the Python/Java world. But I liked the idea enough to
borrow it anyway.

These don't follow the ADR format to the letter. Think of them less as
formal engineering documents and more like a lab notebook crossed with a
personal blog — the kind you'd bring to a weekly exercise session with your
tutor to explain what you tried, why you made a certain choice, and what
you're still unsure about. Writing them also doubles as a way of thinking out
loud: sometimes I need a rubber duck to spot a flaw in my reasoning, sometimes
I need a pair programmer to push an idea further. Either way, the act of
writing it down is the point. A place to talk to my future self.

Part of why I'm doing this at all is that I've learnt from my own past solo
projects — the ones where decisions were made on the fly, never written down,
and became impossible to untangle months later. Writing things down as I go
is a habit I'm deliberately building this time around.

A few things worth keeping in mind when reading these:

- I'm still learning Julia, so some of the ideas here started out very
  Pythonic. I've tried to flag those where I caught them, but there are
  probably more. These records help me go back and correct course once I know
  better.

- My style preference in code is to stay as close to the physics on paper as
  possible. That means some implementation decisions are made for
  physicist-friendliness first, not programmer-friendliness. I'd rather have
  code that reinforces the hand derivation than code that a software engineer
  would call idiomatic.

- The decisions here are also a way of keeping the door open. Especially
  early on when I didn't fully know where the project was going, writing down
  the reasoning helped me avoid committing to something that would be hard to
  undo later.

## Records

- [ADR 0001 — Covariant index type system](0001-covariant-index-type-system.md)
- [ADR 0002 — Bond index arrow orientation](0002-bond-index-arrow-orientation.md)
- [ADR 0003 — Backend dispatch via scoped context](0003-backend-dispatch-scoped-context.md)
- [ADR 0004 — Special tensor types](0004-special-tensor-types.md)
- [ADR 0005 — SVD truncation tracking](0005-svd-truncation-tracking.md)
- [ADR 0006 — Schmidt spectrum type](0006-schmidt-spectrum-type.md)
- [ADR 0007 — MPS type](0007-mps-type.md)
