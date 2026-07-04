
## getting_started/indexed_tensor/
- Diving directly into explaining indices feels abrupt. explain the design idea of leg and how index is related to it. highlight the physics reasoning behind the design before going over the examples indices. put this as a separate page labeled introduction so that it comes before the `indexed_tensor` page


- the line "The inner constructor rejects non-positive dimensions immediately..." doesn't feel suitable for the getting started section. it feels more appropriate inside the docs page for the struct `TIx` explaining the kind of validation machinary built into `TIx`
- Put this in a tip or a suitable admonition box instead of plain text: "Use the constructor helpers upper and lower...."
- add a few lines explaining the concept of a symbol in Julia and how one can define a symbol as that is how one defined an upper or lower index. 
- rephrase the sentence "To build several indices of the same position....": make it "To build multiple upper or lower indices with different labels...."
- put "A dimension-1 virtual index appears..." in a note admonition
  
##  Excerpts from Insight Notes
# ─────────────────────────────────────
The brute-force comparison test exposed a subtle Julia column-major trap: sequential MPS contraction via reshape(psi * reshape(A, χL, d*χR), d^i, χR) produces a state vector with site 1 as the least significant index (varies fastest in memory), but kron(full_op, op_i) places the first-added site as the most significant. The fix — kron(op_i, full_op) — makes the kron grow outward, matching Julia's memory layout. This is a common source of silent sign errors in numerical quantum mechanics code.