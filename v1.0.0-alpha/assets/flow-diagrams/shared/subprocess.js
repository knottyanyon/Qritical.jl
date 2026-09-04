import { startNode, endNode, groupNode } from './components.js';
import { renderFlowDiagram } from './render-flow.js';

// A subprocess is authored once here — its box appearance when used as a
// step in a parent diagram, and its inner flow (no start/stop pills; those
// are added automatically by whichever render path is used, see
// mountSubprocessStandalone/subprocessStep below) — instead of being
// hand-duplicated once per place it's used.
// `box.detail` is carried through for documentation/future use only — it is
// never rendered (groupNode's click is claimed by the expand/collapse
// toggle, and groupNode doesn't even accept a detail param).
export function defineSubprocess({ id, box, entry, exit, nodes, edges }) {
  if (!nodes.some(n => n.id === entry)) {
    throw new Error(`defineSubprocess("${id}"): entry id "${entry}" is not in nodes`);
  }
  if (!nodes.some(n => n.id === exit)) {
    throw new Error(`defineSubprocess("${id}"): exit id "${exit}" is not in nodes`);
  }
  for (const e of edges) {
    if (!nodes.some(n => n.id === e.from)) {
      throw new Error(`defineSubprocess("${id}"): edge references unknown node id "${e.from}"`);
    }
    if (!nodes.some(n => n.id === e.to)) {
      throw new Error(`defineSubprocess("${id}"): edge references unknown node id "${e.to}"`);
    }
  }
  const fullBox = { shape: 'predefined-process', ...box };
  return Object.freeze({ id, box: fullBox, entry, exit, nodes, edges });
}

// Renders `def` as a complete standalone diagram: its inner flow wrapped
// with an auto-generated start pill feeding `entry` and an auto-generated
// end pill fed by `exit`.
export function mountSubprocessStandalone(def, container) {
  // '__start'/'__end' are reserved sentinel ids for the auto-generated pills
  // below; a subprocess node literally named '__start' or '__end' would
  // collide with them.
  const nodes = [
    startNode({ id: '__start' }),
    ...def.nodes,
    endNode({ id: '__end' })
  ];
  const edges = [
    { from: '__start', to: def.entry },
    ...def.edges,
    { from: def.exit, to: '__end' }
  ];
  renderFlowDiagram(container, { nodes, edges });
}

// Returns everything a parent diagram needs to splice `def` in as one
// expandable step: a groupNode header (using def.box) whose `children` are
// every inner node's id namespaced as `${instanceId}:${originalId}` (so the
// same def can be embedded more than once, or alongside an unrelated node
// that happens to reuse an id, without colliding), the inner nodes/edges
// with every id rewritten the same way, and the namespaced entry/exit ids
// for the parent to wire its own boundary edges into — e.g.
// `{ from: 'residual', to: step.entry }`, `{ from: step.exit, to: 'converged' }`.
//
// Calling this twice for the same def with no explicit instanceId (or with
// the same explicit instanceId twice) produces identical namespaced ids for
// both embeddings. If both embeddings end up in the *same* parent diagram's
// nodes/edges arrays, that's a silent id collision — dagre merges them into
// one node with no error, rendering a wrong-but-plausible-looking diagram.
// A second embedding of the same def in one parent therefore requires an
// explicit, distinct instanceId. (Embedding the same def with the same
// default instanceId in two *different*, independently-rendered diagrams —
// e.g. the same subprocess reused standalone and inline elsewhere — is fine
// and common; renderFlowDiagram fails fast with a clear error if two nodes
// with the same id actually land in one render call.)
// `tabs` (optional) is passed straight through to the header's groupNode —
// e.g. an explanationTab/codeTab pulled from `def.box.detail`, or an
// expandTab() to move the fold/unfold trigger off the whole-box click and
// onto its own button (see shared/components.js).
export function subprocessStep(def, instanceId = def.id, tabs) {
  const ns = id => `${instanceId}:${id}`;
  const nodes = def.nodes.map(n => ({
    ...n,
    id: ns(n.id),
    ...(n.children ? { children: n.children.map(ns) } : {})
  }));
  const edges = def.edges.map(e => ({ ...e, from: ns(e.from), to: ns(e.to) }));
  const header = groupNode({
    id: instanceId,
    label: def.box.label,
    shape: def.box.shape,
    children: nodes.map(n => n.id),
    tabs
  });
  return { header, nodes, edges, entry: ns(def.entry), exit: ns(def.exit) };
}
