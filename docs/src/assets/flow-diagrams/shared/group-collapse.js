import { cssColor } from './palette.js';

// Diagram authors declare a collapsible group as a normal node (shape:
// 'group') carrying a `children` id list; every other node/edge is written
// exactly as it would be without any group at all — including edges that
// cross the group's boundary, which are written straight to/from the real
// child they touch. This module derives the node/edge set actually visible
// for a given set of collapsed group ids: while a group is collapsed, its
// children and any edge between two of them are hidden, and a boundary edge
// (real child on one end, something outside the group on the other) is
// rewritten to attach to the group's own node instead — so the diagram
// author never has to write two versions of the same wiring.
export function computeEffectiveGraph(nodes, edges, collapsedGroupIds) {
  // A group is any node authored with a `children` list — its own visual
  // `shape` is independent (dashed generic box, ISO predefined-process
  // symbol, whatever the diagram calls for), so groups aren't identified by
  // shape.
  const groups = nodes.filter(n => Array.isArray(n.children));
  const childToGroup = new Map();
  groups.forEach(gr => gr.children.forEach(childId => childToGroup.set(childId, gr.id)));

  // The collapsed/expanded indicator itself is drawn as a proper SVG shape
  // in render-flow.js, not appended to the label text — see
  // shapes.group/renderCollapseIndicator there.
  const visibleNodes = nodes
    .filter(n => Array.isArray(n.children) || !collapsedGroupIds.has(childToGroup.get(n.id)));

  const seenPairs = new Set();
  const visibleEdges = [];

  edges.forEach(e => {
    const fromGroup = childToGroup.get(e.from);
    const toGroup = childToGroup.get(e.to);
    const fromCollapsed = fromGroup && collapsedGroupIds.has(fromGroup);
    const toCollapsed = toGroup && collapsedGroupIds.has(toGroup);

    // Both ends are children of the same group: purely internal, so it's
    // either hidden (group collapsed) or shown exactly as authored.
    if (fromGroup && toGroup && fromGroup === toGroup) {
      if (fromCollapsed) return;
      visibleEdges.push(e);
      return;
    }

    const from = fromCollapsed ? fromGroup : e.from;
    const to = toCollapsed ? toGroup : e.to;
    if (from === to) return; // both ends folded into the same collapsed group
    const key = `${from}->${to}`;
    if (seenPairs.has(key)) return;
    seenPairs.add(key);
    visibleEdges.push({ ...e, from, to });
  });

  // An expanded group whose boundary edges all now land on real children
  // (rather than the group node itself) has nothing left connecting it to
  // the diagram, so dagre would place it as a disconnected component
  // wherever it likes. A light anchor edge to the group's first child keeps
  // the header laid out right above its own subtree instead of floating off
  // somewhere unrelated.
  groups.forEach(gr => {
    if (collapsedGroupIds.has(gr.id) || !gr.children.length) return;
    const anchorTo = gr.children[0];
    const key = `${gr.id}->${anchorTo}`;
    if (seenPairs.has(key)) return;
    visibleEdges.push({ from: gr.id, to: anchorTo, style: { stroke: cssColor('--diagram-color-muted'), strokeLineDash: [3, 3] } });
  });

  return { nodes: visibleNodes, edges: visibleEdges };
}
