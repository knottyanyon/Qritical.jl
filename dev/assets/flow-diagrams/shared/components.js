// Predefined shapes/patterns for diagram authors: call these instead of hand-writing
// node/edge (flow) or node/bond/leg (tensor) object literals for common structures.

import { cssColor } from './palette.js';

// --- Flow diagram patterns ---------------------------------------------
// One factory per ISO 5807 flowchart symbol, plus `branchEdges` for wiring a
// decision node's two labeled branches. Each factory just fills in `shape`
// and passes the rest through — see the `shapes` registry in render-flow.js
// for how each shape is actually drawn.

// ISO 5807 terminator (start/end). `terminalNode` is the pre-existing alias.
// For a *generic* entry/exit point with its own meaningful label (e.g.
// "Start: solve $J\Delta x = -r$"), use this directly. For the literal,
// unlabeled "this is where the routine begins/ends" marker that shows up in
// nearly every diagram, use startNode/endNode below instead — a full-size
// text-filled stadium is a lot of space to spend on a node that only ever
// says "Start" or "End".
export function terminatorNode({ id, label, detail }) {
  return { id, label, shape: 'terminator', detail };
}
export const terminalNode = terminatorNode;

// Small colored stadium with an icon (no label text) for the generic
// start/end marker every diagram has — see the comment on terminatorNode
// above for when to use that instead. Icons are drawn by shapes.terminator
// in render-flow.js, which checks `variant`.
export function startNode({ id, detail = 'Entry point.' } = {}) {
  return { id, shape: 'terminator', variant: 'start', width: 44, height: 34, fill: cssColor('--diagram-fill-start'), detail };
}
export function endNode({ id, detail = 'Exit point.' } = {}) {
  return { id, shape: 'terminator', variant: 'end', width: 44, height: 34, fill: cssColor('--diagram-fill-end'), detail };
}

// ISO 5807 process (a plain step/action). `tabs` (optional) is an array of
// small clickable file-tab buttons drawn in the box's top-right corner —
// see explanationTab/codeTab/expandTab below and appendNodeTabs in
// render-flow.js. When a node has tabs, they replace the whole-box click
// (the box keeps only its hover highlight).
export function processNode({ id, label, detail, tabs }) {
  return { id, label, shape: 'process', detail, tabs };
}

// ISO 5807 predefined process (a call to a named subroutine). Pass `subprocess`
// (`{ category, name, linkLabel? }`, matching another diagram file's location
// under the repo root, e.g. `{ category: 'numerical-routines', name: 'linear-solve-subprocess' }`)
// to append a link into the node's detail tooltip that opens that diagram —
// see the `data-diagram-link` handler in preview/index.html. `tabs` works the
// same as on processNode/groupNode — e.g. wrap the subprocess link in a
// codeTab/explanationTab instead of (or alongside) `detail`.
export function predefinedProcessNode({ id, label, detail, subprocess, tabs }) {
  let fullDetail = detail || '';
  if (subprocess) {
    const { category, name, linkLabel = 'View subprocess diagram →' } = subprocess;
    const link = `<a href="#" class="diagram-link" data-diagram-link="${category}/${name}">${linkLabel}</a>`;
    fullDetail = fullDetail ? `${fullDetail}<br><br>${link}` : link;
  }
  return { id, label, shape: 'predefined-process', detail: fullDetail || undefined, tabs };
}

// ISO 5807 decision.
export function decisionNode({ id, label, detail }) {
  return { id, label, shape: 'decision', detail };
}

// ISO 5807 input/output (data).
export function inputOutputNode({ id, label, detail }) {
  return { id, label, shape: 'input-output', detail };
}

// ISO 5807 preparation (e.g. loop initialization).
export function preparationNode({ id, label, detail }) {
  return { id, label, shape: 'preparation', detail };
}

// ISO 5807 document.
export function documentNode({ id, label, detail }) {
  return { id, label, shape: 'document', detail };
}

// ISO 5807 off-page connector.
export function offPageConnectorNode({ id, label, detail }) {
  return { id, label, shape: 'offpage-connector', detail };
}

// ISO 5807 (on-page) connector — small circular jump target/label.
export function connectorNode({ id, label, detail, width = 40, height = 40 }) {
  return { id, label, shape: 'connector', width, height, detail };
}

// A collapsible group: `children` lists the ids of the nodes it folds away
// when collapsed (the default). Every edge touching a child is rewritten
// automatically to attach to the group instead — see shared/group-collapse.js
// — so children and their edges are authored exactly as if the group didn't
// exist. Clicking the group toggles it; no `detail` tooltip (the click is
// already spoken for).
//
// `shape` picks the header's visual symbol — defaults to the dashed generic
// group box, but e.g. `'predefined-process'` renders it as an ISO subprocess
// symbol that expands in place instead of linking out to a separate diagram
// (compare predefinedProcessNode's `subprocess` link, which navigates away).
export function groupNode({ id, label, children, collapsed = true, shape = 'group', tabs }) {
  return { id, label, shape, children, collapsed, tabs };
}

// --- File-tab buttons -----------------------------------------------------
// Small clickable buttons drawn in a box's top-right corner (see
// appendNodeTabs in render-flow.js). Pass an array of these to `tabs` on
// processNode/groupNode/predefinedProcessNode, in any combination; when a
// node has tabs they become its only clickable surface (whole-box click is
// disabled, hover highlight is kept).

// Opens a description card, same as a leaf node's `detail` tooltip today.
export function explanationTab(detail) {
  return { kind: 'explanation', detail };
}

// Opens a description card containing a link to a code page.
export function codeTab({ href, label = 'View code →', detail }) {
  const link = `<a href="${href}" target="_blank" rel="noopener" class="diagram-link">${label}</a>`;
  return { kind: 'code', detail: detail ? `${detail}<br><br>${link}` : link };
}

// Toggles expand/collapse of a groupNode's children in place — only
// meaningful on a node that has `children`.
export function expandTab() {
  return { kind: 'expand' };
}

// Default per-branch stroke colors, so every decision in a diagram reads the
// same way at a glance without each call site repeating the styling. Read
// lazily (not module-level consts) since they resolve a CSS custom property.
const yesStyleDefault = () => ({ stroke: cssColor('--diagram-stroke-yes') });
const noStyleDefault = () => ({ stroke: cssColor('--diagram-stroke-no') });

// A single labeled branch edge out of a decision node, styled by outcome
// ('yes'/true or 'no'/false). Used directly for a decision with more than two
// branches, or by `branchEdges` for the common yes/no case.
export function decisionEdge({ from, to, outcome, label, style }) {
  const isYes = outcome === 'yes' || outcome === true;
  return {
    from,
    to,
    label: label ?? (isYes ? 'yes' : 'no'),
    style: { ...(isYes ? yesStyleDefault() : noStyleDefault()), ...style }
  };
}

// Wires the two labeled branches out of a decision node, styled by outcome
// (green for yes, red for no) so branches stay visually distinguishable
// wherever a decision diamond is used. Pass `yesStyle`/`noStyle` to override.
export function branchEdges({ from, yes, no, yesLabel = 'yes', noLabel = 'no', yesStyle, noStyle }) {
  return [
    decisionEdge({ from, to: yes, outcome: 'yes', label: yesLabel, style: yesStyle }),
    decisionEdge({ from, to: no, outcome: 'no', label: noLabel, style: noStyle })
  ];
}

// --- Tensor diagram patterns -------------------------------------------------

export function chainLayout({
  length,
  spacing = 90,
  y = 150,
  dim = 8,
  siteLabel = i => `T_${i}`,
  legLabel = i => `\\sigma_${i}`,
  boxSize = 44
}) {
  const nodes = [], bonds = [], legs = [];
  for (let i = 0; i < length; i++) {
    const x = 60 + i * spacing;
    nodes.push({ id: `s${i}`, x, y, shape: 'box', w: boxSize, h: boxSize, fill: cssColor('--diagram-fill-process'), label: siteLabel(i) });
    legs.push({ node: `s${i}`, angle: 90, length: 35, label: legLabel(i) });
    if (i > 0) bonds.push({ from: `s${i - 1}`, to: `s${i}`, dim, label: '\\chi' });
  }
  return { nodes, bonds, legs };
}

export function gridLayout({
  rows,
  cols,
  spacing = 80,
  x0 = 70,
  y0 = 70,
  dim = 4,
  siteLabel = (r, c) => `T_{${r}${c}}`
}) {
  const nodes = [], bonds = [], legs = [];
  const id = (r, c) => `n${r}_${c}`;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = x0 + c * spacing, y = y0 + r * spacing;
      nodes.push({ id: id(r, c), x, y, shape: 'circle', r: 16, fill: cssColor('--diagram-fill-circle'), label: siteLabel(r, c) });
      legs.push({ node: id(r, c), angle: -45, length: 22 });
      if (c > 0) bonds.push({ from: id(r, c - 1), to: id(r, c), dim });
      if (r > 0) bonds.push({ from: id(r - 1, c), to: id(r, c), dim });
    }
  }
  return { nodes, bonds, legs };
}

export function addGateLayer(spec, { pairs, y, gateSize = 30, fill = cssColor('--diagram-fill-gate'), label = 'U' }) {
  pairs.forEach(([i, j], k) => {
    const a = spec.nodes.find(n => n.id === `s${i}`);
    const b = spec.nodes.find(n => n.id === `s${j}`);
    const gx = (a.x + b.x) / 2;
    const gid = `gate_${y}_${k}`;
    spec.nodes.push({ id: gid, x: gx, y, shape: 'box', w: gateSize, h: gateSize, fill, label });
    spec.bonds.push({ from: 's' + i, to: gid, dim: 2 });
    spec.bonds.push({ from: 's' + j, to: gid, dim: 2 });
  });
  return spec;
}
