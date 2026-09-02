import { renderFlowDiagram } from '../shared/render-flow.js';
import { startNode, endNode, processNode, explanationTab, expandTab } from '../shared/components.js';
import { subprocessStep } from '../shared/subprocess.js';
import { linearSolveDef } from './linear-solve-subprocess.js';

// Demonstrates subprocessStep(): the same LinearSolve definition used
// standalone in linear-solve-subprocess.js and embedded inline in
// adaptive-step-size-control.js is embedded here too, in place, from the
// one shared definition — see docs/superpowers/specs/2026-08-25-subprocess-reuse-design.md.
const solve = subprocessStep(linearSolveDef, undefined, [
  explanationTab(linearSolveDef.box.detail),
  expandTab()
]);

const nodes = [
  startNode({ id: 'start' }),
  processNode({
    id: 'assemble',
    label: 'Assemble linear system',
    detail: 'Placeholder — build the Jacobian and residual for the current iterate.'
  }),
  solve.header,
  ...solve.nodes,
  processNode({ id: 'apply', label: 'Apply update $x \\leftarrow x + \\Delta x$' }),
  endNode({ id: 'end' })
];

const edges = [
  { from: 'start', to: 'assemble' },
  { from: 'assemble', to: solve.entry },
  ...solve.edges,
  { from: solve.exit, to: 'apply' },
  { from: 'apply', to: 'end' }
];

export function mount(container) {
  renderFlowDiagram(container, { nodes, edges });
}
