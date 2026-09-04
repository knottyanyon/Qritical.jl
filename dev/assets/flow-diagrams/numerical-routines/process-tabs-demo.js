import { renderFlowDiagram } from '../shared/render-flow.js';
import {
  startNode, endNode, processNode,
  explanationTab, codeTab, expandTab
} from '../shared/components.js';
import { subprocessStep } from '../shared/subprocess.js';
import { linearSolveDef } from './linear-solve-subprocess.js';

// Demonstrates the optional file-tab buttons (top-right corner) on process
// and subprocess boxes, in increasing counts: one tab, two tabs, and all
// three (explanation, code link, expand/collapse) on the LinearSolve
// subprocess — the same shared definition used in
// adaptive-step-size-control.js / subprocess-inline-expand-demo.js — where
// expandTab() moves the fold/unfold trigger off the whole-box click and onto
// its own button. expandTab() only makes sense on a node with `children`, so
// it's used here on the subprocess header, not on the plain process steps.
const solve = subprocessStep(linearSolveDef, 'solve', [
  explanationTab(linearSolveDef.box.detail),
  codeTab({ href: 'https://example.com/docs/linear-solve' }),
  expandTab()
]);

const nodes = [
  startNode({ id: 'start' }),
  processNode({
    id: 'assemble',
    label: 'Assemble linear system',
    tabs: [explanationTab('Build the Jacobian and residual for the current iterate.')]
  }),
  solve.header,
  ...solve.nodes,
  processNode({
    id: 'apply',
    label: 'Apply update $x \\leftarrow x + \\Delta x$',
    tabs: [
      explanationTab('Placeholder — apply the Newton update to the current iterate.'),
      codeTab({ href: 'https://example.com/docs/apply' })
    ]
  }),
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
