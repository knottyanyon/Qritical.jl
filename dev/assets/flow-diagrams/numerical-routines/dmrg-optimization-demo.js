import { renderFlowDiagram } from '../shared/render-flow.js';
import {
  startNode, endNode, processNode, decisionNode, branchEdges, explanationTab, expandTab
} from '../shared/components.js';
import { subprocessStep } from '../shared/subprocess.js';
import { dmrgSweepDef } from './dmrg-sweep-subprocess.js';

// A minimal DMRG ground-state search: initialize, sweep to convergence.
// Demonstrates subprocessStep() outside the ISO-shape-showcase context of
// adaptive-step-size-control.js — the DMRG sweep box below expands in place
// to the same flow dmrg-sweep-subprocess.js renders standalone, and the
// "no" branch loops back into the subprocess's own entry point for the next
// sweep, same as looping back to any ordinary node.
const sweep = subprocessStep(dmrgSweepDef, undefined, [
  explanationTab(dmrgSweepDef.box.detail),
  expandTab()
]);

const nodes = [
  startNode({ id: 'start' }),
  processNode({
    id: 'init',
    label: 'Initialize MPS (random or product state)',
    detail: 'Placeholder — the starting matrix product state for the search.'
  }),
  sweep.header,
  ...sweep.nodes,
  decisionNode({
    id: 'converged',
    label: 'Energy converged?',
    detail: 'Placeholder — replace with the real energy-convergence criterion.'
  }),
  endNode({ id: 'end' })
];

const edges = [
  { from: 'start', to: 'init' },
  { from: 'init', to: sweep.entry },
  ...sweep.edges,
  { from: sweep.exit, to: 'converged' },
  ...branchEdges({ from: 'converged', yes: 'end', no: sweep.entry })
];

export function mount(container) {
  renderFlowDiagram(container, { nodes, edges });
}
