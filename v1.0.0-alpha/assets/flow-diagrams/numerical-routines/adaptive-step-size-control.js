import { renderFlowDiagram } from '../shared/render-flow.js';
import {
  startNode, endNode, processNode, decisionNode,
  inputOutputNode, preparationNode, documentNode, offPageConnectorNode, connectorNode,
  branchEdges, explanationTab, expandTab
} from '../shared/components.js';
import { subprocessStep } from '../shared/subprocess.js';
import { linearSolveDef } from './linear-solve-subprocess.js';

const solve = subprocessStep(linearSolveDef, undefined, [
  explanationTab(linearSolveDef.box.detail),
  expandTab()
]);

// Exercises every ISO 5807 shape in the `shapes` registry, one node each.
const nodes = [
  startNode({ id: 'start' }),
  preparationNode({
    id: 'prep',
    label: 'Initialize: $k = 0$',
    detail: 'Loop setup — preparation symbol.'
  }),
  inputOutputNode({
    id: 'input',
    label: 'Read $x_0$, $\\Delta_0$',
    detail: 'Placeholder — read the initial guess and step size. Input/output symbol.'
  }),
  processNode({
    id: 'residual',
    label: 'Compute residual $r = f(x)$',
    detail: 'Placeholder — evaluate the residual at the current iterate. Process symbol.'
  }),
  solve.header,
  ...solve.nodes,
  decisionNode({
    id: 'converged',
    label: '$\\|r\\| < \\text{tol}$?',
    detail: 'Placeholder — replace with the real convergence criterion. Decision symbol.'
  }),
  decisionNode({
    id: 'accepted',
    label: 'Step accepted?',
    detail: 'Placeholder — replace with the real step-acceptance test. Decision symbol.'
  }),
  processNode({
    id: 'shrink',
    label: 'Shrink step: $\\Delta \\leftarrow \\Delta / 2$',
    detail: 'Placeholder — halve (or otherwise reduce) the step size and retry. Process symbol.'
  }),
  processNode({
    id: 'take',
    label: 'Take step: $x \\leftarrow x + \\Delta$',
    detail: 'Placeholder — apply the accepted step to the iterate. Process symbol.'
  }),
  connectorNode({
    id: 'iter',
    label: 'A',
    detail: 'On-page connector — loops back into the residual computation. Connector symbol.'
  }),
  documentNode({
    id: 'log',
    label: 'Write log',
    detail: 'Placeholder — record the per-iteration residual/step history. Document symbol.'
  }),
  endNode({ id: 'done' }),
  offPageConnectorNode({
    id: 'offpage',
    label: 'To DMRG sweep',
    detail: 'Continues in the DMRG-sweep-subprocess diagram. Off-page connector symbol.'
  })
];

const edges = [
  { from: 'start', to: 'prep' },
  { from: 'prep', to: 'input' },
  { from: 'input', to: 'residual' },
  { from: 'residual', to: solve.entry },
  ...solve.edges,
  { from: solve.exit, to: 'converged' },
  ...branchEdges({ from: 'converged', yes: 'log', no: 'accepted' }),
  ...branchEdges({ from: 'accepted', yes: 'take', no: 'shrink' }),
  { from: 'shrink', to: 'iter' },
  { from: 'take', to: 'iter' },
  { from: 'iter', to: 'residual' },
  { from: 'log', to: 'done' },
  { from: 'done', to: 'offpage' }
];

export function mount(container) {
  renderFlowDiagram(container, { nodes, edges });
}
