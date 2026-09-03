import { renderFlowDiagram } from '../shared/render-flow.js';
import {
  startNode, endNode, processNode, decisionNode, groupNode, branchEdges
} from '../shared/components.js';

// Minimal fixture for the in-place expand/collapse feature (shared/group-collapse.js):
// a group of two internal steps sits between two ordinary nodes, and a decision
// after the group branches either back into it (a back edge, testing that the
// side-lane routing still works once the group's members are hidden) or on to
// the end.
const nodes = [
  startNode({ id: 'start' }),
  processNode({ id: 'before', label: 'Prepare input' }),
  groupNode({
    id: 'refine',
    label: 'Refine',
    children: ['refine_a', 'refine_b']
  }),
  processNode({ id: 'refine_a', label: 'Normalize' }),
  processNode({ id: 'refine_b', label: 'Truncate' }),
  decisionNode({ id: 'check', label: 'Converged?' }),
  endNode({ id: 'end' })
];

const edges = [
  { from: 'start', to: 'before' },
  { from: 'before', to: 'refine_a' },
  { from: 'refine_a', to: 'refine_b' },
  { from: 'refine_b', to: 'check' },
  ...branchEdges({ from: 'check', yes: 'end', no: 'refine_a' })
];

export function mount(container) {
  renderFlowDiagram(container, { nodes, edges });
}
