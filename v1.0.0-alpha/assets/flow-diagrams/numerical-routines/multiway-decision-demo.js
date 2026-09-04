import { renderFlowDiagram } from '../shared/render-flow.js';
import { decisionNode, terminatorNode, groupNode, processNode } from '../shared/components.js';

// Exercises two features together: the decision-diamond bus routing (more
// than two edges off one diamond share a single vertex-to-spine trunk — see
// computeBusRoutes in shared/edge-routing.js) and in-place subprocess
// expand/collapse (shared/group-collapse.js) — each branch off the diamond
// leads into its own independently collapsible subprocess, rather than a
// single leaf node. Diagram authors always wire edges to a group's real
// child ids, never the group id itself; group-collapse.js remaps a
// collapsed group's boundary edges automatically (see its module comment),
// so bus routing sees the same edge count off `norm` regardless of which
// subprocesses happen to be expanded.
const nodes = [
  decisionNode({ id: 'norm', label: 'Which norm?' }),

  groupNode({ id: 'l1_group', label: 'L1', shape: 'predefined-process', children: ['l1_abs', 'l1_sum'] }),
  processNode({ id: 'l1_abs', label: 'Take $|x_i|$ elementwise' }),
  processNode({ id: 'l1_sum', label: 'Sum elements' }),

  groupNode({ id: 'l2_group', label: 'L2', shape: 'predefined-process', children: ['l2_sq', 'l2_sqrt'] }),
  processNode({ id: 'l2_sq', label: 'Sum $x_i^2$' }),
  processNode({ id: 'l2_sqrt', label: 'Take square root' }),

  groupNode({ id: 'linf_group', label: 'L-infinity', shape: 'predefined-process', children: ['linf_abs', 'linf_max'] }),
  processNode({ id: 'linf_abs', label: 'Take $|x_i|$ elementwise' }),
  processNode({ id: 'linf_max', label: 'Take max element' }),

  groupNode({ id: 'frob_group', label: 'Frobenius', shape: 'predefined-process', children: ['frob_sq', 'frob_sqrt'] }),
  processNode({ id: 'frob_sq', label: 'Sum $a_{ij}^2$' }),
  processNode({ id: 'frob_sqrt', label: 'Take square root' }),

  groupNode({ id: 'other_group', label: 'Custom norm', shape: 'predefined-process', children: ['other_eval'] }),
  processNode({ id: 'other_eval', label: 'Evaluate user-supplied norm function' }),

  terminatorNode({ id: 'done', label: 'Return norm value' })
];

const edges = [
  { from: 'norm', to: 'l1_abs' },
  { from: 'l1_abs', to: 'l1_sum' },
  { from: 'l1_sum', to: 'done' },

  { from: 'norm', to: 'l2_sq' },
  { from: 'l2_sq', to: 'l2_sqrt' },
  { from: 'l2_sqrt', to: 'done' },

  { from: 'norm', to: 'linf_abs' },
  { from: 'linf_abs', to: 'linf_max' },
  { from: 'linf_max', to: 'done' },

  { from: 'norm', to: 'frob_sq' },
  { from: 'frob_sq', to: 'frob_sqrt' },
  { from: 'frob_sqrt', to: 'done' },

  { from: 'norm', to: 'other_eval' },
  { from: 'other_eval', to: 'done' }
];

export function mount(container) {
  renderFlowDiagram(container, { nodes, edges });
}
