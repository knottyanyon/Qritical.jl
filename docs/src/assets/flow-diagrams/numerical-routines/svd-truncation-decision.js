import { defineSubprocess, mountSubprocessStandalone } from '../shared/subprocess.js';
import { processNode, decisionNode, branchEdges } from '../shared/components.js';

// The SVD-and-truncate step behind dmrg-sweep-subprocess.js's "Truncate
// $\Sigma$ via SVD" box — embedded there via subprocessStep().
export const svdTruncationDef = defineSubprocess({
  id: 'svd-truncation',
  box: {
    label: 'Truncate $\\Sigma$ via SVD',
    detail: 'Computes $M = U \\Sigma V^\\dagger$ and truncates $\\Sigma$ if the truncation criterion is met.'
  },
  entry: 'compute',
  exit: 'ready',
  nodes: [
    processNode({
      id: 'compute',
      label: 'Compute SVD: $M = U \\Sigma V^\\dagger$',
      detail: 'Singular value decomposition of the merged two-site tensor.'
    }),
    decisionNode({
      id: 'check',
      label: 'Truncation criterion met?',
      detail: 'Placeholder — replace with the real truncation criteria used in Qritical.jl.'
    }),
    processNode({
      id: 'truncate',
      label: 'Truncate $\\Sigma$, renormalize',
      detail: 'Placeholder — replace with the real truncation/renormalization detail.'
    }),
    processNode({
      id: 'keep',
      label: 'Keep full $\\Sigma$',
      detail: 'Placeholder — replace with the real no-truncation-needed detail.'
    }),
    processNode({
      id: 'ready',
      label: '$\\Sigma$ ready for use',
      detail: 'Either branch converges here — the (possibly truncated) $\\Sigma$ is ready for the calling routine.'
    })
  ],
  edges: [
    { from: 'compute', to: 'check' },
    ...branchEdges({ from: 'check', yes: 'keep', no: 'truncate' }),
    { from: 'keep', to: 'ready' },
    { from: 'truncate', to: 'ready' }
  ]
});

export function mount(container) {
  mountSubprocessStandalone(svdTruncationDef, container);
}
