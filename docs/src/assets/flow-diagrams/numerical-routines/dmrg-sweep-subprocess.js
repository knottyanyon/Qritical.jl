import { defineSubprocess, mountSubprocessStandalone, subprocessStep } from '../shared/subprocess.js';
import { decisionNode, processNode, branchEdges, explanationTab, expandTab } from '../shared/components.js';
import { svdTruncationDef } from './svd-truncation-decision.js';

// The subroutine behind adaptive-step-size-control.js's "To DMRG sweep"
// off-page connector — one full sweep over the MPS chain, embedded as a
// step in dmrg-optimization-demo.js via subprocessStep(). Its own
// truncation step is itself an embedded subprocess (svd-truncation-decision.js)
// — a subprocess nested inside a subprocess, exactly the composition the
// design doc's nesting note describes.
const truncate = subprocessStep(svdTruncationDef, undefined, [
  explanationTab(svdTruncationDef.box.detail),
  expandTab()
]);

export const dmrgSweepDef = defineSubprocess({
  id: 'dmrg-sweep',
  box: {
    label: 'DMRG sweep',
    detail: 'One full left-to-right or right-to-left sweep: locally optimize each site tensor, then truncate via SVD.'
  },
  entry: 'direction',
  exit: truncate.exit,
  nodes: [
    decisionNode({
      id: 'direction',
      label: 'Left-to-right sweep?',
      detail: 'Placeholder — alternates direction each sweep in a standard DMRG schedule.'
    }),
    processNode({
      id: 'optimize-ltr',
      label: 'Optimize site tensors left$\\to$right',
      detail: 'Placeholder — local eigensolver update at each site, sweeping forward.'
    }),
    processNode({
      id: 'optimize-rtl',
      label: 'Optimize site tensors right$\\to$left',
      detail: 'Placeholder — local eigensolver update at each site, sweeping backward.'
    }),
    truncate.header,
    ...truncate.nodes
  ],
  edges: [
    ...branchEdges({ from: 'direction', yes: 'optimize-ltr', no: 'optimize-rtl' }),
    { from: 'optimize-ltr', to: truncate.entry },
    { from: 'optimize-rtl', to: truncate.entry },
    ...truncate.edges
  ]
});

export function mount(container) {
  mountSubprocessStandalone(dmrgSweepDef, container);
}
