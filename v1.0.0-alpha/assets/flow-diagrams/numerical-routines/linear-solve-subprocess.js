import { defineSubprocess, mountSubprocessStandalone } from '../shared/subprocess.js';
import { processNode, decisionNode, branchEdges } from '../shared/components.js';

// The subroutine behind the `LinearSolve` predefined-process box in
// adaptive-step-size-control.js — embedded there via subprocessStep().
export const linearSolveDef = defineSubprocess({
  id: 'linear-solve',
  box: {
    label: 'LinearSolve',
    detail: 'Solves $J \\Delta x = -r$ for the Newton update step.'
  },
  entry: 'sparse',
  exit: 'back-substitute',
  nodes: [
    decisionNode({
      id: 'sparse',
      label: '$J$ sparse?',
      detail: 'Placeholder — dispatch on the Jacobian\'s structure.'
    }),
    processNode({
      id: 'factorize-sparse',
      label: 'Sparse $LU$ factorize $J$',
      detail: 'Placeholder — sparse-direct or iterative factorization path.'
    }),
    processNode({
      id: 'factorize-dense',
      label: 'Dense $LU$ factorize $J$',
      detail: 'Placeholder — dense factorization path for small/dense Jacobians.'
    }),
    processNode({
      id: 'back-substitute',
      label: 'Back-substitute for $\\Delta x$',
      detail: 'Placeholder — solve the factored system for the update step.'
    })
  ],
  edges: [
    ...branchEdges({ from: 'sparse', yes: 'factorize-sparse', no: 'factorize-dense' }),
    { from: 'factorize-sparse', to: 'back-substitute' },
    { from: 'factorize-dense', to: 'back-substitute' }
  ]
});

export function mount(container) {
  mountSubprocessStandalone(linearSolveDef, container);
}
