import { renderTensorDiagram } from '../shared/render-tensor.js';
import { chainLayout } from '../shared/components.js';

const ORTHOGONALITY_CENTER = 2;

const spec = chainLayout({
  length: 5,
  siteLabel: i => `A_${i}`,
  legLabel: i => `\\sigma_${i}`
});

spec.nodes.forEach((n, i) => {
  const role = i < ORTHOGONALITY_CENTER ? 'left-canonical' : i > ORTHOGONALITY_CENTER ? 'right-canonical' : 'orthogonality center';
  n.fill = i === ORTHOGONALITY_CENTER ? '#F7B7A3' : n.fill;
  n.detail = `<strong>${n.label}</strong> (site ${i}): ${role}.`;
});

export function mount(container) {
  renderTensorDiagram(container, spec);
}
