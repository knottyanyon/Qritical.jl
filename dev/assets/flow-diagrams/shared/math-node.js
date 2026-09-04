import { renderMath } from './math-render.js';

// Returns { labelEl, ready }: `labelEl` is the flex container the rendered
// label lives in — already attached to `svg` synchronously, so a caller that
// needs to append something next to the label (e.g. render-flow.js's
// collapse-indicator) can do so once `ready` resolves, without racing
// renderMath()'s own `container.innerHTML = ''` reset. `ready` resolves once
// the label has finished rendering; the MathJax fallback path inside
// renderMath() is async, so returning it (rather than firing and forgetting)
// also lets a caller await a label and surface a failed or hung render
// instead of silently swallowing it.
export function createMathForeignObject(svg, { x, y, width, height, label, engine = 'katex' }) {
  const ns = 'http://www.w3.org/2000/svg';
  const fo = document.createElementNS(ns, 'foreignObject');
  fo.setAttribute('x', x - width / 2);
  fo.setAttribute('y', y - height / 2);
  fo.setAttribute('width', width);
  fo.setAttribute('height', height);
  // Labels sit on top of (and are siblings of, not descendants of) the node
  // shape they annotate, so without this the label would silently swallow
  // every click meant for the shape beneath it.
  fo.style.pointerEvents = 'none';

  const div = document.createElement('div');
  div.className = 'math-node-label';
  fo.appendChild(div);
  svg.appendChild(fo);

  return { labelEl: div, ready: renderMath(label, div, { engine }) };
}

// Renders `label` off-screen (same class as a real label, so it picks up the
// same font and the .math-node-label .katex margin rule) to measure its
// natural size before any node/shape geometry is decided. `maxWidth` caps
// wrapping the same way a node's own foreignObject width eventually will,
// so a short label measures at its own width instead of always the cap.
export async function measureLabelSize(label, maxWidth, { engine = 'katex' } = {}) {
  const div = document.createElement('div');
  div.className = 'math-node-label';
  div.style.position = 'fixed';
  div.style.visibility = 'hidden';
  div.style.left = '-9999px';
  div.style.top = '0';
  div.style.display = 'inline-block';
  div.style.width = 'max-content';
  div.style.maxWidth = `${maxWidth}px`;
  div.style.height = 'auto';
  document.body.appendChild(div);
  await renderMath(label, div, { engine });
  const rect = div.getBoundingClientRect();
  document.body.removeChild(div);
  return { width: rect.width, height: rect.height };
}
