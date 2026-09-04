import { roughRect, roughCircle, roughEdge } from './rough-shapes.js';
import { createMathForeignObject } from './math-node.js';
import { attachDetailCard } from './detail-card.js';
import { cssColor } from './palette.js';
import { ensureDocumenterThemeSync } from './documenter-theme-sync.js';

const DEFAULTS = { roughness: 1.1, strokeWidth: 1.6 };
const PADDING = 24;
const LABEL_OFFSET = 14;

function requireNode(nodes, id, what) {
  const n = nodes.find(x => x.id === id);
  if (!n) throw new Error(`renderTensorDiagram: ${what} references unknown node id "${id}"`);
  return n;
}

// Shape functions draw a node centred on (x, y) and report their own half-extent
// (used for the label's foreignObject size and, before anything is drawn, the bbox),
// so extending the registry never needs a second place to teach a shape its size.
export const shapes = {
  box(rc, n, opts) {
    const w = n.w || 40, h = n.h || 40;
    const el = roughRect(rc, n.x, n.y, w, h, {
      fill: n.fill || cssColor('--diagram-fill-process'), fillStyle: 'solid', stroke: cssColor('--diagram-color-outline'),
      strokeWidth: opts.strokeWidth, roughness: opts.roughness
    });
    return { el, halfExtent: [w / 2, h / 2] };
  },
  circle(rc, n, opts) {
    const r = n.r || 18;
    const el = roughCircle(rc, n.x, n.y, r, {
      fill: n.fill || cssColor('--diagram-fill-circle'), fillStyle: 'solid', stroke: cssColor('--diagram-color-outline'),
      strokeWidth: opts.strokeWidth, roughness: opts.roughness
    });
    return { el, halfExtent: [r, r] };
  }
};

function shapeFor(n) {
  return shapes[n.shape] || shapes.circle;
}

// Half-extent for bbox computation, before any node is drawn (mirrors the sizing
// each shape function above reports once it draws).
function nodeHalfExtent(n) {
  if (n.shape === 'box') return [(n.w || 40) / 2, (n.h || 40) / 2];
  const r = n.r || 18;
  return [r, r];
}

// Bonds are skipped deliberately: both endpoints are nodes already in the box.
// Leg labels sit `LABEL_OFFSET` past the leg tip, so the tip is padded for.
function contentBBox(nodes, legs) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const grow = (x, y) => {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  };
  nodes.forEach(n => {
    const [hw, hh] = nodeHalfExtent(n);
    grow(n.x - hw, n.y - hh);
    grow(n.x + hw, n.y + hh);
  });
  legs.forEach(l => {
    const n = requireNode(nodes, l.node, 'leg');
    const rad = (l.angle || 0) * Math.PI / 180;
    const len = (l.length || 35) + (l.label ? LABEL_OFFSET + 13 : 0);
    grow(n.x + Math.cos(rad) * len, n.y + Math.sin(rad) * len);
  });
  if (!Number.isFinite(minX)) return { x: 0, y: 0, width: 1, height: 1 };
  return {
    x: minX - PADDING,
    y: minY - PADDING,
    width: (maxX - minX) + PADDING * 2,
    height: (maxY - minY) + PADDING * 2
  };
}

export function renderTensorDiagram(container, { nodes, bonds = [], legs = [] }, style = {}) {
  const opts = { ...DEFAULTS, ...style };
  container.innerHTML = '';
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'tensor-diagram');
  const bbox = contentBBox(nodes, legs);
  svg.setAttribute('width', bbox.width);
  svg.setAttribute('height', bbox.height);
  svg.setAttribute('viewBox', `${bbox.x} ${bbox.y} ${bbox.width} ${bbox.height}`);
  container.appendChild(svg);
  const rc = rough.svg(svg);

  bonds.forEach(b => {
    const a = requireNode(nodes, b.from, 'bond');
    const c = requireNode(nodes, b.to, 'bond');
    const strokeWidth = b.dim ? Math.min(1 + Math.log2(b.dim), 5) : 1.8;
    svg.appendChild(roughEdge(rc, [[a.x, a.y], [c.x, c.y]], {
      stroke: cssColor('--diagram-color-outline'), strokeWidth, roughness: opts.roughness
    }));
    if (b.label) {
      createMathForeignObject(svg, {
        x: (a.x + c.x) / 2, y: (a.y + c.y) / 2 - 12, width: 30, height: 20, label: b.label
      });
    }
  });

  legs.forEach(l => {
    const n = requireNode(nodes, l.node, 'leg');
    const rad = (l.angle || 0) * Math.PI / 180;
    const len = l.length || 35;
    const x2 = n.x + Math.cos(rad) * len;
    const y2 = n.y + Math.sin(rad) * len;
    svg.appendChild(roughEdge(rc, [[n.x, n.y], [x2, y2]], {
      stroke: cssColor('--diagram-color-outline'), strokeWidth: 1.8, roughness: opts.roughness
    }));
    if (l.label) {
      createMathForeignObject(svg, {
        x: x2 + Math.cos(rad) * LABEL_OFFSET, y: y2 + Math.sin(rad) * LABEL_OFFSET, width: 26, height: 20, label: l.label
      });
    }
  });

  nodes.forEach(n => {
    const { el: shapeEl, halfExtent } = shapeFor(n)(rc, n, opts);
    svg.appendChild(shapeEl);
    if (n.label) {
      createMathForeignObject(svg, {
        x: n.x, y: n.y, width: halfExtent[0] * 2, height: halfExtent[1] * 2, label: n.label
      });
    }
    if (n.detail) attachDetailCard(shapeEl, n.detail);
  });

  ensureDocumenterThemeSync();
  return svg;
}
