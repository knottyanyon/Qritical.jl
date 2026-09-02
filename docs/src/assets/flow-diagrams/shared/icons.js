// Icon glyphs embedded from Lucide (https://lucide.dev, ISC license) — just
// the paths this package actually uses, copied in directly rather than
// vendoring the whole icon set for a handful of shapes. `play`/`square` are
// single filled shapes (start/stop terminator nodes); `expand`/`shrink`/
// `square-check`/`square-x` are Lucide's original multi-path outline icons,
// kept as strokes rather than filled — flattening a multi-segment glyph into
// a solid fill loses the shape entirely. The `-filled` variants instead draw
// a solid colored badge with the original glyph cut into it in white.
const ICONS = {
  play: { d: 'M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z' },
  square: { rect: { x: 3, y: 3, width: 18, height: 18, rx: 2 } },
  expand: {
    stroke: true,
    paths: ['m15 15 6 6', 'm15 9 6-6', 'M21 16v5h-5', 'M21 8V3h-5', 'M3 16v5h5', 'm3 21 6-6', 'M3 8V3h5', 'M9 9 3 3']
  },
  shrink: {
    stroke: true,
    paths: ['m15 15 6 6m-6-6v4.8m0-4.8h4.8', 'M9 19.8V15m0 0H4.2M9 15l-6 6', 'M15 4.2V9m0 0h4.8M15 9l6-6', 'M9 4.2V9m0 0H4.2M9 9 3 3']
  },
  'square-check': {
    stroke: true,
    rects: [{ x: 3, y: 3, width: 18, height: 18, rx: 2 }],
    paths: ['m9 12 2 2 4-4']
  },
  'square-x': {
    stroke: true,
    rects: [{ x: 3, y: 3, width: 18, height: 18, rx: 2 }],
    paths: ['m15 9-6 6', 'm9 9 6 6']
  },
  // Solid badges (colored square, white glyph cut into it) rather than a thin
  // outline — reads as boldly at a glance as the filled play/square icons on
  // a start/stop node do, for the same reason: a filled shape carries more
  // visual weight than a stroke of the same size.
  'square-check-filled': {
    filled: true,
    rect: { x: 2, y: 2, width: 20, height: 20, rx: 4 },
    paths: ['m8 12.5 2.5 2.5 5-5']
  },
  'square-x-filled': {
    filled: true,
    rect: { x: 2, y: 2, width: 20, height: 20, rx: 4 },
    paths: ['m8.5 8.5 7 7', 'm15.5 8.5-7 7']
  },
  // Used by the file-tab buttons on process/subprocess boxes (see
  // appendNodeTabs in render-flow.js) — 'search-code' for the code-link
  // tab, 'message-circle-question-mark' for the explanation tab;
  // 'expand'/'shrink' above double as the expandable tab's icon.
  'search-code': {
    stroke: true,
    circles: [{ cx: 11, cy: 11, r: 8 }],
    paths: ['M9 8.5 7 11l2 2.5', 'm13 13.5 2-2.5-2-2.5', 'm21 21-4.3-4.3']
  },
  'message-circle-question-mark': {
    stroke: true,
    paths: [
      'M2.992 16.342a2 2 0 0 1 .094 1.167l-1.065 3.29a1 1 0 0 0 1.236 1.168l3.413-.998a2 2 0 0 1 1.099.092 10 10 0 1 0-4.777-4.719',
      'M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3',
      'M12 17h.01'
    ]
  }
};

function buildIconGroup(name, color) {
  const ns = 'http://www.w3.org/2000/svg';
  const icon = ICONS[name];
  const g = document.createElementNS(ns, 'g');
  if (icon.filled) {
    const rect = document.createElementNS(ns, 'rect');
    Object.entries(icon.rect).forEach(([attr, value]) => rect.setAttribute(attr, value));
    rect.setAttribute('fill', color);
    g.appendChild(rect);
    icon.paths.forEach(d => {
      const path = document.createElementNS(ns, 'path');
      path.setAttribute('d', d);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', 'white');
      path.setAttribute('stroke-width', '2');
      path.setAttribute('stroke-linecap', 'round');
      path.setAttribute('stroke-linejoin', 'round');
      g.appendChild(path);
    });
    (icon.circles || []).forEach(({ cx, cy, r }) => {
      const circle = document.createElementNS(ns, 'circle');
      circle.setAttribute('cx', cx);
      circle.setAttribute('cy', cy);
      circle.setAttribute('r', r);
      circle.setAttribute('fill', 'none');
      circle.setAttribute('stroke', 'white');
      circle.setAttribute('stroke-width', '2');
      g.appendChild(circle);
    });
  } else if (icon.stroke) {
    g.setAttribute('fill', 'none');
    g.setAttribute('stroke', color);
    g.setAttribute('stroke-width', '2');
    g.setAttribute('stroke-linecap', 'round');
    g.setAttribute('stroke-linejoin', 'round');
    (icon.rects || []).forEach(attrs => {
      const rect = document.createElementNS(ns, 'rect');
      Object.entries(attrs).forEach(([attr, value]) => rect.setAttribute(attr, value));
      g.appendChild(rect);
    });
    icon.paths.forEach(d => {
      const path = document.createElementNS(ns, 'path');
      path.setAttribute('d', d);
      g.appendChild(path);
    });
    (icon.circles || []).forEach(({ cx, cy, r }) => {
      const circle = document.createElementNS(ns, 'circle');
      circle.setAttribute('cx', cx);
      circle.setAttribute('cy', cy);
      circle.setAttribute('r', r);
      g.appendChild(circle);
    });
  } else {
    g.setAttribute('fill', color);
    if (icon.d) {
      const path = document.createElementNS(ns, 'path');
      path.setAttribute('d', icon.d);
      g.appendChild(path);
    } else if (icon.rect) {
      const rect = document.createElementNS(ns, 'rect');
      Object.entries(icon.rect).forEach(([attr, value]) => rect.setAttribute(attr, value));
      g.appendChild(rect);
    }
  }
  return g;
}

// Appends one icon centered at (x, y) inside an existing SVG graphics
// context (e.g. a node's own <g>), scaled from its native 24x24 viewBox
// down to `size` px.
export function appendIcon(parent, name, { x = 0, y = 0, size = 14, fill = '#222' } = {}) {
  const ns = 'http://www.w3.org/2000/svg';
  const scale = size / 24;
  const wrapper = document.createElementNS(ns, 'g');
  wrapper.setAttribute('transform', `translate(${x - size / 2}, ${y - size / 2}) scale(${scale})`);
  wrapper.style.pointerEvents = 'none';
  wrapper.appendChild(buildIconGroup(name, fill));
  parent.appendChild(wrapper);
  return wrapper;
}

// Returns a standalone, sized <svg> for embedding directly into HTML flow
// layout (e.g. a foreignObject's flex-laid-out label div, which isn't
// itself an SVG graphics context — see appendInlineCollapseIndicator in
// render-flow.js).
export function createIconSvg(name, { size = 16, color = '#222' } = {}) {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('width', size);
  svg.setAttribute('height', size);
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.appendChild(buildIconGroup(name, color));
  return svg;
}
