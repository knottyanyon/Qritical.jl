export function roughRect(rc, x, y, w, h, options = {}) {
  return rc.rectangle(x - w / 2, y - h / 2, w, h, options);
}

export function roughCircle(rc, x, y, r, options = {}) {
  return rc.circle(x, y, r * 2, options);
}

export function roughDiamond(rc, x, y, w, h, options = {}) {
  const hw = w / 2, hh = h / 2;
  const d = `M ${x} ${y - hh} L ${x + hw} ${y} L ${x} ${y + hh} L ${x - hw} ${y} Z`;
  return rc.path(d, options);
}

export function roughRoundedRect(rc, x, y, w, h, radius, options = {}) {
  const x0 = x - w / 2, y0 = y - h / 2;
  const r = Math.min(radius, w / 2, h / 2);
  const d = `M ${x0 + r} ${y0} ` +
    `L ${x0 + w - r} ${y0} A ${r} ${r} 0 0 1 ${x0 + w} ${y0 + r} ` +
    `L ${x0 + w} ${y0 + h - r} A ${r} ${r} 0 0 1 ${x0 + w - r} ${y0 + h} ` +
    `L ${x0 + r} ${y0 + h} A ${r} ${r} 0 0 1 ${x0} ${y0 + h - r} ` +
    `L ${x0} ${y0 + r} A ${r} ${r} 0 0 1 ${x0 + r} ${y0} Z`;
  return rc.path(d, options);
}

// ISO 5807 terminator: a "stadium" — a rectangle capped with semicircles on
// the short ends. Used for flowchart start/end nodes.
export function roughStadium(rc, x, y, w, h, options = {}) {
  const x0 = x - w / 2, y0 = y - h / 2;
  const r = h / 2;
  const d = `M ${x0 + r} ${y0} ` +
    `L ${x0 + w - r} ${y0} A ${r} ${r} 0 0 1 ${x0 + w - r} ${y0 + h} ` +
    `L ${x0 + r} ${y0 + h} A ${r} ${r} 0 0 1 ${x0 + r} ${y0} Z`;
  return rc.path(d, options);
}

// ISO 5807 input/output ("data"): a parallelogram, slanted along the top/bottom.
export function roughParallelogram(rc, x, y, w, h, options = {}) {
  const hw = w / 2, hh = h / 2, skew = w * 0.2;
  const d = `M ${x - hw + skew} ${y - hh} L ${x + hw} ${y - hh} ` +
    `L ${x + hw - skew} ${y + hh} L ${x - hw} ${y + hh} Z`;
  return rc.path(d, options);
}

// ISO 5807 preparation: a hexagon with pointed left/right ends.
export function roughHexagon(rc, x, y, w, h, options = {}) {
  const hw = w / 2, hh = h / 2, notch = w * 0.2;
  const d = `M ${x - hw + notch} ${y - hh} L ${x + hw - notch} ${y - hh} ` +
    `L ${x + hw} ${y} L ${x + hw - notch} ${y + hh} ` +
    `L ${x - hw + notch} ${y + hh} L ${x - hw} ${y} Z`;
  return rc.path(d, options);
}

// ISO 5807 document: a rectangle with a wavy bottom edge.
export function roughDocument(rc, x, y, w, h, options = {}) {
  const x0 = x - w / 2, y0 = y - h / 2, x1 = x + w / 2, y1 = y + h / 2;
  const waveDip = h * 0.12;
  const d = `M ${x0} ${y0} L ${x1} ${y0} L ${x1} ${y1 - waveDip} ` +
    `Q ${x0 + w * 0.75} ${y1 + waveDip}, ${x0 + w * 0.5} ${y1 - waveDip} ` +
    `Q ${x0 + w * 0.25} ${y1 - 3 * waveDip}, ${x0} ${y1 - waveDip} Z`;
  return rc.path(d, options);
}

// ISO 5807 off-page connector: a "home plate" pentagon, pointed at the bottom.
export function roughPentagon(rc, x, y, w, h, options = {}) {
  const hw = w / 2, hh = h / 2, pointDrop = h * 0.35;
  const d = `M ${x - hw} ${y - hh} L ${x + hw} ${y - hh} ` +
    `L ${x + hw} ${y + hh - pointDrop} L ${x} ${y + hh} ` +
    `L ${x - hw} ${y + hh - pointDrop} Z`;
  return rc.path(d, options);
}

// ISO 5807 predefined process (subroutine): a rectangle with a double vertical
// bar just inside each side. Compound shape, so it's drawn as a <g> group.
export function roughPredefinedProcess(rc, x, y, w, h, options = {}) {
  const ns = 'http://www.w3.org/2000/svg';
  const g = document.createElementNS(ns, 'g');
  const x0 = x - w / 2, y0 = y - h / 2;
  const barInset = Math.min(12, w * 0.12);
  g.appendChild(rc.rectangle(x0, y0, w, h, options));
  const barOptions = { stroke: options.stroke, strokeWidth: options.strokeWidth, roughness: options.roughness };
  g.appendChild(rc.line(x0 + barInset, y0, x0 + barInset, y0 + h, barOptions));
  g.appendChild(rc.line(x0 + w - barInset, y0, x0 + w - barInset, y0 + h, barOptions));
  return g;
}

export function roughEdge(rc, points, options = {}) {
  if (points.length === 2) {
    const [[x1, y1], [x2, y2]] = points;
    return rc.line(x1, y1, x2, y2, options);
  }
  const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p[0]} ${p[1]}`).join(' ');
  return rc.path(d, options);
}
