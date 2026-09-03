import {
  roughRect, roughRoundedRect, roughDiamond, roughCircle,
  roughStadium, roughParallelogram, roughHexagon, roughDocument, roughPentagon, roughPredefinedProcess,
  roughEdge
} from './rough-shapes.js';
import { createMathForeignObject, measureLabelSize } from './math-node.js';
import { attachDetailCard } from './detail-card.js';
import { appendIcon, createIconSvg } from './icons.js';
import { orthogonalEdgePoints, isBackEdge, isRankSkipping, rankLevels, sideLaneEdgePoints, computeBusRoutes } from './edge-routing.js';
import { computeEffectiveGraph } from './group-collapse.js';
import { cssColor } from './palette.js';
import { ensureDocumenterThemeSync } from './documenter-theme-sync.js';

const DEFAULTS = { roughness: 0.6, strokeWidth: 1.6, nodeWidth: 160, nodeHeight: 56 };

// Matches the `- 12` inset the real label foreignObject is drawn with
// (see the node.width - 12 / node.height - 12 call below).
const LABEL_INSET = 12;

// Shapes whose slanted/pointed sides eat into their own bounding box need
// extra width beyond a plain content+padding rectangle, or a label that just
// fits a box shape would run into the slant. Expressed as extra width per
// unit of content height (roughly tan of the shape's corner angle).
const SHAPE_WIDTH_SLANT = { 'input-output': 0.4, preparation: 0.3, 'offpage-connector': 0.18 };

// Grows a node's box to fit its label's measured content size, respecting
// whatever width/height the node (or the shared default) already asked for
// as a floor — this only ever grows a node, never shrinks one below its
// author-requested or default size.
function fitNodeToContent(shape, contentW, contentH, baseW, baseH) {
  if (shape === 'diamond' || shape === 'decision') {
    // A w×h rectangle inscribes in a rhombus of bounding box W×H iff
    // w/W + h/H <= 1, so solve for the smallest uniform scale of the
    // node's own aspect ratio that satisfies that with a little slack.
    const MARGIN = 1.15;
    const s = Math.max(1, (contentW / baseW + contentH / baseH) * MARGIN);
    return { width: baseW * s, height: baseH * s };
  }
  if (shape === 'circle' || shape === 'connector') {
    // A w×h rectangle inscribes in a circle of diameter D iff
    // D >= sqrt(w^2 + h^2).
    const MARGIN = 1.15;
    const d = Math.max(Math.min(baseW, baseH), Math.sqrt(contentW ** 2 + contentH ** 2) * MARGIN);
    return { width: d, height: d };
  }
  const slant = SHAPE_WIDTH_SLANT[shape] || 0;
  const height = Math.max(baseH, contentH + LABEL_INSET);
  const width = Math.max(baseW, contentW + LABEL_INSET + slant * height);
  return { width, height };
}

// How long a collapse/expand relayout's node-move and edge-crossfade
// transitions run for — must match the durations in flow-diagram.css.
const RELAYOUT_MS = 250;

function shapeFillOptions(node, opts, fallbackFill) {
  return {
    fill: node.fill || fallbackFill,
    fillStyle: 'solid',
    stroke: cssColor('--diagram-color-outline'),
    strokeWidth: opts.strokeWidth,
    roughness: opts.roughness,
    ...(node.style || {})
  };
}

// Shape registry. `box`/`diamond`/`circle` are the generic primitives; the rest
// implement the ISO 5807 flowchart symbol set (terminator, predefined process,
// input/output, preparation, document, off-page connector), plus `group` for
// a collapsible-subtree header (see shared/group-collapse.js).
export const shapes = {
  box(rc, node, opts) {
    return roughRect(rc, node.x, node.y, node.width, node.height, shapeFillOptions(node, opts, cssColor('--diagram-fill-process')));
  },
  'rounded-box'(rc, node, opts) {
    return roughRoundedRect(rc, node.x, node.y, node.width, node.height, node.radius || 12, shapeFillOptions(node, opts, cssColor('--diagram-fill-process')));
  },
  diamond(rc, node, opts) {
    return roughDiamond(rc, node.x, node.y, node.width, node.height, shapeFillOptions(node, opts, cssColor('--diagram-fill-decision')));
  },
  circle(rc, node, opts) {
    const r = Math.min(node.width, node.height) / 2;
    return roughCircle(rc, node.x, node.y, r, shapeFillOptions(node, opts, cssColor('--diagram-fill-circle')));
  },
  // ISO 5807: terminator (start/end).
  terminator(rc, node, opts) {
    return roughStadium(rc, node.x, node.y, node.width, node.height, shapeFillOptions(node, opts, cssColor('--diagram-fill-process')));
  },
  // ISO 5807: process — same symbol as `box`.
  process(rc, node, opts) {
    return shapes.box(rc, node, opts);
  },
  // ISO 5807: predefined process (subroutine/call-out).
  'predefined-process'(rc, node, opts) {
    return roughPredefinedProcess(rc, node.x, node.y, node.width, node.height, shapeFillOptions(node, opts, cssColor('--diagram-fill-process')));
  },
  // ISO 5807: decision — same symbol as `diamond`.
  decision(rc, node, opts) {
    return shapes.diamond(rc, node, opts);
  },
  // ISO 5807: input/output (data).
  'input-output'(rc, node, opts) {
    return roughParallelogram(rc, node.x, node.y, node.width, node.height, shapeFillOptions(node, opts, cssColor('--diagram-fill-io')));
  },
  // ISO 5807: preparation.
  preparation(rc, node, opts) {
    return roughHexagon(rc, node.x, node.y, node.width, node.height, shapeFillOptions(node, opts, cssColor('--diagram-fill-preparation')));
  },
  // ISO 5807: document.
  document(rc, node, opts) {
    return roughDocument(rc, node.x, node.y, node.width, node.height, shapeFillOptions(node, opts, cssColor('--diagram-fill-document')));
  },
  // ISO 5807: off-page connector.
  'offpage-connector'(rc, node, opts) {
    return roughPentagon(rc, node.x, node.y, node.width, node.height, shapeFillOptions(node, opts, cssColor('--diagram-fill-offpage')));
  },
  // ISO 5807: (on-page) connector — same symbol as `circle`.
  connector(rc, node, opts) {
    return shapes.circle(rc, node, opts);
  },
  // Collapsible-group header: a dashed rounded box, click-toggled — see
  // shared/group-collapse.js for how its children/edges fold in and out.
  group(rc, node, opts) {
    const fill = shapeFillOptions(node, opts, cssColor('--diagram-fill-group'));
    return roughRoundedRect(rc, node.x, node.y, node.width, node.height, node.radius || 10, {
      ...fill,
      strokeLineDash: fill.strokeLineDash || [6, 4]
    });
  }
};

// A collapse/expand disclosure indicator for a group node — Lucide's
// expand/shrink icon pair (see shared/icons.js), appended as a sibling
// inside the label's own flex container (see math-node.js) right after its
// text, rather than positioned by hand — so it naturally sits with a little
// breathing room right next to the label, vertically centered with it,
// wherever that text happens to end.
function appendInlineCollapseIndicator(labelDiv, collapsed) {
  const svg = createIconSvg(collapsed ? 'expand' : 'shrink', { size: 16, color: cssColor('--diagram-color-outline') });
  svg.setAttribute('class', 'flow-collapse-indicator');
  labelDiv.appendChild(svg);
}

// File-tab buttons: small clickable rectangles poking above a box's
// top-right corner (see explanationTab/codeTab/expandTab in
// shared/components.js). When a node has `tabs`, they replace the
// whole-box click entirely — the box itself keeps only its hover
// highlight (applied by the caller) — so each tab is its own click
// target here. Tabs lay out right-to-left from the box's right edge, so
// `node.tabs[0]` sits rightmost.
const TAB_W = 22, TAB_H = 16, TAB_GAP = 4, TAB_OVERLAP = 4, TAB_RIGHT_MARGIN = 10;

function tabIconName(tab, collapsed) {
  if (tab.kind === 'explanation') return 'message-circle-question-mark';
  if (tab.kind === 'code') return 'search-code';
  if (tab.kind === 'expand') return collapsed ? 'expand' : 'shrink';
  return 'message-circle-question-mark';
}

// Each tab kind gets its own fill, distinct from the process box it sits on
// (--diagram-fill-process) so the tabs read as controls rather than part of
// the box itself — 'expand' reuses --diagram-fill-group, tying it to the
// same subprocess/group concept its click controls.
function tabFill(tab) {
  if (tab.kind === 'code') return cssColor('--diagram-fill-tab-code');
  if (tab.kind === 'expand') return cssColor('--diagram-fill-group');
  return cssColor('--diagram-fill-tab-explanation');
}

function appendNodeTabs(rc, el, node, opts, { isCollapsed, toggleCollapse }) {
  node.tabs.forEach((tab, i) => {
    const x = node.width / 2 - TAB_RIGHT_MARGIN - TAB_W / 2 - i * (TAB_W + TAB_GAP);
    const y = -node.height / 2 - TAB_H / 2 + TAB_OVERLAP;
    // Pass a fill-less node (keeping only `style`) so a node's own custom
    // fill/style overrides don't leak into the tab's intentionally distinct
    // color, while author overrides like a custom stroke dash still do.
    const tabEl = roughRect(rc, x, y, TAB_W, TAB_H, shapeFillOptions({ style: node.style }, opts, tabFill(tab)));
    tabEl.classList.add('diagram-node--interactive');
    tabEl.style.cursor = 'pointer';
    el.appendChild(tabEl);
    appendIcon(el, tabIconName(tab, isCollapsed), { x, y, size: 12, fill: cssColor('--diagram-color-outline') });

    if (tab.kind === 'expand') {
      tabEl.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleCollapse();
      });
    } else {
      attachDetailCard(tabEl, tab.detail);
    }
  });
}

// Marker ids must be unique document-wide: `url(#id)` resolves against the whole
// document, so two flow diagrams on one page sharing a hardcoded id would make the
// second diagram's edges point at the first diagram's marker.
let markerSeq = 0;

export const arrows = {
  // Returns the generated marker id so the caller can reference this exact marker.
  // `color` lets a styled edge (e.g. a green/red decision branch) get an
  // arrowhead that matches its line instead of always drawing in black.
  normal(defs, color = cssColor('--diagram-color-outline')) {
    const ns = 'http://www.w3.org/2000/svg';
    const id = `arrow-normal-${++markerSeq}`;
    const marker = document.createElementNS(ns, 'marker');
    marker.setAttribute('id', id);
    marker.setAttribute('viewBox', '0 0 10 10');
    marker.setAttribute('refX', '9');
    marker.setAttribute('refY', '5');
    marker.setAttribute('markerWidth', '8');
    marker.setAttribute('markerHeight', '8');
    marker.setAttribute('orient', 'auto-start-reverse');
    const path = document.createElementNS(ns, 'path');
    path.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z');
    path.setAttribute('fill', color);
    marker.appendChild(path);
    defs.appendChild(marker);
    return id;
  }
};

export function renderFlowDiagram(container, { nodes: allNodes, edges: allEdges }, style = {}) {
  // Fail fast on duplicate node ids instead of letting dagre silently merge
  // them into one node — this is the real point at which a repeated
  // subprocessStep() embedding (see shared/subprocess.js) would otherwise
  // produce a wrong-but-plausible-looking diagram with no error.
  const seenIds = new Set();
  for (const n of allNodes) {
    if (seenIds.has(n.id)) {
      throw new Error(`renderFlowDiagram: duplicate node id "${n.id}"`);
    }
    seenIds.add(n.id);
  }
  const opts = { ...DEFAULTS, ...style };
  const outlineColor = cssColor('--diagram-color-outline');
  const yesColor = cssColor('--diagram-stroke-yes');
  const noColor = cssColor('--diagram-stroke-no');
  const collapsedGroupIds = new Set(
    allNodes.filter(n => n.children && n.collapsed !== false).map(n => n.id)
  );

  container.innerHTML = '';
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('class', 'flow-diagram');
  container.appendChild(svg);

  const defs = document.createElementNS(ns, 'defs');
  svg.appendChild(defs);
  const defaultArrowId = arrows.normal(defs);
  // One marker per distinct stroke color actually used, created lazily so the
  // common (uncolored) case still pays for only a single marker.
  const arrowIdByColor = new Map();
  const rc = rough.svg(svg);

  // Edges live in one persistent layer that's crossfaded wholesale on
  // relayout (rough.js's hand-drawn wobble means a redrawn edge never has a
  // stable path to tween against its predecessor). Nodes live in another,
  // where each node keeps its own <g> across relayouts so its position can
  // be tweened instead of replaced.
  const edgesLayer = document.createElementNS(ns, 'g');
  edgesLayer.setAttribute('class', 'flow-edges-layer');
  const nodesLayer = document.createElementNS(ns, 'g');
  nodesLayer.setAttribute('class', 'flow-nodes-layer');
  svg.appendChild(edgesLayer);
  svg.appendChild(nodesLayer);

  const nodeEls = new Map(); // node id -> its persistent <g class="flow-node">
  let positions = new Map(); // node id -> {x, y} from the most recent layout

  function arrowFor(stroke) {
    if (stroke === outlineColor) return defaultArrowId;
    let id = arrowIdByColor.get(stroke);
    if (!id) {
      id = arrows.normal(defs, stroke);
      arrowIdByColor.set(stroke, id);
    }
    return id;
  }

  async function layoutAndRender({ animate }) {
    const { nodes, edges } = computeEffectiveGraph(allNodes, allEdges, collapsedGroupIds);

    // Measure each node's label before deciding its box — a node's declared
    // width/height (or the shared default) is only ever a floor from here on;
    // fitNodeToContent grows it to whatever the label actually needs, using
    // the shape's real geometry (a diamond/circle need a bigger bounding box
    // than a rectangle would for the same content — see fitNodeToContent).
    const sizedNodes = await Promise.all(nodes.map(async n => {
      const baseW = n.width || opts.nodeWidth, baseH = n.height || opts.nodeHeight;
      if (!n.label) return { ...n, width: baseW, height: baseH };
      const { width: contentW, height: contentH } = await measureLabelSize(n.label, baseW - LABEL_INSET);
      const { width, height } = fitNodeToContent(n.shape, contentW, contentH, baseW, baseH);
      return { ...n, width, height };
    }));

    const g = new dagre.graphlib.Graph();
    g.setGraph({});
    g.setDefaultEdgeLabel(() => ({}));
    sizedNodes.forEach(n => g.setNode(n.id, { width: n.width, height: n.height, ...n }));
    edges.forEach(e => g.setEdge(e.from, e.to, e));
    dagre.layout(g);

    // Any edge that can't be trusted to route straight through the diagram
    // without risking an overlap — a back edge (loop closing to an earlier
    // step) or a forward edge that skips over an intervening rank — gets
    // routed through a dedicated vertical lane clear of every node instead.
    // See isBackEdge/isRankSkipping/sideLaneEdgePoints in edge-routing.js.
    // Each such edge gets its own lane so parallel ones don't overlap either.
    const nodeList = g.nodes().map(id => g.node(id));
    const maxNodeRight = nodeList.length ? Math.max(...nodeList.map(n => n.x + n.width / 2)) : 0;
    const levels = rankLevels(nodeList);
    const LANE_MARGIN = 40, LANE_SPACING = 22;
    const laneXByEdgeKey = new Map();
    g.edges().forEach(e => {
      const fromNode = g.node(e.v), toNode = g.node(e.w);
      if (isBackEdge(fromNode, toNode) || isRankSkipping(fromNode, toNode, levels)) {
        laneXByEdgeKey.set(`${e.v}->${e.w}`, maxNodeRight + LANE_MARGIN + laneXByEdgeKey.size * LANE_SPACING);
      }
    });

    const graphInfo = g.graph();
    const svgWidth = laneXByEdgeKey.size
      ? Math.max(graphInfo.width, maxNodeRight + LANE_MARGIN + laneXByEdgeKey.size * LANE_SPACING + LANE_MARGIN)
      : graphInfo.width;
    svg.setAttribute('width', svgWidth);
    svg.setAttribute('height', graphInfo.height);

    // --- edges: fully redrawn every time, crossfaded as one layer -----------
    const oldEdgesLayer = animate && edgesLayer.hasChildNodes() ? edgesLayer.cloneNode(true) : null;
    edgesLayer.innerHTML = '';

    // A decision node with more than two same-direction edges gets a shared
    // vertex-to-spine trunk instead of each edge fanning out from its own
    // independently-chosen point, and the mirror case — more than two edges
    // converging on one node — merges onto a shared spine before a single
    // trunk reaches it — see computeBusRoutes in edge-routing.js. The
    // trunk/spine are drawn once, before any individual edge, so they sit as
    // the structural line branches hang off; only the converging trunk's
    // final spine-to-node stem carries an arrowhead (arrowedTrunks) — every
    // other trunk segment has flow continuing past it, not ending there.
    const busRoutes = computeBusRoutes(g);
    busRoutes.trunks.forEach(points => {
      edgesLayer.appendChild(roughEdge(rc, points, { stroke: outlineColor, strokeWidth: opts.strokeWidth, roughness: opts.roughness }));
    });
    busRoutes.arrowedTrunks.forEach(points => {
      const path = roughEdge(rc, points, { stroke: outlineColor, strokeWidth: opts.strokeWidth, roughness: opts.roughness });
      path.setAttribute('marker-end', `url(#${arrowFor(outlineColor)})`);
      edgesLayer.appendChild(path);
    });

    g.edges().forEach(e => {
      const edgeData = g.edge(e);
      const edgeKey = `${e.v}->${e.w}`;
      // Route independently of dagre's own (diagonal-capable) edge points —
      // only the node positions/sizes it computed are used, so every edge is
      // a straight horizontal/vertical segment or a right-angle elbow, and a
      // back edge gets its own clear lane instead of cutting through the
      // diagram (see the laneXByEdgeKey setup above).
      const laneX = laneXByEdgeKey.get(edgeKey);
      const busDrop = busRoutes.drops.get(edgeKey);
      const points = busDrop
        ? busDrop.points
        : (laneX !== undefined
          ? sideLaneEdgePoints(g.node(e.v), g.node(e.w), laneX)
          : orthogonalEdgePoints(g.node(e.v), g.node(e.w)));
      const stroke = (edgeData.style && edgeData.style.stroke) || outlineColor;
      const path = roughEdge(rc, points, {
        stroke, strokeWidth: opts.strokeWidth, roughness: opts.roughness, ...(edgeData.style || {})
      });
      if (!busDrop || busDrop.arrow) {
        path.setAttribute('marker-end', `url(#${arrowFor(stroke)})`);
      }
      edgesLayer.appendChild(path);
      if (edgeData.label) {
        // Straight edges label at the midpoint; an elbow labels at its bend,
        // which sits right where a branch leaves its decision/circle node.
        const mid = points.length === 2
          ? [(points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2]
          : points[1];
        // Offset the label off to the side of its segment rather than
        // straddling the line itself: above/below for a horizontal run,
        // left/right for a vertical one — inferred from the segment the
        // label actually sits on (the first one, i.e. how the edge leaves
        // its source, which is what a reader's eye follows right after the
        // decision).
        const [x1, y1] = points[0], [x2, y2] = points[1];
        const horizontal = Math.abs(x2 - x1) >= Math.abs(y2 - y1);
        const LABEL_OFFSET = 12;
        const lx = horizontal ? mid[0] : mid[0] + LABEL_OFFSET;
        const ly = horizontal ? mid[1] - LABEL_OFFSET : mid[1];
        // A decision's yes/no branch (see decisionEdge/branchEdges in
        // components.js) reads its outcome from a check/x icon in the
        // branch's own color instead of text — a colored branch that isn't
        // yes/no keeps the uppercased colored-text label, and an uncolored
        // edge stays the plain diagram body style.
        if (stroke === yesColor || stroke === noColor) {
          appendIcon(edgesLayer, stroke === yesColor ? 'square-check-filled' : 'square-x-filled', { x: lx, y: ly, size: 24, fill: stroke });
        } else {
          const { labelEl } = createMathForeignObject(edgesLayer, { x: lx, y: ly, width: 60, height: 20, label: edgeData.label });
          if (stroke !== outlineColor) {
            labelEl.style.color = stroke;
            labelEl.style.textTransform = 'uppercase';
            labelEl.style.fontWeight = '700';
          }
        }
      }
    });

    if (animate) {
      if (oldEdgesLayer) {
        svg.insertBefore(oldEdgesLayer, edgesLayer);
        requestAnimationFrame(() => { oldEdgesLayer.style.opacity = '0'; });
        setTimeout(() => oldEdgesLayer.remove(), RELAYOUT_MS);
      }
      edgesLayer.style.transition = 'none';
      edgesLayer.style.opacity = '0';
      edgesLayer.getBoundingClientRect(); // commit opacity:0 before re-enabling the transition
      edgesLayer.style.transition = '';
      requestAnimationFrame(() => { edgesLayer.style.opacity = '1'; });
    } else {
      edgesLayer.style.opacity = '1';
    }

    // --- nodes: reuse each id's <g> across relayouts, tweening its position -
    const seenIds = new Set();
    g.nodes().forEach(id => {
      seenIds.add(id);
      const node = g.node(id);
      let el = nodeEls.get(id);
      const isNewNode = !el;
      if (isNewNode) {
        el = document.createElementNS(ns, 'g');
        el.setAttribute('class', 'flow-node');
        el.dataset.nodeId = id;
        nodeEls.set(id, el);
        nodesLayer.appendChild(el);
      } else {
        el.innerHTML = ''; // rough.js redraws with a fresh hand-drawn wobble every time
      }

      // Drawn at the node's own local origin so only this <g>'s transform
      // (tweenable) carries its position — the shape/label never need to
      // know where they sit in the diagram.
      const drawShape = shapes[node.shape] || shapes.box;
      const shapeEl = drawShape(rc, { ...node, x: 0, y: 0 }, opts);
      el.appendChild(shapeEl);
      if (node.label) {
        const { labelEl, ready } = createMathForeignObject(el, {
          x: 0, y: 0, width: node.width - 12, height: node.height - 12, label: node.label
        });
        const hasExpandTab = node.tabs && node.tabs.some(t => t.kind === 'expand');
        if (node.children && !hasExpandTab) {
          const collapsed = collapsedGroupIds.has(id);
          ready.then(() => appendInlineCollapseIndicator(labelEl, collapsed));
        }
      }
      if (node.variant) {
        // startNode/endNode (shared/components.js): no label text at all —
        // an icon in place of it is the whole point (see their comment).
        appendIcon(el, node.variant === 'start' ? 'play' : 'square', { size: node.height * 0.5, fill: outlineColor });
      }

      if (node.tabs && node.tabs.length) {
        // Tabs take over as the node's only click targets — the box itself
        // keeps just its hover highlight (see appendNodeTabs above).
        shapeEl.classList.add('diagram-node--interactive');
        appendNodeTabs(rc, el, node, opts, {
          isCollapsed: collapsedGroupIds.has(id),
          toggleCollapse: () => {
            if (collapsedGroupIds.has(id)) collapsedGroupIds.delete(id); else collapsedGroupIds.add(id);
            layoutAndRender({ animate: true });
          }
        });
      } else if (node.children) {
        shapeEl.style.cursor = 'pointer';
        shapeEl.classList.add('diagram-node--interactive');
        shapeEl.addEventListener('click', (e) => {
          e.stopPropagation();
          if (collapsedGroupIds.has(id)) collapsedGroupIds.delete(id); else collapsedGroupIds.add(id);
          layoutAndRender({ animate: true });
        });
      } else if (node.detail) {
        attachDetailCard(shapeEl, node.detail);
      }

      const target = `translate(${node.x}, ${node.y})`;
      const prev = positions.get(id);
      if (animate && prev) {
        el.style.transition = 'none';
        el.setAttribute('transform', `translate(${prev.x}, ${prev.y})`);
        el.style.opacity = '1';
        el.getBoundingClientRect(); // commit the start position before animating to the target
        el.style.transition = '';
        requestAnimationFrame(() => el.setAttribute('transform', target));
      } else if (animate && isNewNode) {
        el.style.transition = 'none';
        el.setAttribute('transform', target);
        el.style.opacity = '0';
        el.getBoundingClientRect();
        el.style.transition = '';
        requestAnimationFrame(() => { el.style.opacity = '1'; });
      } else {
        el.style.transition = 'none';
        el.setAttribute('transform', target);
        el.style.opacity = '1';
        el.style.transition = '';
      }
    });

    // Nodes that dropped out of the visible set (a group just folded its
    // children away) fade out instead of vanishing instantly.
    nodeEls.forEach((el, id) => {
      if (seenIds.has(id)) return;
      nodeEls.delete(id);
      if (animate) {
        el.style.opacity = '0';
        setTimeout(() => el.remove(), RELAYOUT_MS);
      } else {
        el.remove();
      }
    });

    positions = new Map(g.nodes().map(id => {
      const n = g.node(id);
      return [id, { x: n.x, y: n.y }];
    }));
  }

  layoutAndRender({ animate: false });
  ensureDocumenterThemeSync();
  return svg;
}
