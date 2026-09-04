// Orthogonal (Manhattan-style) edge routing for flow diagrams: every edge is
// one straight segment (horizontal or vertical) when the nodes are already
// aligned, or a two-segment elbow otherwise — never a diagonal line. Each
// node shape also gets a shape-appropriate connection point instead of a
// generic bounding-box corner (diamond vertices, circle N/E/S/W points, box
// edge midpoints), so a line never crosses into the interior of the symbol
// it's leaving/entering.

const DIAMOND_SHAPES = new Set(['diamond', 'decision']);
const CIRCLE_SHAPES = new Set(['circle', 'connector']);

// Snap-to-aligned threshold, in px: displacement below this on an axis is
// treated as "already aligned" on that axis, so the path is a single
// straight segment instead of a needless one-pixel elbow.
const ALIGN_EPS = 4;

// The point where a straight segment along `axis` crosses `node`'s boundary,
// on the side given by `sign` (+1 = right/below, -1 = left/above). For a
// diamond this IS its vertex on that axis; for a circle it's the matching
// cardinal point; for every other (rectangular) shape it's the midpoint of
// the facing bounding-box edge. Crucially, a diamond's left/right vertex can
// only be entered/left by a *horizontal* segment (it's the one point where
// the diamond touches its own vertical centerline's side extreme), and its
// top/bottom vertex only by a *vertical* one — so callers must never ask for
// an axis that doesn't match how the segment actually approaches the shape.
function boundaryPoint(node, axis, sign) {
  const hw = node.width / 2, hh = node.height / 2;
  if (CIRCLE_SHAPES.has(node.shape)) {
    const r = Math.min(node.width, node.height) / 2;
    return axis === 'h' ? { x: node.x + sign * r, y: node.y } : { x: node.x, y: node.y + sign * r };
  }
  return axis === 'h' ? { x: node.x + sign * hw, y: node.y } : { x: node.x, y: node.y + sign * hh };
}

// Which axis a node prefers to route its side of an edge along, given the
// direction (dx, dy) toward the other node's center.
function preferredAxis(node, dx, dy) {
  if (DIAMOND_SHAPES.has(node.shape)) {
    // Horizontal by default, so branches read as "out the side, then down"
    // rather than a diagonal into the diamond's corner; vertical only when
    // the other node is essentially directly above/below.
    return Math.abs(dx) < ALIGN_EPS ? 'v' : 'h';
  }
  if (CIRCLE_SHAPES.has(node.shape)) {
    return Math.abs(dx) > Math.abs(dy) ? 'h' : 'v';
  }
  // Rect-like shapes: vertical by default (top-to-bottom flowchart order),
  // horizontal only when essentially side-by-side.
  return Math.abs(dy) < ALIGN_EPS ? 'h' : 'v';
}

// A "back edge" — one whose target sits at or above its source, e.g. a loop
// closing back to an earlier step — can't be routed as a direct elbow: doing
// so would send it straight up through whatever nodes/edges sit between
// source and target, which is how an unrelated edge ends up overlapping a
// node or another edge's trunk line (see sideLaneEdgePoints). Some slack
// (rather than a strict `<=`) avoids flagging edges that are only a few
// pixels off from a same-rank forward edge.
export function isBackEdge(fromNode, toNode) {
  return toNode.y < fromNode.y + ALIGN_EPS;
}

// dagre lays every node out onto a small number of shared y "rows" (ranks) —
// one per step of the flowchart. Groups a flat list of node y-coordinates
// into those rows (nearby y's collapse into one row, tolerating dagre's
// per-node height differences within a rank) and returns them sorted
// top-to-bottom, for isRankSkipping below to count rows crossed.
export function rankLevels(nodeList) {
  const ys = [...new Set(nodeList.map(n => Math.round(n.y)))].sort((a, b) => a - b);
  const levels = [];
  for (const y of ys) {
    if (levels.length && y - levels[levels.length - 1] < ALIGN_EPS * 3) continue;
    levels.push(y);
  }
  return levels;
}

function rankIndexFor(y, levels) {
  let closest = 0, bestDiff = Infinity;
  levels.forEach((ly, i) => {
    const diff = Math.abs(ly - y);
    if (diff < bestDiff) { bestDiff = diff; closest = i; }
  });
  return closest;
}

// A forward edge that spans more than one rank (e.g. a node feeding a step
// two rows down, skipping the row in between) risks its straight/elbow path
// running through whatever sits in that skipped row — the same overlap risk
// a back edge has, just going the "normal" direction. Routed through a side
// lane too rather than attempting precise per-node collision detection.
export function isRankSkipping(fromNode, toNode, levels) {
  return rankIndexFor(toNode.y, levels) - rankIndexFor(fromNode.y, levels) > 1;
}

// Routes an edge out and around every node via a dedicated vertical lane at
// `laneX`, entering/leaving both nodes from whichever side (left/right)
// actually faces the lane. Caller picks `laneX` clear of every node's
// bounding box (see the lane setup in render-flow.js), so this path can
// never cross through a node's interior, and a distinct `laneX` per lane
// edge (also computed there) keeps parallel lane edges from overlapping.
export function sideLaneEdgePoints(fromNode, toNode, laneX) {
  const exitSign = Math.sign(laneX - fromNode.x || 1);
  const entrySign = Math.sign(laneX - toNode.x || 1);
  const p1 = boundaryPoint(fromNode, 'h', exitSign);
  const p2 = boundaryPoint(toNode, 'h', entrySign);
  return [[p1.x, p1.y], [laneX, p1.y], [laneX, p2.y], [p2.x, p2.y]];
}

// Fixed vertical drop from a diamond's vertex to the shared horizontal
// "spine" a multi-way bus route fans out along (see computeBusRoutes),
// before dropping straight down/up into each target/source. Kept short so
// the spine still reads as belonging to the diamond, not floating loose in
// the rank gap.
const BUS_SPINE_OFFSET = 26;

// Below this vertical gap (diamond vertex to nearest branch box top), a
// plain two-way Yes/No decision is cramped enough that each branch's
// independent elbow (see orthogonalEdgePoints) reads as a near-diagonal
// jog right under the diamond's point. Routing it through the same
// trunk+spine pattern as a >2-way bus instead gives both branches a common
// drop off the vertex before they split, which reads cleanly even with
// very little room. Left alone (i.e. still handled by orthogonalEdgePoints)
// when there's enough clearance that a direct elbow already looks fine.
const SHORT_GAP_THRESHOLD = 50;

function groupBy(edges, keyOf) {
  const map = new Map();
  edges.forEach(e => {
    const key = keyOf(e);
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(e);
  });
  return map;
}

// Two related fan patterns, both meant to replace independently-chosen exit
// points (which read as a diagonal mess) with one shared trunk:
//
// - Branching out of a decision: a diamond with more than two edges leaving
//   in the same direction — e.g. "Which Fruit? -> Apple/Pear/Peach/Orange/
//   Other" — gets a single line off its vertex opening into a shared
//   horizontal spine, with one short drop per branch. Gated on the *source*
//   being a diamond: this is specifically the multi-way-switch convention,
//   not a general "many edges leave here" declutter.
//
// - Converging into any node: more than two edges arriving at the *same*
//   node, regardless of its shape, merge onto a shared spine before a single
//   trunk reaches the node. Unlike the branching case this isn't restricted
//   to diamonds — it expresses that several alternative paths (e.g. sibling
//   options off a decision, each a different "recipe" for the same task)
//   produce the same kind of output and share one interface into whatever
//   comes next, which is a structural fact about the node being converged
//   on, not about where the paths came from.
//
// Both only apply to *forward* edges (a loop-back branch still routes
// through sideLaneEdgePoints), and only once there are more than two edges
// sharing an endpoint and a direction — a plain 2-way branch, or an
// incidental 2-source convergence, is left untouched.
//
// Returns `{ trunks, arrowedTrunks, drops }`. `trunks` are shared polylines
// drawn once with no arrowhead. `arrowedTrunks` are shared polylines drawn
// once *with* an arrowhead at their end — only the converging case has one
// (the spine-to-node stem: that's where the arrow belongs, since every
// individual drop merges into the spine rather than terminating anywhere).
// `drops` maps each qualifying edge's `"from->to"` key to `{ points, arrow }`
// — its own short polyline between the spine and its target/source, and
// whether that polyline should render its own arrowhead (true when
// branching, since each drop really does end at a distinct target; false
// when converging, since the drop ends at the shared spine, not the node).
export function computeBusRoutes(g) {
  const trunks = [];
  const arrowedTrunks = [];
  const drops = new Map();
  const forwardEdges = g.edges().filter(e => !isBackEdge(g.node(e.v), g.node(e.w)));

  const outByFrom = groupBy(
    forwardEdges.filter(e => DIAMOND_SHAPES.has(g.node(e.v).shape) && g.node(e.w).y > g.node(e.v).y),
    e => e.v
  );
  outByFrom.forEach((edges, fromId) => {
    if (edges.length < 2) return;
    const fromNode = g.node(fromId);
    const trunk = boundaryPoint(fromNode, 'v', 1); // bottom vertex
    const targets = edges.map(e => g.node(e.w));
    if (edges.length === 2) {
      const minGap = Math.min(...targets.map(t => (t.y - t.height / 2) - trunk.y));
      if (minGap > SHORT_GAP_THRESHOLD) return; // plenty of room: independent elbows already look fine
    }
    const spineY = trunk.y + BUS_SPINE_OFFSET;
    const xs = [trunk.x, ...targets.map(t => t.x)];
    trunks.push([[trunk.x, trunk.y], [trunk.x, spineY]]);
    trunks.push([[Math.min(...xs), spineY], [Math.max(...xs), spineY]]);
    edges.forEach((e, i) => {
      const entry = boundaryPoint(targets[i], 'v', -1); // target's top vertex
      drops.set(`${e.v}->${e.w}`, { points: [[targets[i].x, spineY], [entry.x, entry.y]], arrow: true });
    });
  });

  const inByTo = groupBy(
    forwardEdges.filter(e => g.node(e.v).y < g.node(e.w).y),
    e => e.w
  );
  inByTo.forEach((edges, toId) => {
    if (edges.length <= 2) return;
    const toNode = g.node(toId);
    const trunk = boundaryPoint(toNode, 'v', -1); // top vertex
    const spineY = trunk.y - BUS_SPINE_OFFSET;
    const sources = edges.map(e => g.node(e.v));
    const xs = [trunk.x, ...sources.map(s => s.x)];
    arrowedTrunks.push([[trunk.x, spineY], [trunk.x, trunk.y]]); // spine -> node, arrowhead lands on the node
    trunks.push([[Math.min(...xs), spineY], [Math.max(...xs), spineY]]);
    edges.forEach((e, i) => {
      const exit = boundaryPoint(sources[i], 'v', 1); // source's bottom vertex
      drops.set(`${e.v}->${e.w}`, { points: [[exit.x, exit.y], [sources[i].x, spineY]], arrow: false });
    });
  });

  return { trunks, arrowedTrunks, drops };
}

// Builds the polyline (as [x, y] pairs) for one edge between two laid-out
// dagre nodes, routed as straight-when-aligned or a single axis-aligned
// elbow. The entry side's axis is *derived* from the exit side's — never
// picked independently — so a two-segment path always approaches its target
// on the axis that segment can actually enter cleanly. Only used for edges
// that aren't back edges or rank-skipping (see above) — those go through
// sideLaneEdgePoints instead.
export function orthogonalEdgePoints(fromNode, toNode) {
  const dx = toNode.x - fromNode.x, dy = toNode.y - fromNode.y;

  const exitAxis = preferredAxis(fromNode, dx, dy);
  const exitSign = exitAxis === 'h' ? Math.sign(dx || 1) : Math.sign(dy || 1);
  const p1 = boundaryPoint(fromNode, exitAxis, exitSign);

  const aligned = exitAxis === 'v' ? Math.abs(dx) < ALIGN_EPS : Math.abs(dy) < ALIGN_EPS;
  const entryAxis = aligned ? exitAxis : (exitAxis === 'h' ? 'v' : 'h');
  const entrySign = entryAxis === 'h' ? Math.sign(-dx || -1) : Math.sign(-dy || -1);
  const p2 = boundaryPoint(toNode, entryAxis, entrySign);

  if (aligned) return [[p1.x, p1.y], [p2.x, p2.y]];

  const corner = exitAxis === 'h' ? [p2.x, p1.y] : [p1.x, p2.y];
  return [[p1.x, p1.y], corner, [p2.x, p2.y]];
}
