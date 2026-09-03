using LinearAlgebra, Qritical, JSON

"""
Build a self-contained HTML widget for the TEBD Néel-quench explorer.

Pre-computes — for every combination of bond dimension D and time step dt:

  - S(t): entanglement entropy at the central bond
  - ⟨S_i^z⟩(t): site-resolved magnetisation profile at every time step

The data is serialised as JSON and embedded in the HTML.  The browser renders
a live S(t) line chart and a space–time heatmap of ⟨S_i^z⟩ that update
immediately when the D or dt sliders change.  No Julia runtime is needed after
the docs build.
"""
function build_tebd_explorer(;
    L=10, T_max=4.8, D_values=[4, 8, 16, 32, 64], dt_values=[0.05, 0.1, 0.2, 0.4]
)
    dof = SpinHalf()
    g = Chain(L)
    H = XXZ(g; J=1.0, Jz=1.0, h=0.0)
    ops = algebra_generators(dof)
    Sz = ops.Sz

    all_times = Vector{Vector{Float64}}()
    all_data = Vector{Vector{Dict{String,Any}}}()

    for dt in dt_values
        nsteps = round(Int, T_max / dt)
        times = collect(dt .* (0:nsteps))
        push!(all_times, round.(times; digits=6))

        dt_row = Vector{Dict{String,Any}}()
        for D in D_values
            ψ = canonicalize(neel_state(g; dof=dof), LeftCanonical())

            sz_mat = Matrix{Float64}(undef, L, nsteps + 1)
            for i in 1:L
                sz_mat[i, 1] = real(local_expectation(ψ, Sz, i))
            end

            S_vec = [0.0]

            for step in 1:nsteps
                ψ = trotter_step(ψ, H, dt, SuzukiTrotter(2); trunc=MaxBondDimTrunc(D))
                ψ_c = canonicalize(ψ, BondCanonical(L÷2))

                for i in 1:L
                    sz_mat[i, step + 1] = real(local_expectation(ψ_c, Sz, i))
                end

                sv = ψ_c.bond_svs[L ÷ 2 + 1].values
                p = sv .^ 2 ./ sum(sv .^ 2)
                push!(S_vec, -sum(pᵢ -> pᵢ > 0 ? pᵢ * log2(pᵢ) : 0.0, p))
                ψ = ψ_c
            end

            # Row-major (time-outer): sz_flat[t*L + i] = Sz at step t, site i (0-indexed)
            sz_flat = round.(vec(transpose(sz_mat)); digits=5)

            push!(
                dt_row,
                Dict(
                    "S" => round.(S_vec; digits=5), "Sz_flat" => sz_flat, "nSteps" => nsteps
                ),
            )
            @info "Built TEBD slice" L D dt nsteps
        end
        push!(all_data, dt_row)
    end

    dataset_json = JSON.json(
        Dict(
            "L" => L,
            "T_MAX" => T_max,
            "D_VALUES" => D_values,
            "DT_VALUES" => dt_values,
            "TIMES" => all_times,
            "DATA" => all_data,
        ),
    )

    return _tebd_html(; L, T_max, D_values, dt_values, dataset_json)
end

# ─────────────────────────────────────────────────────────────────────────────

function _tebd_html(; L, T_max, D_values, dt_values, dataset_json)
    S_YMAX = log2(min(last(D_values), 2^(L÷2))) + 0.4
    S_YTICKS_JS = "[" * join(collect(0:floor(Int, S_YMAX)), ",") * "]"
    T_TICKS_JS = "[" * join(collect(0:round(Int, T_max)), ",") * "]"

    D_LABELS = join(["<span>$(v)</span>" for v in D_values], "")
    DT_LABELS = join(["<span>$(v)</span>" for v in dt_values], "")

    D_MAX_IDX = length(D_values) - 1
    DT_INIT_IDX = 1
    D_INIT_VAL = last(D_values)
    DT_INIT_VAL = dt_values[2]

    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TEBD Néel Quench Explorer</title>
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/lato-font/3.0.0/css/lato-font.min.css"
      crossorigin="anonymous">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --accent:      #9558B2;
  --ref:         #b0b0c0;
  --text:        #4c4f69;
  --label:       #7c6f8c;
  --muted:       #9ca3af;
  --bg:          #ffffff;
  --panel-bg:    #f7f0fb;
  --panel-border:#e2d3ef;
  --grid:        #e8e0f0;
  --axis:        #7c6f8c;
  --stats-border:#e2d3ef;
}
@media (prefers-color-scheme: dark) {
  :root {
    --accent:      #c08ae0;
    --ref:         #555570;
    --text:        #cdd6f4;
    --label:       #a6adc8;
    --muted:       #7f849c;
    --bg:          #1e1e2e;
    --panel-bg:    #2a1a38;
    --panel-border:#45335a;
    --grid:        #302040;
    --axis:        #7a6a8a;
    --stats-border:#45335a;
  }
}

body {
  font-family: 'Lato', system-ui, -apple-system, sans-serif;
  font-size: 13px;
  background: var(--bg);
  color: var(--text);
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 0.85rem 1rem 0.65rem;
}

.widget { width: 100%; max-width: 680px; }

.charts { display: flex; gap: 10px; margin-bottom: 0.8rem; }

.panel {
  flex: 1;
  min-width: 0;
  background: var(--panel-bg);
  border: 1px solid var(--panel-border);
  border-radius: 8px;
  padding: 0.5rem 0.5rem 0.35rem;
}

.panel-label {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.055em;
  text-transform: uppercase;
  color: var(--label);
  margin-bottom: 0.3rem;
  text-align: center;
}

svg.chart { display: block; width: 100%; height: auto; overflow: visible; }

.heatmap-wrap { display: flex; gap: 0; }

.site-axis {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 0 4px 28px 0;
  flex-shrink: 0;
}
.site-axis span { font-size: 8px; color: var(--label); line-height: 1; }

.hm-inner { flex: 1; min-width: 0; }

canvas#hm-canvas {
  display: block;
  width: 100%;
  height: 130px;
  image-rendering: pixelated;
}

.time-axis {
  display: flex;
  justify-content: space-between;
  padding: 2px 0 0;
  font-size: 8px;
  color: var(--label);
}

.hm-xlabel { text-align: center; font-size: 8.5px; color: var(--label); margin-top: 1px; }

.sliders { margin-bottom: 0.6rem; }

.slider-row {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  margin-bottom: 0.45rem;
}

.slider-name {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--label);
  white-space: nowrap;
  min-width: 6.2rem;
}

.slider-ticks { flex: 1; display: flex; flex-direction: column; gap: 2px; }

input[type=range] {
  width: 100%;
  accent-color: var(--accent);
  cursor: pointer;
  height: 4px;
}

.tick-labels {
  display: flex;
  justify-content: space-between;
  font-size: 9px;
  color: var(--muted);
  padding: 0 2px;
}

.slider-val {
  font-size: 13px;
  font-weight: 900;
  color: var(--accent);
  min-width: 3.5rem;
  text-align: right;
}

.info-row {
  display: flex;
  justify-content: space-between;
  border-top: 1px solid var(--stats-border);
  padding-top: 0.5rem;
  font-size: 11px;
  color: var(--muted);
  flex-wrap: wrap;
  gap: 4px;
}

.ref-note { display: flex; align-items: center; gap: 5px; }
.ref-swatch {
  display: inline-block;
  width: 18px;
  height: 0;
  border-top: 2px dashed var(--ref);
  flex-shrink: 0;
}
</style>
</head>
<body>
<div class="widget">

  <div class="charts">
    <div class="panel">
      <div class="panel-label">Entanglement entropy S(t)</div>
      <svg id="s-svg" class="chart" viewBox="0 0 300 155"></svg>
    </div>

    <div class="panel">
      <div class="panel-label">&#x27E8;S&#x1D93;&#x1D38;&#x27E9; space&#8211;time map</div>
      <div class="heatmap-wrap">
        <div class="site-axis" id="site-axis"></div>
        <div class="hm-inner">
          <canvas id="hm-canvas"></canvas>
          <div class="time-axis" id="time-axis"></div>
          <div class="hm-xlabel">time t</div>
        </div>
      </div>
    </div>
  </div>

  <div class="sliders">
    <div class="slider-row">
      <span class="slider-name">Bond dim D</span>
      <div class="slider-ticks">
        <input type="range" id="d-slider" min="0" max="$(D_MAX_IDX)" step="1" value="$(D_MAX_IDX)">
        <div class="tick-labels">$(D_LABELS)</div>
      </div>
      <span class="slider-val" id="d-val">D&nbsp;=&nbsp;$(D_INIT_VAL)</span>
    </div>
    <div class="slider-row">
      <span class="slider-name">Time step dt</span>
      <div class="slider-ticks">
        <input type="range" id="dt-slider" min="0" max="$(length(dt_values)-1)" step="1" value="$(DT_INIT_IDX)">
        <div class="tick-labels">$(DT_LABELS)</div>
      </div>
      <span class="slider-val" id="dt-val">dt&nbsp;=&nbsp;$(DT_INIT_VAL)</span>
    </div>
  </div>

  <div class="info-row">
    <span>L&nbsp;=&nbsp;$(L) &nbsp;&middot;&nbsp; T<sub>max</sub>&nbsp;=&nbsp;$(T_max) &nbsp;&middot;&nbsp; XXZ, 2nd-order Trotter</span>
    <span class="ref-note">
      <span class="ref-swatch"></span>
      ref:&nbsp;D=$(last(D_values)),&nbsp;dt=$(first(dt_values))
    </span>
  </div>

</div>
<script>
const DATASET = $dataset_json;
const L       = DATASET.L;
const T_MAX   = DATASET.T_MAX;

// ── SVG chart geometry ────────────────────────────────────────────
const VBW = 300, VBH = 155;
const PAD_L = 36, PAD_R = 8, PAD_T = 11, PAD_B = 27;
const PW = VBW - PAD_L - PAD_R;
const PH = VBH - PAD_T - PAD_B;
const S_YMAX   = $S_YMAX;
const X_TICKS  = $T_TICKS_JS;
const SY_TICKS = $S_YTICKS_JS;

function tx(t)   { return PAD_L + (t / T_MAX) * PW; }
function ty_S(v) { return PAD_T + PH - (v / S_YMAX) * PH; }

// ── Build S(t) SVG axes (once on page load) ───────────────────────
function initSVG() {
  const svg = document.getElementById('s-svg');
  let h = '';

  h += '<defs><clipPath id="scp"><rect x="' + PAD_L + '" y="' + PAD_T +
       '" width="' + PW + '" height="' + PH + '"/></clipPath></defs>';

  for (const v of SY_TICKS) {
    const py = ty_S(v).toFixed(1);
    h += '<line x1="' + PAD_L + '" y1="' + py + '" x2="' + (PAD_L + PW) +
         '" y2="' + py + '" stroke="var(--grid)" stroke-width="0.5"/>';
  }
  for (const t of X_TICKS) {
    const px = tx(t).toFixed(1);
    h += '<line x1="' + px + '" y1="' + PAD_T + '" x2="' + px +
         '" y2="' + (PAD_T + PH) + '" stroke="var(--grid)" stroke-width="0.5"/>';
  }

  h += '<rect x="' + PAD_L + '" y="' + PAD_T + '" width="' + PW + '" height="' + PH +
       '" fill="none" stroke="var(--axis)" stroke-width="0.8"/>';

  for (const v of SY_TICKS) {
    const py = ty_S(v).toFixed(1);
    h += '<line x1="' + (PAD_L - 3) + '" y1="' + py + '" x2="' + PAD_L +
         '" y2="' + py + '" stroke="var(--axis)" stroke-width="0.8"/>';
    h += '<text x="' + (PAD_L - 5) + '" y="' + (parseFloat(py) + 3).toFixed(1) +
         '" text-anchor="end" font-size="8.5" fill="var(--label)">' + v + '</text>';
  }

  const midY = (PAD_T + PH / 2).toFixed(1);
  h += '<text x="8" y="' + midY + '" text-anchor="middle" font-size="8.5" fill="var(--label)"' +
       ' transform="rotate(-90,8,' + midY + ')">S [bits]</text>';

  for (const t of X_TICKS) {
    const px = tx(t).toFixed(1);
    h += '<line x1="' + px + '" y1="' + (PAD_T + PH) + '" x2="' + px +
         '" y2="' + (PAD_T + PH + 3) + '" stroke="var(--axis)" stroke-width="0.8"/>';
    h += '<text x="' + px + '" y="' + (PAD_T + PH + 12) +
         '" text-anchor="middle" font-size="8.5" fill="var(--label)">' + t + '</text>';
  }
  h += '<text x="' + (PAD_L + PW / 2).toFixed(1) + '" y="' + (VBH - 1) +
       '" text-anchor="middle" font-size="8.5" fill="var(--label)">time t</text>';

  svg.innerHTML = h;

  const ref = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  ref.id = 's-ref';
  ref.setAttribute('fill', 'none');
  ref.setAttribute('stroke', 'var(--ref)');
  ref.setAttribute('stroke-width', '1.2');
  ref.setAttribute('stroke-dasharray', '5,3');
  ref.setAttribute('clip-path', 'url(#scp)');
  svg.appendChild(ref);

  const sel = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  sel.id = 's-sel';
  sel.setAttribute('fill', 'none');
  sel.setAttribute('stroke', 'var(--accent)');
  sel.setAttribute('stroke-width', '2');
  sel.setAttribute('stroke-linejoin', 'round');
  sel.setAttribute('clip-path', 'url(#scp)');
  svg.appendChild(sel);
}

function setPath(id, times, values, txFn, tyFn) {
  const pts = times.map(function(t, i) {
    return txFn(t).toFixed(1) + ',' + tyFn(values[i]).toFixed(1);
  });
  document.getElementById(id).setAttribute('d', 'M ' + pts.join(' L '));
}

// ── Heatmap colormap ──────────────────────────────────────────────
// Julia blue (#4063D8) at Sz=-0.5, white at Sz=0, Julia red (#CB3C33) at Sz=+0.5
function szColor(sz) {
  const t = sz + 0.5;   // 0 = down (blue end), 1 = up (red end)
  var r, g, b;
  if (t <= 0.5) {
    const u = t * 2;
    r = Math.round(64  + u * (245 - 64));
    g = Math.round(99  + u * (243 - 99));
    b = Math.round(216 + u * (248 - 216));
  } else {
    const u = (t - 0.5) * 2;
    r = Math.round(245 + u * (203 - 245));
    g = Math.round(243 + u * (60  - 243));
    b = Math.round(248 + u * (51  - 248));
  }
  return [r, g, b];
}

function drawHeatmap(szFlat, nSteps, nSites) {
  const canvas = document.getElementById('hm-canvas');
  canvas.width  = nSteps + 1;
  canvas.height = nSites;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(nSteps + 1, nSites);
  const d   = img.data;
  for (let s = 0; s < nSites; s++) {
    for (let ti = 0; ti <= nSteps; ti++) {
      const c = szColor(szFlat[ti * nSites + s]);
      const p = (s * (nSteps + 1) + ti) * 4;
      d[p] = c[0]; d[p+1] = c[1]; d[p+2] = c[2]; d[p+3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function buildSiteAxis() {
  const el = document.getElementById('site-axis');
  let h = '';
  for (let i = 1; i <= L; i++) h += '<span>' + i + '</span>';
  el.innerHTML = h;
}

function buildTimeAxis() {
  const el = document.getElementById('time-axis');
  const n  = 5;
  let h = '';
  for (let k = 0; k <= n; k++) {
    const t = T_MAX * k / n;
    h += '<span>' + (k === 0 || k === n ? t.toFixed(0) : t.toFixed(1)) + '</span>';
  }
  el.innerHTML = h;
}

// ── Boot ──────────────────────────────────────────────────────────
initSVG();
buildSiteAxis();
buildTimeAxis();

const REF_DT = 0;
const REF_D  = DATASET.D_VALUES.length - 1;
setPath('s-ref', DATASET.TIMES[REF_DT], DATASET.DATA[REF_DT][REF_D].S, tx, ty_S);

function update() {
  const dIdx  = parseInt(document.getElementById('d-slider').value);
  const dtIdx = parseInt(document.getElementById('dt-slider').value);
  const D     = DATASET.D_VALUES[dIdx];
  const dt    = DATASET.DT_VALUES[dtIdx];
  const dat   = DATASET.DATA[dtIdx][dIdx];
  const ts    = DATASET.TIMES[dtIdx];

  setPath('s-sel', ts, dat.S, tx, ty_S);
  drawHeatmap(dat.Sz_flat, dat.nSteps, L);

  document.getElementById('d-val').textContent  = 'D = ' + D;
  document.getElementById('dt-val').textContent = 'dt = ' + dt.toFixed(2);
}

document.getElementById('d-slider').addEventListener('input', update);
document.getElementById('dt-slider').addEventListener('input', update);
update();
</script>
</body>
</html>"""
end
