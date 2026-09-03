using LinearAlgebra, Base64, FileIO, PNGFiles, ColorTypes, FixedPointNumbers

"""
Build a self-contained HTML widget for SVD image compression.

Loads `image_path`, center-crops to a square, downsamples so each side is
≈ `target_size` pixels, computes a truncated SVD of each RGB channel, and
serialises the first `max_rank` singular triplets as base64 Float32 binary.
Returns a complete HTML string suitable for embedding in an `<iframe>`.

The widget renders two fixed-size canvases side-by-side (original vs. compressed)
with a rank slider.  All computation after build time runs in the browser — no
server or Julia runtime needed at read time.
"""
function build_svd_compression(image_path::String; target_size::Int=210, max_rank::Int=80)
    img = load(image_path)
    m0, n0 = size(img)

    # Center-crop to square so a portrait image doesn't inflate widget height.
    crop = min(m0, n0)
    r0 = (m0 - crop) ÷ 2 + 1
    c0 = (n0 - crop) ÷ 2 + 1
    img = img[r0:(r0 + crop - 1), c0:(c0 + crop - 1)]

    step = max(1, crop ÷ target_size)
    img_s = img[1:step:end, 1:step:end]
    m, n = size(img_s)           # should be roughly square
    r = min(max_rank, min(m, n))

    R = Float32.(red.(img_s))
    G = Float32.(green.(img_s))
    B = Float32.(blue.(img_s))

    function channel_svd(M)
        F = svd(M)
        U = Float32.(F.U[:, 1:r])
        S = Float32.(F.S[1:r])
        Vt = Float32.(F.Vt[1:r, :])
        S_full = Float32.(F.S)
        return (; U, S, Vt, S_full)
    end

    cr, cg, cb = channel_svd(R), channel_svd(G), channel_svd(B)

    # Precompute relative Frobenius errors and storage ratios (red channel representative).
    σ = cr.S_full
    total = sum(abs2, σ)
    cum = cumsum(σ .^ 2)
    errors = [sqrt(max(0.0f0, total - get(cum, k, total))) / sqrt(total) for k in 1:r]
    storage = [k * (m + n + 1) / (m * n) for k in 1:r]

    enc(A) = base64encode(reinterpret(UInt8, vec(A)))

    return _widget_html(;
        m,
        n,
        r,
        U_R=enc(cr.U),
        S_R=enc(cr.S),
        Vt_R=enc(cr.Vt),
        U_G=enc(cg.U),
        S_G=enc(cg.S),
        Vt_G=enc(cg.Vt),
        U_B=enc(cb.U),
        S_B=enc(cb.S),
        Vt_B=enc(cb.Vt),
        errors_js="[" * join(round.(errors; digits=5), ",") * "]",
        storage_js="[" * join(round.(storage; digits=5), ",") * "]",
    )
end

function _widget_html(;
    m, n, r, U_R, S_R, Vt_R, U_G, S_G, Vt_G, U_B, S_B, Vt_B, errors_js, storage_js
)

    # canvas display size — fixed so layout is predictable regardless of img dimensions.
    # The canvas pixel data is m×n; we display it at DISP×DISP via CSS.
    disp = 230

    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SVD Image Compression</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/lato-font/3.0.0/css/lato-font.min.css"
      crossorigin="anonymous">
<style>
/* ── Reset ─────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Design tokens — light mode (matches Qritical / Catppuccin Latte) ─ */
:root {
  --accent:        #9558B2;   /* Julia purple */
  --accent-dark:   #c08ae0;   /* lighter purple for dark bg */
  --bg:            #ffffff;
  --panel-bg:      #f7f0fb;   /* matches .is-category-question body */
  --panel-border:  #e2d3ef;
  --text:          #4c4f69;   /* Catppuccin Latte text */
  --label:         #7c6f8c;
  --muted:         #9ca3af;
  --stats-border:  #e2d3ef;
  --slider-track:  #d6c8e8;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg:           #1e1e2e;  /* Catppuccin Mocha base */
    --panel-bg:     #2a1a38;  /* dark purple tint */
    --panel-border: #45335a;
    --text:         #cdd6f4;  /* Mocha text */
    --label:        #a6adc8;  /* Mocha subtext0 */
    --muted:        #7f849c;
    --stats-border: #45335a;
    --slider-track: #45335a;
    --accent:       #c08ae0;
    --accent-dark:  #c08ae0;
  }
}

/* ── Layout ─────────────────────────────────────────────────────── */
body {
  font-family: 'Lato', system-ui, -apple-system, sans-serif;
  font-size: 13px;
  background: var(--bg);
  color: var(--text);
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 1rem 1rem 0.75rem;
  line-height: 1.5;
}

.widget { width: 100%; max-width: 660px; }

.canvases {
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin-bottom: 0.85rem;
}

.panel {
  flex: 1;
  min-width: 0;
  background: var(--panel-bg);
  border: 1px solid var(--panel-border);
  border-radius: 8px;
  padding: 0.6rem 0.6rem 0.55rem;
  text-align: center;
}

.panel-label {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.055em;
  text-transform: uppercase;
  color: var(--label);
  margin-bottom: 0.4rem;
}

/* Fixed display size regardless of canvas pixel dimensions.  The image-rendering
   directive keeps it crisp rather than bilinear-blurred at the 1× upscale. */
canvas {
  display: block;
  width: $(disp)px;
  height: $(disp)px;
  max-width: 100%;
  image-rendering: pixelated;
  border-radius: 4px;
  margin: 0 auto;
}

/* ── Controls ───────────────────────────────────────────────────── */
.controls {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.7rem;
  padding: 0 0.1rem;
}

.ctrl-label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--label);
  white-space: nowrap;
  min-width: 2.5rem;
}

input[type=range] {
  flex: 1;
  height: 4px;
  accent-color: var(--accent);
  cursor: pointer;
  border-radius: 2px;
}

.ctrl-value {
  font-size: 13px;
  font-weight: 700;
  color: var(--accent);
  min-width: 2.2rem;
  text-align: right;
}

/* ── Stats row ──────────────────────────────────────────────────── */
.stats {
  display: flex;
  justify-content: space-around;
  border-top: 1px solid var(--stats-border);
  padding-top: 0.6rem;
}

.stat { text-align: center; }

.stat-label {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.1rem;
}

.stat-value {
  font-size: 1.1rem;
  font-weight: 900;
  color: var(--accent);
  letter-spacing: -0.01em;
}

.stat-unit {
  font-size: 10px;
  font-weight: 500;
  color: var(--muted);
  margin-left: 1px;
}
</style>
</head>
<body>
<div class="widget">
  <div class="canvases">
    <div class="panel">
      <div class="panel-label">Original (rank $r)</div>
      <canvas id="orig-canvas" width="$n" height="$m"></canvas>
    </div>
    <div class="panel">
      <div class="panel-label">Compressed — rank <span id="rank-display">20</span></div>
      <canvas id="comp-canvas" width="$n" height="$m"></canvas>
    </div>
  </div>

  <div class="controls">
    <span class="ctrl-label">Rank</span>
    <input type="range" id="slider" min="1" max="$r" value="20">
    <span class="ctrl-value" id="ctrl-val">20</span>
  </div>

  <div class="stats">
    <div class="stat">
      <div class="stat-label">Rank kept</div>
      <div class="stat-value">
        <span id="stat-rank">20</span><span class="stat-unit">/ $r</span>
      </div>
    </div>
    <div class="stat">
      <div class="stat-label">Storage</div>
      <div class="stat-value">
        <span id="stat-storage">--</span><span class="stat-unit">%</span>
      </div>
    </div>
    <div class="stat">
      <div class="stat-label">Rel. error</div>
      <div class="stat-value">
        <span id="stat-error">--</span><span class="stat-unit">%</span>
      </div>
    </div>
  </div>
</div>

<script>
const M = $m, N = $n, R_MAX = $r;
const ERRORS  = $errors_js;
const STORAGE = $storage_js;

// Decode base64-encoded column-major Float32 array (Julia's vec() layout).
function decodeF32(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8  = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

const CH = {
  R: { U: decodeF32("$U_R"), S: decodeF32("$S_R"), Vt: decodeF32("$Vt_R") },
  G: { U: decodeF32("$U_G"), S: decodeF32("$S_G"), Vt: decodeF32("$Vt_G") },
  B: { U: decodeF32("$U_B"), S: decodeF32("$S_B"), Vt: decodeF32("$Vt_B") },
};

// Rank-k reconstruction via outer product accumulation.
// Julia column-major layout: U[i,l] = U_flat[i + l*M], Vt[l,j] = Vt_flat[l + j*R_MAX].
// A_k[i,j] = Σ_{l=0}^{k-1} S[l] · U[i,l] · Vt[l,j]
function reconstruct(ch, k) {
  const { U, S, Vt } = ch;
  const out = new Float32Array(M * N);
  for (let l = 0; l < k; l++) {
    const sl = S[l];
    const lM = l * M;
    for (let i = 0; i < M; i++) {
      const u = sl * U[i + lM];
      const iN = i * N;
      for (let j = 0; j < N; j++) {
        out[iN + j] += u * Vt[l + j * R_MAX];
      }
    }
  }
  return out;
}

function drawRGB(canvas, rD, gD, bD) {
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(N, M);
  const d   = img.data;
  for (let p = 0, len = M * N; p < len; p++) {
    const p4 = p << 2;
    d[p4]     = Math.max(0, Math.min(255, rD[p] * 255 + 0.5 | 0));
    d[p4 + 1] = Math.max(0, Math.min(255, gD[p] * 255 + 0.5 | 0));
    d[p4 + 2] = Math.max(0, Math.min(255, bD[p] * 255 + 0.5 | 0));
    d[p4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

const origCanvas = document.getElementById('orig-canvas');
const compCanvas = document.getElementById('comp-canvas');
const slider     = document.getElementById('slider');

drawRGB(origCanvas,
  reconstruct(CH.R, R_MAX),
  reconstruct(CH.G, R_MAX),
  reconstruct(CH.B, R_MAX));

function update() {
  const k = slider.valueAsNumber;
  drawRGB(compCanvas,
    reconstruct(CH.R, k),
    reconstruct(CH.G, k),
    reconstruct(CH.B, k));

  const kStr = k.toString();
  document.getElementById('rank-display').textContent = kStr;
  document.getElementById('ctrl-val').textContent     = kStr;
  document.getElementById('stat-rank').textContent    = kStr;
  document.getElementById('stat-storage').textContent = (STORAGE[k - 1] * 100).toFixed(1);
  document.getElementById('stat-error').textContent   = (ERRORS[k - 1]  * 100).toFixed(2);
}

slider.addEventListener('input', update);
update();
</script>
</body>
</html>"""
end
