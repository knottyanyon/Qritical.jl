import katex from '../vendor/katex/katex.mjs';

// A label with no `\` and no `$` carries no LaTeX at all — it's plain prose,
// whether that's a multi-word phrase like "Truncation criterion met?" or a
// single word like "Start"/"Refine". Two reasons this must go straight to a
// text node rather than KaTeX: multi-word prose run through math mode loses
// its spacing ("Truncationcriterionmet?", since math mode has no concept of
// word spacing), and single-word prose renders in KaTeX's italic math font
// instead of the diagram's own sans body font (see .math-node-label in
// flow-diagram.css) — indistinguishable from an actual math variable even
// though it's just a caption.
function isPlainProse(text) {
  return !text.includes('\\') && !text.includes('$');
}

export async function renderMath(text, container, { engine = 'katex', displayMode = false } = {}) {
  container.innerHTML = '';
  if (isPlainProse(text)) {
    container.textContent = text;
    return;
  }
  if (text.includes('$')) {
    return renderMixedLabel(text, container, engine, displayMode);
  }
  if (engine === 'mathjax') {
    return renderWithMathJax(text, container, displayMode);
  }
  try {
    katex.render(text, container, { throwOnError: true, displayMode });
  } catch (err) {
    return renderWithMathJax(text, container, displayMode);
  }
}

// Labels that mix plain text with inline `$...$` math segments (e.g. "Truncate
// $\Sigma$, renormalize") aren't valid raw TeX, so passing them wholesale to
// katex.render() throws, and the MathJax fallback also can't parse the literal
// `$` characters (it just hangs, since tex2svgPromise treats its input as pure
// math, not a document with inline-math delimiters). Split on `$...$` pairs and
// render each math segment with KaTeX, keeping the rest as plain text.
//
// `engine` is accepted here (rather than silently dropped) so the contract with
// renderMath() is visible, but mixed labels intentionally only support
// per-segment KaTeX with a plain-text fallback: MathJax's tex2svgPromise is
// async per call, and firing one per segment inside this synchronous forEach
// would append segments out of order as each promise resolves independently.
// A `engine: 'mathjax'` request for a mixed label still renders via KaTeX
// rather than being silently ignored outright; wiring true per-segment MathJax
// support would need an ordered/awaited render loop, which no current caller
// needs (nothing passes `engine: 'mathjax'` with a `$...$`-mixed label today).
function renderMixedLabel(text, container, engine = 'katex', displayMode = false) {
  const parts = text.split(/(\$[^$]+\$)/g).filter(part => part.length > 0);
  parts.forEach(part => {
    if (part.startsWith('$') && part.endsWith('$') && part.length > 1) {
      const span = document.createElement('span');
      const inner = part.slice(1, -1);
      try {
        katex.render(inner, span, { throwOnError: true, displayMode });
      } catch (err) {
        span.textContent = inner;
      }
      container.appendChild(span);
    } else {
      container.appendChild(document.createTextNode(part));
    }
  });
}

// Detail-card content (tooltips) is a mix of author-written HTML — links from
// codeTab/predefinedProcessNode, `<br>` separators — and inline `$...$` math,
// e.g. 'Solves $J \\Delta x = -r$ for the Newton update step.' Unlike a plain
// label, this can't go through renderMixedLabel (it would escape the HTML into
// literal text); instead set the HTML first, then walk the resulting text
// nodes and swap any `$...$` segment for a KaTeX-rendered span in place,
// leaving tags untouched.
export function renderMathInHtml(html, container) {
  container.innerHTML = html;
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  let node;
  while ((node = walker.nextNode())) {
    if (node.nodeValue.includes('$')) textNodes.push(node);
  }
  for (const textNode of textNodes) {
    const parts = textNode.nodeValue.split(/(\$[^$]+\$)/g).filter(part => part.length > 0);
    if (parts.length <= 1 && !parts[0]?.startsWith('$')) continue;
    const frag = document.createDocumentFragment();
    parts.forEach(part => {
      if (part.startsWith('$') && part.endsWith('$') && part.length > 1) {
        const span = document.createElement('span');
        try {
          katex.render(part.slice(1, -1), span, { throwOnError: true });
        } catch (err) {
          span.textContent = part.slice(1, -1);
        }
        frag.appendChild(span);
      } else {
        frag.appendChild(document.createTextNode(part));
      }
    });
    textNode.replaceWith(frag);
  }
}

// Grace period to let a host page's own MathJax config appear (e.g.
// Documenter.jl's site-wide MathJax3 setup, injected asynchronously via
// requirejs) before assuming no host engine is coming.
const MATHJAX_HOST_GRACE_MS = 3000;
// Once a MathJax config (ours or the host's) exists, how long to wait for
// the engine to finish booting and expose tex2svgPromise.
const MATHJAX_READY_TIMEOUT_MS = 8000;

let mathJaxReadyPromise = null;

function pollFor(predicate, { intervalMs, timeoutMs }) {
  return new Promise(resolve => {
    const deadline = Date.now() + timeoutMs;
    const tick = () => {
      if (predicate()) return resolve(true);
      if (Date.now() >= deadline) return resolve(false);
      setTimeout(tick, intervalMs);
    };
    tick();
  });
}

function loadVendoredMathJax() {
  window.MathJax = window.MathJax || {
    tex: { inlineMath: [['\\(', '\\)']] },
    svg: { fontCache: 'none' },
    // Required. MathJax v4's Speech Rule Engine loads a Web Worker that isn't
    // vendored here; leaving enrichment on makes every render hang instead of
    // failing. The `menuOptions.settings.enrich` line is not redundant — the
    // bundled menu re-derives the three enable* flags from it.
    options: {
      enableMenu: false,
      menuOptions: { settings: { enrich: false } },
      enableEnrichment: false,
      enableSpeech: false,
      enableBraille: false,
    },
  };
  const script = document.createElement('script');
  script.src = new URL('../vendor/mathjax/tex-svg.js', import.meta.url).href;
  document.head.appendChild(script);
}

// Resolves to the page's MathJax runtime once it's ready to typeset — either
// a host page's own engine (e.g. Documenter.jl's MathJax3, configured
// site-wide for the `physics` LaTeX package) or, only if no host engine
// shows up at all, a vendored copy loaded lazily on first use here. Loading
// our own MathJax unconditionally — regardless of what the host page
// provides — causes two engines to fight over the same `window.MathJax`
// global (Documenter overwrites it asynchronously and unconditionally),
// producing "Can't find handler for document" and stack overflow errors.
// This loads at most one MathJax runtime per page, and only ever calls this
// once (subsequent renders reuse the same promise).
function ensureMathJax() {
  if (mathJaxReadyPromise) return mathJaxReadyPromise;
  mathJaxReadyPromise = (async () => {
    const hostAppeared = await pollFor(() => !!window.MathJax, {
      intervalMs: 100,
      timeoutMs: MATHJAX_HOST_GRACE_MS,
    });
    if (!hostAppeared) loadVendoredMathJax();
    const ready = await pollFor(
      () => !!(window.MathJax && typeof window.MathJax.tex2svgPromise === 'function'),
      { intervalMs: 100, timeoutMs: MATHJAX_READY_TIMEOUT_MS }
    );
    return ready ? window.MathJax : null;
  })();
  return mathJaxReadyPromise;
}

async function renderWithMathJax(text, container, displayMode) {
  const mathJax = await ensureMathJax();
  if (!mathJax) {
    container.textContent = text;
    return;
  }
  const wrapper = await mathJax.tex2svgPromise(text, { display: displayMode });
  const svg = wrapper.querySelector('svg');
  container.appendChild(svg);
}
