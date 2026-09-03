// Keeps a diagram wrapper's `data-theme` attribute in sync with Documenter.jl's
// active theme, so the dark-mode SVG invert filter in flow-diagram.css (which
// only engages via `.flow-diagram-wrapper[data-theme="dark"]`) actually turns
// on when a page is viewed in one of Documenter's dark themes. Without this,
// an embedded diagram stays rendered for light mode regardless of what theme
// the surrounding page is actually in.
//
// Detects dark vs. light via Documenter's own `docs-dark-only`/`docs-light-only`
// convention — each of its themes' CSS hides one or the other — rather than
// hardcoding theme names, so this keeps working for `documenter-dark`, any of
// the bundled catppuccin variants, and any custom theme that follows the same
// convention. No-ops entirely outside a Documenter page (detected via the
// `#documenter` root element it always renders), so hosts with their own
// theming — like this repo's own preview app, which toggles `data-theme`
// directly via a button — are unaffected.
let observing = false;

function isDocumenterHost() {
  return !!document.getElementById('documenter');
}

function isDocumenterDark() {
  const probe = document.createElement('span');
  probe.className = 'docs-dark-only';
  probe.style.cssText = 'position:absolute;width:0;height:0;overflow:hidden;pointer-events:none;';
  document.body.appendChild(probe);
  const dark = getComputedStyle(probe).display !== 'none';
  probe.remove();
  return dark;
}

function syncAllWrappers() {
  const theme = isDocumenterDark() ? 'dark' : 'light';
  document.querySelectorAll('.flow-diagram-wrapper, .tensor-diagram-wrapper')
    .forEach(el => el.setAttribute('data-theme', theme));
}

// Called after every diagram mount; cheap to call repeatedly since the
// MutationObserver is only ever attached once per page.
export function ensureDocumenterThemeSync() {
  if (!isDocumenterHost()) return;
  syncAllWrappers();
  if (observing) return;
  observing = true;
  // Documenter's themeswap.js sets the active theme by rewriting
  // `document.documentElement.className` (e.g. to "theme--documenter-dark"),
  // both on load and whenever the theme picker changes.
  new MutationObserver(syncAllWrappers)
    .observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
}
