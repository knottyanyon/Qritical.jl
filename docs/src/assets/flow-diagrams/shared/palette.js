// Resolves a diagram color from the CSS custom properties defined in
// flow-diagram.css (see that file's header for the Excalidraw palette these
// come from), so shape/edge colors have one source of truth instead of
// being hardcoded at every call site.
export function cssColor(varName) {
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
}
