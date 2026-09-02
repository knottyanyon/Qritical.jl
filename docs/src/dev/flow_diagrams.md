# Flow Diagrams

Reference page for the interactive flow diagrams from `qritical-numroutines-diagrams`,
embedded via the `flowdiagram` token (see
[`flowdiagram_preprocessor.jl`](https://github.com/knottyanyon/qritical-numroutines-diagrams)).
Not part of the public docs; kept here to preview new diagrams against the real
Documenter.jl page shell before wiring them into a tutorial.

---

## Group expand/collapse

```@raw html
<link rel="stylesheet" href="../../assets/flow-diagrams/flow-diagram.css">
<link rel="stylesheet" href="../../assets/flow-diagrams/vendor/katex/katex.css">
<script>
// Documenter's own MathJax3 config loads require.js, which defines a global
// define() that looks AMD-compatible. rough.min.js/dagre.min.js are UMD
// bundles that, when they see a global define(), register as anonymous AMD
// modules instead of attaching window.rough/window.dagre — so every
// flowdiagram mount() call would fail with "rough is not defined". Hiding
// define() while these two vendor scripts load forces the UMD branch that
// sets the global, then restores it for anything else on the page that
// needs it (e.g. Documenter's own MathJax bootstrapping).
window.__flowdiagram_saved_define = window.define;
window.define = undefined;
</script>
<script src="../../assets/flow-diagrams/vendor/rough.min.js"></script>
<script src="../../assets/flow-diagrams/vendor/dagre.min.js"></script>
<script>
window.define = window.__flowdiagram_saved_define;
delete window.__flowdiagram_saved_define;
</script>
<div class="flow-diagram-wrapper">
  <div id="flowdiagram-numerical-routines-group-collapse-demo"></div>
</div>
<script type="module">
import('../../assets/flow-diagrams/numerical-routines/group-collapse-demo.js')
  .then(m => m.mount(document.getElementById('flowdiagram-numerical-routines-group-collapse-demo')));
</script>
```

---

## Subprocess inline expand

```@raw html
<link rel="stylesheet" href="../../assets/flow-diagrams/flow-diagram.css">
<link rel="stylesheet" href="../../assets/flow-diagrams/vendor/katex/katex.css">
<script>
// Documenter's own MathJax3 config loads require.js, which defines a global
// define() that looks AMD-compatible. rough.min.js/dagre.min.js are UMD
// bundles that, when they see a global define(), register as anonymous AMD
// modules instead of attaching window.rough/window.dagre — so every
// flowdiagram mount() call would fail with "rough is not defined". Hiding
// define() while these two vendor scripts load forces the UMD branch that
// sets the global, then restores it for anything else on the page that
// needs it (e.g. Documenter's own MathJax bootstrapping).
window.__flowdiagram_saved_define = window.define;
window.define = undefined;
</script>
<script src="../../assets/flow-diagrams/vendor/rough.min.js"></script>
<script src="../../assets/flow-diagrams/vendor/dagre.min.js"></script>
<script>
window.define = window.__flowdiagram_saved_define;
delete window.__flowdiagram_saved_define;
</script>
<div class="flow-diagram-wrapper">
  <div id="flowdiagram-numerical-routines-subprocess-inline-expand-demo"></div>
</div>
<script type="module">
import('../../assets/flow-diagrams/numerical-routines/subprocess-inline-expand-demo.js')
  .then(m => m.mount(document.getElementById('flowdiagram-numerical-routines-subprocess-inline-expand-demo')));
</script>
```

---

## Multi-way decision routing

```@raw html
<link rel="stylesheet" href="../../assets/flow-diagrams/flow-diagram.css">
<link rel="stylesheet" href="../../assets/flow-diagrams/vendor/katex/katex.css">
<script>
// Documenter's own MathJax3 config loads require.js, which defines a global
// define() that looks AMD-compatible. rough.min.js/dagre.min.js are UMD
// bundles that, when they see a global define(), register as anonymous AMD
// modules instead of attaching window.rough/window.dagre — so every
// flowdiagram mount() call would fail with "rough is not defined". Hiding
// define() while these two vendor scripts load forces the UMD branch that
// sets the global, then restores it for anything else on the page that
// needs it (e.g. Documenter's own MathJax bootstrapping).
window.__flowdiagram_saved_define = window.define;
window.define = undefined;
</script>
<script src="../../assets/flow-diagrams/vendor/rough.min.js"></script>
<script src="../../assets/flow-diagrams/vendor/dagre.min.js"></script>
<script>
window.define = window.__flowdiagram_saved_define;
delete window.__flowdiagram_saved_define;
</script>
<div class="flow-diagram-wrapper">
  <div id="flowdiagram-numerical-routines-multiway-decision-demo"></div>
</div>
<script type="module">
import('../../assets/flow-diagrams/numerical-routines/multiway-decision-demo.js')
  .then(m => m.mount(document.getElementById('flowdiagram-numerical-routines-multiway-decision-demo')));
</script>
```

---

## DMRG sweep subprocess

```@raw html
<link rel="stylesheet" href="../../assets/flow-diagrams/flow-diagram.css">
<link rel="stylesheet" href="../../assets/flow-diagrams/vendor/katex/katex.css">
<script>
// Documenter's own MathJax3 config loads require.js, which defines a global
// define() that looks AMD-compatible. rough.min.js/dagre.min.js are UMD
// bundles that, when they see a global define(), register as anonymous AMD
// modules instead of attaching window.rough/window.dagre — so every
// flowdiagram mount() call would fail with "rough is not defined". Hiding
// define() while these two vendor scripts load forces the UMD branch that
// sets the global, then restores it for anything else on the page that
// needs it (e.g. Documenter's own MathJax bootstrapping).
window.__flowdiagram_saved_define = window.define;
window.define = undefined;
</script>
<script src="../../assets/flow-diagrams/vendor/rough.min.js"></script>
<script src="../../assets/flow-diagrams/vendor/dagre.min.js"></script>
<script>
window.define = window.__flowdiagram_saved_define;
delete window.__flowdiagram_saved_define;
</script>
<div class="flow-diagram-wrapper">
  <div id="flowdiagram-numerical-routines-dmrg-sweep-subprocess"></div>
</div>
<script type="module">
import('../../assets/flow-diagrams/numerical-routines/dmrg-sweep-subprocess.js')
  .then(m => m.mount(document.getElementById('flowdiagram-numerical-routines-dmrg-sweep-subprocess')));
</script>
```

---

## DMRG optimization

```@raw html
<link rel="stylesheet" href="../../assets/flow-diagrams/flow-diagram.css">
<link rel="stylesheet" href="../../assets/flow-diagrams/vendor/katex/katex.css">
<script>
// Documenter's own MathJax3 config loads require.js, which defines a global
// define() that looks AMD-compatible. rough.min.js/dagre.min.js are UMD
// bundles that, when they see a global define(), register as anonymous AMD
// modules instead of attaching window.rough/window.dagre — so every
// flowdiagram mount() call would fail with "rough is not defined". Hiding
// define() while these two vendor scripts load forces the UMD branch that
// sets the global, then restores it for anything else on the page that
// needs it (e.g. Documenter's own MathJax bootstrapping).
window.__flowdiagram_saved_define = window.define;
window.define = undefined;
</script>
<script src="../../assets/flow-diagrams/vendor/rough.min.js"></script>
<script src="../../assets/flow-diagrams/vendor/dagre.min.js"></script>
<script>
window.define = window.__flowdiagram_saved_define;
delete window.__flowdiagram_saved_define;
</script>
<div class="flow-diagram-wrapper">
  <div id="flowdiagram-numerical-routines-dmrg-optimization-demo"></div>
</div>
<script type="module">
import('../../assets/flow-diagrams/numerical-routines/dmrg-optimization-demo.js')
  .then(m => m.mount(document.getElementById('flowdiagram-numerical-routines-dmrg-optimization-demo')));
</script>
```

---

## Process tabs

```@raw html
<link rel="stylesheet" href="../../assets/flow-diagrams/flow-diagram.css">
<link rel="stylesheet" href="../../assets/flow-diagrams/vendor/katex/katex.css">
<script>
// Documenter's own MathJax3 config loads require.js, which defines a global
// define() that looks AMD-compatible. rough.min.js/dagre.min.js are UMD
// bundles that, when they see a global define(), register as anonymous AMD
// modules instead of attaching window.rough/window.dagre — so every
// flowdiagram mount() call would fail with "rough is not defined". Hiding
// define() while these two vendor scripts load forces the UMD branch that
// sets the global, then restores it for anything else on the page that
// needs it (e.g. Documenter's own MathJax bootstrapping).
window.__flowdiagram_saved_define = window.define;
window.define = undefined;
</script>
<script src="../../assets/flow-diagrams/vendor/rough.min.js"></script>
<script src="../../assets/flow-diagrams/vendor/dagre.min.js"></script>
<script>
window.define = window.__flowdiagram_saved_define;
delete window.__flowdiagram_saved_define;
</script>
<div class="flow-diagram-wrapper">
  <div id="flowdiagram-numerical-routines-process-tabs-demo"></div>
</div>
<script type="module">
import('../../assets/flow-diagrams/numerical-routines/process-tabs-demo.js')
  .then(m => m.mount(document.getElementById('flowdiagram-numerical-routines-process-tabs-demo')));
</script>
```
