/* Tensor-Network Representation switcher
 *
 * Usage in @raw html blocks:
 *
 *   <div class="tn-repr">
 *     <div class="tn-repr-alg">
 *       LaTeX here — MathJax will render it
 *     </div>
 *     <div class="tn-repr-diag">
 *       <img src="../foo_light.svg" class="docs-light-only" style="max-width:100%"/>
 *       <img src="../foo_dark.svg"  class="docs-dark-only"  style="max-width:100%"/>
 *     </div>
 *   </div>
 *
 * Add data-default="alg" to show the algebraic panel first (default: diag).
 */
(function () {
  "use strict";

  function initTnRepr(root) {
    var containers = (root || document).querySelectorAll(".tn-repr");
    containers.forEach(function (el) {
      if (el.dataset.tnReprInit) return;
      el.dataset.tnReprInit = "1";

      var algPanel  = el.querySelector(".tn-repr-alg");
      var diagPanel = el.querySelector(".tn-repr-diag");
      if (!algPanel || !diagPanel) return;

      algPanel.classList.add("tn-repr-panel");
      diagPanel.classList.add("tn-repr-panel");

      var defaultKey = el.dataset.default === "alg" ? "alg" : "diag";

      var tabs = [
        { key: "alg",  label: "Algebraic",    panel: algPanel  },
        { key: "diag", label: "Diagrammatic",  panel: diagPanel },
      ];

      /* Build tab bar */
      var tabbar = document.createElement("div");
      tabbar.className = "tn-repr-tabbar";

      var btnGroup = document.createElement("div");
      btnGroup.className = "tn-repr-btngroup";

      tabs.forEach(function (t) {
        var btn = document.createElement("button");
        btn.className = "tn-repr-tab" + (t.key === defaultKey ? " active" : "");
        btn.textContent = t.label;
        btn.setAttribute("type", "button");

        btn.addEventListener("click", function () {
          /* Update panels */
          tabs.forEach(function (u) {
            u.panel.classList.toggle("active", u.key === t.key);
          });
          /* Update buttons */
          btnGroup.querySelectorAll(".tn-repr-tab").forEach(function (b) {
            b.classList.remove("active");
          });
          btn.classList.add("active");

          /* Re-typeset MathJax when revealing the algebraic panel,
             because MathJax skips hidden elements on page load.    */
          if (t.key === "alg") {
            if (window.MathJax && window.MathJax.typesetPromise) {
              window.MathJax.typesetPromise([t.panel]);
            } else if (window.MathJax && window.MathJax.Hub) {
              window.MathJax.Hub.Queue(["Typeset", window.MathJax.Hub, t.panel]);
            }
          }
        });

        btnGroup.appendChild(btn);

        /* Activate the default panel */
        if (t.key === defaultKey) t.panel.classList.add("active");
      });

      tabbar.appendChild(btnGroup);
      el.insertBefore(tabbar, el.firstChild);
    });
  }

  document.addEventListener("DOMContentLoaded", function () { initTnRepr(); });

  /* Expose for pages that inject content dynamically */
  window.initTnRepr = initTnRepr;
})();
