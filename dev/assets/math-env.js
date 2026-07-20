// math-env.js — auto-numbering, anchor IDs, and badge injection for .math-env boxes.
//
// Header layout produced for each environment:
//
//   [ Title text (left, larger)          ] [ ★ Theorem 1 (right, opaque badge) ]
//
// The badge is injected into the RIGHT of the header via flex; the user's title
// text is wrapped in .math-env-title and stays LEFT-aligned.
//
// Anchor IDs:
//   User-supplied id="" is preserved; auto-generated id ("theorem-2") otherwise.
//   Cross-page links: [Theorem 2](other-page.html#theorem-2)
//
// Security: uses only DOM node creation (textContent, appendChild, insertBefore).
// No innerHTML writes.
//
// Icons and their mathematical motivation:
//   ★  theorem     — star, denotes a major/important result
//   ◆  lemma       — diamond, a supporting stepping stone
//   ∴  corollary   — "therefore" symbol; corollaries follow from what precedes
//   ≜  definition  — "equal by definition" (standard notation)
//   ⊢  proposition — turnstile; "is provable / follows from the axioms"
//   ⚑  remark      — flag / annotation

(function () {
  "use strict";

  var TYPES = [
    { cls: "math-theorem",     label: "Theorem",     icon: "★" }, // ★
    { cls: "math-lemma",       label: "Lemma",       icon: "◆" }, // ◆
    { cls: "math-corollary",   label: "Corollary",   icon: "∴" }, // ∴
    { cls: "math-definition",  label: "Definition",  icon: "≜" }, // ≜
    { cls: "math-proposition", label: "Proposition", icon: "⊢" }, // ⊢
    { cls: "math-remark",      label: "Remark",      icon: "⚑" }, // ⚑
  ];

  document.addEventListener("DOMContentLoaded", function () {
    TYPES.forEach(function (t) {
      var n = 0;
      document.querySelectorAll("." + t.cls).forEach(function (env) {
        n++;

        // Auto-assign anchor id if absent.
        if (!env.id) {
          env.id = t.cls.replace("math-", "") + "-" + n;
        }

        var header = env.querySelector(".math-env-header");
        if (!header) return;

        // 1. Wrap existing header content (the user's title) in .math-env-title
        //    so flex layout treats title and badge as two clean children.
        //    When the title is empty (e.g. a headerless remark) we skip the
        //    wrapper so the badge is the sole flex child and margin-left:auto
        //    pushes it to the right on its own.
        var hasTitle = header.textContent.trim().length > 0;

        if (hasTitle) {
          var titleSpan = document.createElement("span");
          titleSpan.className = "math-env-title";
          while (header.firstChild) {
            titleSpan.appendChild(header.firstChild);
          }
          header.appendChild(titleSpan);
        }

        // 2. Build the opaque badge: [icon] [Type N]
        var badge = document.createElement("span");
        badge.className = "math-env-badge";

        var iconSpan = document.createElement("span");
        iconSpan.className = "math-env-icon";
        iconSpan.setAttribute("aria-hidden", "true");
        iconSpan.textContent = t.icon;

        var numSpan = document.createElement("span");
        numSpan.className = "math-env-num";
        numSpan.textContent = t.label + " " + n; // non-breaking space

        badge.appendChild(iconSpan);
        badge.appendChild(numSpan);

        // 3. Append badge at the END of the header — CSS flex + margin-left:auto
        //    on .math-env-badge pushes it to the far right.
        header.appendChild(badge);
      });
    });
  });
})();
