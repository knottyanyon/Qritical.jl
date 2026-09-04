/* page-toc.js — two enhancements for Qritical.jl docs
 *
 * 1. Breadcrumb fixer: Documenter renders parent section crumbs as
 *    <a class="is-disabled"> with no href.  This reads the sidebar
 *    to find the first page in each section and turns those crumbs
 *    into real clickable links.
 *
 * 2. On-page TOC: injects <aside id="qr-page-toc"> into <body> as a
 *    position:fixed panel in the blank space to the right of .docs-main
 *    (which has max-width:52rem).  Builds a heading list from h2/h3 in
 *    the page article and updates the active item on scroll.
 */
(function () {
    "use strict";

    /* ── 1. Breadcrumb fixer ─────────────────────────────────────── */
    function fixBreadcrumbs() {
        var crumbs = document.querySelectorAll("nav.breadcrumb a.is-disabled");
        if (!crumbs.length) return;

        /* Build section-name → first child URL from the sidebar menu */
        var sectionMap = {};
        var spans = document.querySelectorAll(
            "nav.docs-sidebar ul.docs-menu > li > span.tocitem"
        );
        spans.forEach(function (span) {
            var name = span.textContent.trim();
            var li   = span.closest("li");
            var link = li && li.querySelector("a.tocitem");
            if (link) sectionMap[name] = link.href;
        });

        crumbs.forEach(function (a) {
            var url = sectionMap[a.textContent.trim()];
            if (url) {
                a.href = url;
                a.classList.remove("is-disabled");
                a.style.pointerEvents = "auto";
                a.style.cursor = "pointer";
            }
        });
    }

    /* ── 2. On-page TOC ──────────────────────────────────────────── */
    function buildPageTOC() {
        var article = document.getElementById("documenter-page");
        if (!article) return;

        var headings = Array.prototype.slice.call(
            article.querySelectorAll("h2[id], h3[id]")
        );
        if (headings.length < 2) return;

        var aside = document.createElement("aside");
        aside.id = "qr-page-toc";
        aside.setAttribute("aria-label", "On this page");

        var label = document.createElement("p");
        label.className = "qr-toc-label";
        label.textContent = "On this page";
        aside.appendChild(label);

        var ul = document.createElement("ul");
        ul.className = "qr-toc-list";

        var items = [];

        headings.forEach(function (h) {
            var anchor = h.querySelector("a.docs-heading-anchor");
            if (!anchor) return;
            var text = anchor.textContent.trim();
            if (!text) return;

            var li = document.createElement("li");
            li.className = h.tagName === "H3" ? "qr-toc-h3" : "qr-toc-h2";

            var a = document.createElement("a");
            a.href = "#" + h.id;
            a.textContent = text;

            li.appendChild(a);
            ul.appendChild(li);
            items.push({ li: li, heading: h });
        });

        if (!items.length) return;
        aside.appendChild(ul);

        /* Inject into body — it is position:fixed so flow placement
           doesn't matter; body avoids any overflow:hidden on .docs-main */
        document.body.appendChild(aside);

        /* Scroll spy — throttled via requestAnimationFrame */
        var ticking = false;

        function updateActive() {
            var scrollY   = window.scrollY + 110;
            var activeIdx = -1;

            for (var i = 0; i < items.length; i++) {
                if (items[i].heading.getBoundingClientRect().top + window.scrollY <= scrollY) {
                    activeIdx = i;
                } else {
                    break;
                }
            }

            items.forEach(function (item, idx) {
                item.li.classList.toggle("is-active", idx === activeIdx);
            });
        }

        window.addEventListener("scroll", function () {
            if (!ticking) {
                window.requestAnimationFrame(function () {
                    updateActive();
                    ticking = false;
                });
                ticking = true;
            }
        }, { passive: true });

        updateActive();
    }

    /* ── 3. Version badge ────────────────────────────────────────── */
    function injectVersionBadge() {
        var pkgNameDiv = document.querySelector(".docs-package-name");
        if (!pkgNameDiv) return;

        /* Read version from the version selector Documenter already renders */
        var sel = document.getElementById("documenter-version-selector");
        var ver = sel && sel.options[sel.selectedIndex]
            ? sel.options[sel.selectedIndex].text.trim()
            : null;
        if (!ver) return;

        var badge = document.createElement("div");
        badge.className = "qr-version-badge";
        badge.setAttribute("aria-label", "Package version " + ver);

        var left  = document.createElement("span");
        left.className = "qr-badge-left";
        left.textContent = "docs";

        var right = document.createElement("span");
        right.className = "qr-badge-right";
        right.textContent = "v" + ver;

        badge.appendChild(left);
        badge.appendChild(right);
        /* Place between logo and package name */
        pkgNameDiv.insertAdjacentElement("beforebegin", badge);
    }

    function init() {
        fixBreadcrumbs();
        buildPageTOC();
        injectVersionBadge();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
}());
