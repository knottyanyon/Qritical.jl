(function () {
    var THRESHOLD = 15; // outputs with more lines than this get folded

    function foldOutputs() {
        document.querySelectorAll("pre:has(> code.nohighlight)").forEach(function (pre) {
            var code = pre.querySelector("code.nohighlight");
            if (!code) return;
            var lines = code.textContent.trimEnd().split("\n").length;
            if (lines <= THRESHOLD) return;

            var details = document.createElement("details");
            details.className = "output-fold";

            var summary = document.createElement("summary");
            summary.textContent = lines + " lines";

            pre.parentNode.insertBefore(details, pre);
            details.appendChild(summary);
            details.appendChild(pre);
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", foldOutputs);
    } else {
        foldOutputs();
    }
})();
