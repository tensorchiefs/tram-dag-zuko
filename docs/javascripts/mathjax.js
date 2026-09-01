window.MathJax = {
  loader: { load: ["[tex]/boldsymbol"] },
  tex: {
    packages: { "[+]": ["boldsymbol"] },
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    // arithmatex: the guides; jupyter-wrapper: mkdocs-jupyter renders the
    // notebook markdown itself, so its math never gets an arithmatex span
    processHtmlClass: "arithmatex|jupyter-wrapper",
  },
};
// the first document$ emission fires before the MathJax script (loaded after
// this file) exists — guard, and let the CDN's own startup typeset handle the
// initial render
document$.subscribe(() => {
  if (window.MathJax && MathJax.typesetPromise) {
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  }
});
