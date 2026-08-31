"""MkDocs hooks: make the repository's relative links work on the site.

README.md is the landing page and the guides link to source files, tests and
experiments. A link to another docs page becomes a page link; a link into the
repository becomes a GitHub URL for the ref being built (``REF``, default
``main``). The PDF link is added only when the PDF was built.
"""

import os
import pathlib
import posixpath
import re

REPO = "https://github.com/tensorchiefs/tramdag"
_LINK = re.compile(r"(\]\()([^)\s#][^)\s]*)(\))")


def on_config(config):
    """Execute the notebooks only when CI asks for it (DOCS_EXECUTE=true)."""
    config.plugins["mkdocs-jupyter"].config["execute"] = (
        os.environ.get("DOCS_EXECUTE", "").lower() == "true"
    )
    return config


def on_page_markdown(markdown, page, config, files):
    """Rewrite repo-relative links of one page; index.md is README.md itself."""
    if page.file.src_uri == "index.md":
        markdown = pathlib.Path("README.md").read_text()
    page_dir = "" if page.file.src_uri == "index.md" else "docs"
    ref = os.environ.get("REF", "main")
    docs_pages = {f.src_uri for f in files}

    def repl(m):
        target = m.group(2)
        if target.startswith(("http://", "https://", "mailto:")):
            return m.group(0)
        path, _, anchor = target.partition("#")
        repo_path = posixpath.normpath(posixpath.join(page_dir, path))
        for prefix, strip in (("docs/", "docs/"), ("notebooks/", "")):
            candidate = (
                repo_path[len(strip) :] if repo_path.startswith(prefix) else None
            )
            if candidate and candidate in docs_pages:
                rel = (
                    posixpath.relpath(candidate, page_dir and ".")
                    if page_dir
                    else candidate
                )
                return f"{m.group(1)}{rel}{'#' + anchor if anchor else ''}{m.group(3)}"
        kind = (
            "tree"
            if repo_path.endswith("/") or "." not in posixpath.basename(repo_path)
            else "blob"
        )
        return f"{m.group(1)}{REPO}/{kind}/{ref}/{repo_path}{m.group(3)}"

    markdown = _LINK.sub(repl, markdown)
    if (
        page.file.src_uri == "index.md"
        and pathlib.Path("docs/tramdag-docs.pdf").exists()
    ):
        markdown = "[Download these docs as one PDF](tramdag-docs.pdf)\n\n" + markdown
    return markdown
