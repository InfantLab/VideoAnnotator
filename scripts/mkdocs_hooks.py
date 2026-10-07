"""mkdocs hook: links that leave the site point at the file on GitHub.

The docs link to the code, the specs, CONTRIBUTING.md and the archive with
relative paths, which work when reading them on GitHub. The site holds only
docs/ (without the archive), so those links are rewritten to GitHub URLs when
the site is built; the Markdown files are untouched.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path

LINK = re.compile(r"(\]\(\s*<?)([^)\s>#?]+)([^)\s>]*>?(?:\s+\"[^\"]*\")?\s*\))")
EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//|#)", re.IGNORECASE)
EXCLUDED = ("archive/", "development/handover_")


def on_page_markdown(markdown, page, config, files):
    docs_dir = Path(config["docs_dir"]).resolve()
    repo_root = docs_dir.parent
    blob = config["repo_url"].rstrip("/") + "/blob/master/"
    tree = config["repo_url"].rstrip("/") + "/tree/master/"
    page_dir = posixpath.dirname(page.file.src_uri)

    def rewrite(match: re.Match) -> str:
        target = match.group(2)
        if EXTERNAL.match(target) or target.startswith("/"):
            return match.group(0)
        in_docs = posixpath.normpath(posixpath.join(page_dir, target))
        leaves_site = in_docs.startswith("../") or in_docs.startswith(EXCLUDED)
        if not leaves_site:
            return match.group(0)
        path = (docs_dir / in_docs).resolve()
        if not path.exists() or repo_root not in (path, *path.parents):
            return match.group(0)
        rel = path.relative_to(repo_root).as_posix()
        return (
            f"{match.group(1)}{(tree if path.is_dir() else blob) + rel}{match.group(3)}"
        )

    return LINK.sub(rewrite, markdown)
