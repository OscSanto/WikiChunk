from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

_SKIP_RE = re.compile(
    r"^(Category:|Template:|Portal:|File:|Help:|Special:|Talk:|"
    r"Wikipedia:|User:|MediaWiki:|Module:|Draft:)",
    re.I,
)


def zim_article_count(zim_path: Path) -> int:
    """Return the total entry count from the ZIM archive (fast, no iteration)."""
    from libzim.reader import Archive
    return Archive(str(zim_path)).article_count


def iter_zim(zim_path: Path) -> Iterator[tuple[str | None, str, str, str | None]]:
    """
    Yield (html, path, title, skip_reason) for every entry in the ZIM file.
    skip_reason is None for valid articles; non-None means the article was skipped.
    """
    from libzim.reader import Archive

    archive = Archive(str(zim_path))
    for i in range(archive.article_count):
        path = title = ""
        try:
            entry = archive._get_entry_by_id(i)
            path  = entry.path
            title = entry.title or path

            if entry.is_redirect:
                yield None, path, title, "redirect"
                continue

            if _SKIP_RE.match(path) or _SKIP_RE.match(title):
                yield None, path, title, "namespace"
                continue

            item = entry.get_item()
            if "html" not in item.mimetype.lower():
                yield None, path, title, f"non-html ({item.mimetype})"
                continue

            html = bytes(item.content).decode("utf-8", errors="replace")
            yield html, path, title, None

        except Exception as exc:
            yield None, path or str(i), title, f"error: {exc}"
