from __future__ import annotations

import re
from bs4 import BeautifulSoup, Tag
from wikichunk.models import Section

_HEADING_TAGS = {"h2", "h3", "h4", "h5", "h6"}

_STOP_SECTIONS = re.compile(
    r"^(references|further reading|external links|see also|notes|bibliography|citations)$",
    re.I,
)


def _depth(tag_name: str) -> int:
    return int(tag_name[1]) - 1  # h2→1, h3→2, h4→3


def _cls(tag: Tag, fragment: str) -> bool:
    """True if any CSS class on tag contains fragment."""
    return any(fragment in c for c in (tag.get("class") or []))


def _infobox_to_sections(table: Tag) -> list[Section]:
    """
    Parse a Wikipedia infobox using its own header rows as group boundaries.

    Handles both row formats emitted by Wikipedia templates:
      Old: <th scope="row">Label</th><td>Value</td>
      New: <td class="infobox-label">Label</td><td class="infobox-data">Value</td>
    Section headers appear as either a lone <th> or a <td class="infobox-header">.
    """
    sections: list[Section] = []
    current_header = "Info"
    current_rows: list[tuple[str, str]] = []

    def _flush() -> None:
        if current_rows:
            text = "\n".join(f"{k}: {v}" for k, v in current_rows)
            sections.append(Section(title=f"Infobox ({current_header})", depth=0, text=text))

    for tr in table.find_all("tr"):
        th  = tr.find("th")
        tds = tr.find_all("td")

        # ── Detect row type ───────────────────────────────────────────────────

        # New format: section header cell
        header_td = next((t for t in tds if _cls(t, "infobox-header")), None)
        # New format: label + data cells
        label_td  = next((t for t in tds if _cls(t, "infobox-label")), None)
        data_td   = next((t for t in tds if _cls(t, "infobox-data")),  None)

        if header_td:
            # New format: explicit infobox-header cell
            label = header_td.get_text(" ", strip=True)
            if label and label != current_header:
                _flush()
                current_rows = []
                current_header = label

        elif label_td and data_td:
            # New format: infobox-label + infobox-data cells
            key = label_td.get_text(" ", strip=True)
            val = data_td.get_text(" ", strip=True)
            if key and val:
                current_rows.append((key, val))

        elif th and not tds:
            # Old/Kiwix format: lone <th> = section header
            label = th.get_text(" ", strip=True)
            if label and label != current_header:
                _flush()
                current_rows = []
                current_header = label

        elif th and tds:
            # Old format: <th> key + <td> value
            key = th.get_text(" ", strip=True)
            val = tds[0].get_text(" ", strip=True)
            if key and val:
                current_rows.append((key, val))

        elif not th and len(tds) == 2:
            # Kiwix plain td/td format: first cell is key, second is value
            key = tds[0].get_text(" ", strip=True).rstrip(":")
            val = tds[1].get_text(" ", strip=True)
            if key and val:
                current_rows.append((key, val))

    _flush()
    return sections


def extract_sections(
    html: str,
    keep_tables:   bool = False,
    keep_captions: bool = False,
    keep_infobox:  bool = False,
) -> tuple[list[Section], dict]:
    """
    Clean Wikipedia HTML and extract a list of Section objects.
    Returns (sections, removal_stats).
    """
    soup = BeautifulSoup(html, "html.parser")
    stats = {"infoboxes": 0, "navboxes": 0, "images": 0, "tables": 0, "refs_inline": 0}

    root = (
        soup.find("div", class_="mw-parser-output")
        or soup.find("div", id="mw-content-text")
        or soup.body
        or soup
    )

    # Infoboxes — extract as structure-grouped sections or discard
    # class_ lambda checks each individual class token so "infobox-3" etc. also match
    infobox_sections: list[Section] = []
    for table in root.find_all("table", class_=lambda c: c and "infobox" in c):
        stats["infoboxes"] += 1
        if keep_infobox:
            infobox_sections.extend(_infobox_to_sections(table))
        table.decompose()

    # Navboxes
    for tag in root.find_all("table", class_=lambda c: c and "navbox" in c):
        stats["navboxes"] += 1
        tag.decompose()

    # Images and figures
    for tag in root.find_all(["figure", "img"]):
        stats["images"] += 1
        if keep_captions:
            cap = tag.find("figcaption")
            tag.replace_with(cap) if cap else tag.decompose()
        else:
            tag.decompose()
    for tag in root.find_all("div", class_="thumb"):
        stats["images"] += 1
        tag.decompose()

    # Inline citation markers [1], [2]
    for tag in root.find_all("sup", class_="reference"):
        stats["refs_inline"] += 1
        tag.decompose()

    # Reference lists and reflist divs
    for tag in root.find_all("ol", class_="references"):
        tag.decompose()
    for tag in root.find_all("div", class_=lambda c: c and "reflist" in c):
        tag.decompose()

    # General tables
    if not keep_tables:
        for tag in root.find_all("table"):
            stats["tables"] += 1
            tag.decompose()

    # Misc junk
    for sel in ["div.hatnote", "div.dablink", "span.mw-editsection", "style", "script"]:
        for tag in root.select(sel):
            tag.decompose()

    # Extract article sections
    sections: list[Section] = []
    current_title = "Lead"
    current_depth = 0
    current_parts: list[str] = []

    def _flush() -> None:
        text = "\n\n".join(p.strip() for p in current_parts if p.strip())
        if text:
            sections.append(Section(title=current_title, depth=current_depth, text=text))

    for child in root.find_all(["h2", "h3", "h4", "h5", "h6", "p", "li", "dt", "dd"], recursive=True):
        name = (child.name or "").lower()
        if name in _HEADING_TAGS:
            _flush()
            current_parts = []
            heading = child.get_text(" ", strip=True)
            if _STOP_SECTIONS.match(heading.strip()):
                break
            current_title = heading
            current_depth = _depth(name)
        else:
            text = child.get_text(" ", strip=True)
            if text:
                current_parts.append(text)

    _flush()

    # Infobox sections come first (before Lead)
    return infobox_sections + sections, stats
