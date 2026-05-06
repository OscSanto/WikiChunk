from __future__ import annotations

import json
import io
from pathlib import Path

import streamlit as st

from wikichunk.cleaner import extract_sections
from wikichunk.chunkers import SectionChunker, FlatChunker, ParagraphChunker
from wikichunk.reader import iter_zim

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="wikichunk",
    page_icon="📄",
    layout="wide",
)

st.title("📄 wikichunk")
st.caption("ZIM / Wikipedia article chunker for RAG pipelines")

# ── Sidebar — settings ────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    strategy = st.selectbox(
        "Chunking strategy",
        options=["section", "flat", "paragraph"],
        index=0,
        help=(
            "**section** — respects Wikipedia section boundaries; long sections are "
            "split with a sliding window.\n\n"
            "**flat** — ignores structure; sliding window over the full article text.\n\n"
            "**paragraph** — splits on paragraph boundaries; short paragraphs are merged."
        ),
    )

    size_mode = st.selectbox(
        "Measure size by",
        options=["characters", "words", "tokens (tiktoken cl100k)"],
        index=0,
        help=(
            "**characters** — default, no extra dependencies.\n\n"
            "**words** — rough approximation (`len(text.split())`).\n\n"
            "**tokens (tiktoken cl100k)** — accurate token count matching GPT-4 / "
            "most modern LLMs. Requires `pip install tiktoken`."
        ),
    )

    if size_mode == "characters":
        size_fn    = len
        size_label = "chars"
    elif size_mode == "words":
        size_fn    = lambda t: len(t.split())
        size_label = "words"
    else:
        try:
            import tiktoken
            _enc    = tiktoken.get_encoding("cl100k_base")
            size_fn = lambda t: len(_enc.encode(t))
            size_label = "tokens"
        except ImportError:
            st.warning("`tiktoken` not installed. Run `pip install tiktoken`. Falling back to characters.")
            size_fn    = len
            size_label = "chars"

    chunk_size = st.slider(f"Max chunk size ({size_label})", min_value=50,  max_value=2000, value=500, step=50)
    if strategy == "paragraph":
        st.slider(f"Overlap ({size_label}) — N/A for paragraph", min_value=0, max_value=400, value=0, step=10, disabled=True)
        overlap = 0
    else:
        overlap = st.slider(f"Overlap ({size_label})", min_value=0, max_value=400, value=50, step=10)
        if overlap >= chunk_size:
            st.warning(f"Overlap ({overlap}) must be less than chunk size ({chunk_size}) — overlap is currently ignored.")
    min_size   = st.slider(f"Min chunk size ({size_label})", min_value=5,   max_value=200,  value=50,  step=5)
    if min_size >= chunk_size:
        st.warning(f"Min chunk size ({min_size}) ≥ chunk size ({chunk_size}) — all chunks will be dropped.")

    st.divider()
    st.subheader("Depth filter")
    depth_filter_on = st.toggle(
        "Filter by section depth",
        value=False,
        disabled=(strategy == "flat"),
        help="Not available for flat strategy — section depth is not preserved.",
    )
    if strategy == "flat":
        st.caption("N/A — flat strategy discards section structure.")
        min_depth = 0
        max_depth = 5
    elif depth_filter_on:
        min_depth = st.slider("Min depth (inclusive)", min_value=0, max_value=5, value=0,
                              help="0 = Lead, 1 = h2 sections, 2 = h3 subsections, …")
        max_depth = st.slider("Max depth (inclusive)", min_value=0, max_value=5, value=5)
        if min_depth > max_depth:
            st.warning("Min depth is greater than max depth.")
    else:
        min_depth = 0
        max_depth = 5

    st.divider()
    st.subheader("Content filters")
    keep_tables   = st.toggle("Keep tables",                   value=False)
    keep_captions = st.toggle("Keep image captions",           value=False)
    keep_infobox  = st.toggle("Keep infoboxes (grouped by structure)", value=False,
                               help=(
                                   "Extracts the infobox as small grouped chunks using Wikipedia's "
                                   "own section headers as boundaries. Each group becomes one chunk "
                                   "with section = 'Infobox (<header>)', depth 0."
                               ))

    st.divider()
    expand_sections = st.toggle("Expand all sections", value=False)
    expand_chunks   = st.toggle("Expand all chunks",   value=False)

    st.divider()
    st.markdown("**wikichunk v0.1.0**")

# ── ZIM path input ────────────────────────────────────────────────────────────

zim_path_str = st.text_input(
    "ZIM file path",
    placeholder="/library/zims/content/wikipedia_en_medicine_maxi_2026-04.zim",
    help="Absolute path to your local ZIM file.",
)

# ── Load ZIM and pick article ─────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ZIM article list…")
def load_article_list(zim_path: str) -> list[tuple[str, str]]:
    """Return list of (path, title) for all non-redirect articles."""
    from libzim.reader import Archive
    import re
    _SKIP = re.compile(
        r"^(Category:|Template:|Portal:|File:|Help:|Special:|Talk:|"
        r"Wikipedia:|User:|MediaWiki:|Module:|Draft:)", re.I,
    )
    archive = Archive(zim_path)
    articles = []
    for i in range(archive.article_count):
        try:
            entry = archive._get_entry_by_id(i)
            if entry.is_redirect:
                continue
            if _SKIP.match(entry.path) or _SKIP.match(entry.title or ""):
                continue
            articles.append((entry.path, entry.title or entry.path))
        except Exception:
            continue
    return articles


@st.cache_data(show_spinner="Fetching article HTML…")
def fetch_article_html(zim_path: str, article_path: str) -> str:
    from libzim.reader import Archive
    archive = Archive(zim_path)
    entry   = archive.get_entry_by_path(article_path)
    item    = entry.get_item()
    return bytes(item.content).decode("utf-8", errors="replace")


if zim_path_str:
    zim_path = Path(zim_path_str)
    if not zim_path.exists():
        st.error(f"File not found: `{zim_path_str}`")
        st.stop()
    if zim_path.is_dir():
        st.error(f"That path is a directory, not a ZIM file. Did you mean `{zim_path_str}.zim`?")
        st.stop()
    if zim_path.suffix.lower() != ".zim":
        st.warning(f"Path doesn't end in `.zim` — make sure you're pointing to the ZIM file itself.")

    articles = load_article_list(zim_path_str)
    if not articles:
        st.error("No articles found in this ZIM file.")
        st.stop()

    titles     = [title for _, title in articles]
    path_by_title = {title: path for path, title in articles}

    selected_title = st.selectbox(
        f"Select article  ({len(articles):,} available)",
        options=titles,
    )
    selected_path = path_by_title[selected_title]

    # ── Process selected article ──────────────────────────────────────────────

    html = fetch_article_html(zim_path_str, selected_path)

    sections, elem_stats = extract_sections(
        html,
        keep_tables=keep_tables,
        keep_captions=keep_captions,
        keep_infobox=keep_infobox,
    )

    if strategy == "section":
        chunker = SectionChunker(chunk_size=chunk_size, chunk_overlap=overlap, min_chunk_size=min_size, size_fn=size_fn)
    elif strategy == "flat":
        chunker = FlatChunker(chunk_size=chunk_size, chunk_overlap=overlap, min_chunk_size=min_size, size_fn=size_fn)
    else:
        chunker = ParagraphChunker(max_chunk_size=chunk_size, min_chunk_size=min_size, size_fn=size_fn)

    all_chunks = chunker.chunk(selected_path, selected_title, sections)

    # Apply depth filter
    chunks           = [c for c in all_chunks  if min_depth <= c.depth <= max_depth]
    visible_sections = [s for s in sections    if min_depth <= s.depth <= max_depth]

    # ── Stats bar ─────────────────────────────────────────────────────────────

    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sections",  f"{len(visible_sections)} / {len(sections)}")
    c2.metric("Chunks",    f"{len(chunks)} / {len(all_chunks)}")
    c3.metric(f"Avg chunk ({size_label})", f"{sum(size_fn(c.text) for c in chunks) // max(len(chunks), 1)}")
    removed_count = (
        (0 if keep_infobox else elem_stats["infoboxes"])
        + elem_stats["navboxes"]
        + elem_stats["images"]
        + elem_stats["tables"]
    )
    c4.metric("Removed elements", removed_count)
    c5.metric("Strategy", strategy)

    st.divider()

    # ── Two-column layout ─────────────────────────────────────────────────────

    left, right = st.columns(2)

    with left:
        st.subheader("Extracted sections")
        infobox_label = (
            f"Infoboxes extracted: {elem_stats['infoboxes']}"
            if keep_infobox else
            f"Infoboxes removed: {elem_stats['infoboxes']}"
        )
        st.caption(
            f"{infobox_label}  ·  "
            f"Navboxes: {elem_stats['navboxes']}  ·  "
            f"Images: {elem_stats['images']}  ·  "
            f"Tables: {elem_stats['tables']}"
        )
        if not visible_sections:
            st.warning("No sections match the current depth filter.")
        for sec in visible_sections:
            depth_indent = "—" * sec.depth + " " if sec.depth else ""
            is_open = expand_sections or sec.title == "Lead"
            with st.expander(
                f"{depth_indent}**{sec.title}**  ·  depth {sec.depth}  ·  "
                f"{len(sec.text)} chars  ·  {size_fn(sec.text)} {size_label}",
                expanded=is_open,
            ):
                st.text(sec.text[:1000] + ("…" if len(sec.text) > 1000 else ""))

    with right:
        st.subheader("Chunks")
        st.caption(f"{len(chunks)} chunks produced using **{strategy}** strategy")
        if not chunks:
            st.warning("No chunks match the current depth filter. Try widening the depth range.")
        for chunk in chunks:
            if strategy == "flat":
                label = (
                    f"[{chunk.chunk_index}]  ·  "
                    f"{chunk.char_count} chars  ·  {size_fn(chunk.text)} {size_label}"
                )
            else:
                label = (
                    f"[{chunk.chunk_index}] §{chunk.section}  ·  "
                    f"depth {chunk.depth}  ·  {chunk.char_count} chars  ·  "
                    f"{size_fn(chunk.text)} {size_label}"
                )
            with st.expander(label, expanded=expand_chunks):
                st.text(chunk.text)

    # ── Download ──────────────────────────────────────────────────────────────

    st.divider()

    # ── Single article ────────────────────────────────────────────────────────
    st.subheader("Download — this article")
    st.caption(
        f"Exports only **{selected_title}** with current settings."
        + (
            f"  Depth filter active: {len(chunks)} / {len(all_chunks)} chunks, "
            f"{len(visible_sections)} / {len(sections)} sections."
            if depth_filter_on and strategy != "flat" else ""
        )
    )

    chunks_jsonl   = "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks)
    sections_jsonl = "\n".join(
        json.dumps({"title": s.title, "depth": s.depth, "text": s.text}, ensure_ascii=False)
        for s in visible_sections
    )

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        label="⬇ chunks.jsonl",
        data=chunks_jsonl,
        file_name=f"{selected_title.replace(' ', '_')}_chunks.jsonl",
        mime="application/jsonl",
    )
    dl2.download_button(
        label="⬇ sections.jsonl",
        data=sections_jsonl,
        file_name=f"{selected_title.replace(' ', '_')}_sections.jsonl",
        mime="application/jsonl",
    )

    # ── Full ZIM export ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Export — full ZIM")
    st.caption(
        "Processes every article in the ZIM with the current settings and saves "
        "`chunks.jsonl` + `skipped.jsonl` to an output directory on disk. "
        "Output can be several GB — not suitable for browser download."
    )

    default_out = str(Path.home() / "wikichunk_output" / zim_path.stem)
    out_dir_str = st.text_input(
        "Output directory",
        value=default_out,
        help="Directory where chunks.jsonl and skipped.jsonl will be written.",
    )

    if st.button("▶ Run full ZIM export", type="primary"):
        out_dir = Path(out_dir_str)
        try:
            from wikichunk.pipeline import WikiChunker
            from libzim.reader import Archive as _Archive

            total        = _Archive(str(zim_path)).article_count
            progress_bar = st.progress(0, text="Starting…")
            status_text  = st.empty()

            def _tick(stats: dict) -> None:
                n   = stats["articles_scanned"]
                pct = min(n / max(total, 1), 1.0)
                progress_bar.progress(
                    pct,
                    text=f"{n:,} / {total:,} articles scanned",
                )
                status_text.caption(
                    f"chunks: {stats['chunks_produced']:,}  ·  "
                    f"processed: {stats['articles_processed']:,}  ·  "
                    f"skipped: {stats['articles_skipped']:,}"
                )

            result = WikiChunker(
                zim_path=zim_path,
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                min_chunk_size=min_size,
                size_fn=size_fn,
                size_label=size_label,
                keep_tables=keep_tables,
                keep_captions=keep_captions,
                keep_infobox=keep_infobox,
                output_dir=out_dir,
                tick_cb=_tick,
                tick_every=200,
            ).run()

            progress_bar.progress(1.0, text="Done!")
            status_text.empty()
            st.success(
                f"Export complete — "
                f"**{result['chunks_produced']:,} chunks** from "
                f"**{result['articles_processed']:,} articles**"
            )
            st.code(f"chunks  → {out_dir}/chunks.jsonl\nskipped → {out_dir}/skipped.jsonl")

        except Exception as e:
            st.error(f"Export failed: {e}")

else:
    st.info("Enter a ZIM file path above to get started.")
