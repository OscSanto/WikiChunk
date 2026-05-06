from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

from wikichunk.models import Chunk, SkippedItem
from wikichunk.reader import iter_zim
from wikichunk.cleaner import extract_sections
from wikichunk.chunkers import SectionChunker, FlatChunker, ParagraphChunker

_STRATEGIES = {
    "section":   SectionChunker,
    "flat":      FlatChunker,
    "paragraph": ParagraphChunker,
}

_BAR = "━" * 51


class WikiChunker:
    def __init__(
        self,
        zim_path:       str | Path,
        strategy:       str                  = "section",
        chunk_size:     int                  = 500,
        chunk_overlap:  int                  = 50,
        min_chunk_size: int                  = 50,
        size_fn:        Callable[[str], int] = len,
        size_label:     str                  = "chars",
        keep_tables:    bool                 = False,
        keep_captions:  bool                 = False,
        keep_infobox:   bool                 = False,
        output_dir:     str | Path | None    = None,
        tick_cb:        Callable | None      = None,
        tick_every:     int                  = 200,
    ):
        self.zim_path       = Path(zim_path)
        self.strategy       = strategy
        self.chunk_size     = chunk_size
        self.chunk_overlap  = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.size_fn        = size_fn
        self.size_label     = size_label
        self.keep_tables    = keep_tables
        self.keep_captions  = keep_captions
        self.keep_infobox   = keep_infobox
        self.output_dir     = Path(output_dir) if output_dir else Path("output") / self.zim_path.stem
        self.tick_cb        = tick_cb
        self.tick_every     = tick_every

        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}. Choose: {list(_STRATEGIES)}")

        cls    = _STRATEGIES[strategy]
        kwargs = dict(size_fn=size_fn, min_chunk_size=min_chunk_size)
        if strategy in ("section", "flat"):
            kwargs.update(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            kwargs["max_chunk_size"] = chunk_size
        self._chunker = cls(**kwargs)

    def _print_header(self, resuming: bool, resume_count: int) -> None:
        skipped_els = ["navboxes", "references"]
        if not self.keep_tables:   skipped_els.append("tables")
        if not self.keep_captions: skipped_els.append("images")
        if not self.keep_infobox:  skipped_els.append("infoboxes")

        u = self.size_label
        print(f"\nwikichunk v0.1.0")
        print(_BAR)
        print(f"ZIM file    : {self.zim_path.name}")
        print(f"Strategy    : {self.strategy}Chunker")
        print(f"Chunk size  : {self.chunk_size} {u}")
        print(f"Overlap     : {self.chunk_overlap} {u}")
        print(f"Min size    : {self.min_chunk_size} {u}")
        print(f"Skipping    : {', '.join(skipped_els)}")
        print(f"Output dir  : {self.output_dir}/")
        if resuming:
            print(f"Resuming    : {resume_count:,} articles already done")
        print(_BAR)

    def _load_seen(self, *paths: Path) -> set[str]:
        """Read article paths already written to any of the given JSONL files."""
        seen: set[str] = set()
        for p in paths:
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        seen.add(json.loads(line)["path"])
                    except Exception:
                        pass
        return seen

    def run(self) -> dict:
        chunks_path  = self.output_dir / "chunks.jsonl"
        skipped_path = self.output_dir / "skipped.jsonl"
        done_path    = self.output_dir / ".done"

        # ── Detect state: fresh / resume / already-complete ───────────────────
        if done_path.exists():
            print(
                f"\nOutput already complete at {self.output_dir}\n"
                f"Delete {done_path} to re-run from scratch."
            )
            return {}

        resuming = chunks_path.exists() or skipped_path.exists()
        seen: set[str] = set()
        if resuming:
            seen = self._load_seen(chunks_path, skipped_path)
            print(f"\nIncomplete previous run — resuming ({len(seen):,} articles already done).")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._print_header(resuming, len(seen))

        stats = dict(
            articles_scanned=0, articles_processed=0, articles_skipped=0,
            articles_resumed=len(seen),
            chunks_produced=0,  chars_total=0,
            redirects=0, namespaces=0, empty=0, errors=0,
            removed_infoboxes=0, removed_navboxes=0,
            removed_images=0,   removed_tables=0,
        )

        t0 = time.time()
        file_mode = "a" if resuming else "w"
        print("Processing...")

        with open(chunks_path,  file_mode, encoding="utf-8") as cf, \
             open(skipped_path, file_mode, encoding="utf-8") as sf:

            for html, path, title, skip_reason in iter_zim(self.zim_path):
                stats["articles_scanned"] += 1

                # ── Skip already-processed articles on resume ─────────────────
                if path in seen:
                    continue

                # ── Skipped at ZIM level ──────────────────────────────────────
                if skip_reason is not None:
                    stats["articles_skipped"] += 1
                    if "redirect"   in skip_reason: stats["redirects"]  += 1
                    elif "namespace" in skip_reason: stats["namespaces"] += 1
                    elif "error"    in skip_reason: stats["errors"]    += 1
                    sf.write(json.dumps(
                        SkippedItem("article", path, title, skip_reason).to_dict(),
                        ensure_ascii=False,
                    ) + "\n")
                    continue

                # ── Clean + chunk ─────────────────────────────────────────────
                try:
                    sections, elem_stats = extract_sections(
                        html,
                        keep_tables=self.keep_tables,
                        keep_captions=self.keep_captions,
                        keep_infobox=self.keep_infobox,
                    )
                    stats["removed_infoboxes"] += elem_stats["infoboxes"]
                    stats["removed_navboxes"]  += elem_stats["navboxes"]
                    stats["removed_images"]    += elem_stats["images"]
                    stats["removed_tables"]    += elem_stats["tables"]

                    chunks = self._chunker.chunk(path, title, sections)
                except Exception as exc:
                    stats["articles_skipped"] += 1
                    stats["errors"] += 1
                    sf.write(json.dumps(
                        SkippedItem("article", path, title, f"error: {exc}").to_dict(),
                        ensure_ascii=False,
                    ) + "\n")
                    continue

                if not chunks:
                    stats["articles_skipped"] += 1
                    stats["empty"] += 1
                    sf.write(json.dumps(
                        SkippedItem("article", path, title, "empty_after_cleaning").to_dict(),
                        ensure_ascii=False,
                    ) + "\n")
                    continue

                stats["articles_processed"] += 1
                for chunk in chunks:
                    stats["chunks_produced"] += 1
                    stats["chars_total"]     += chunk.char_count
                    cf.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

                # ── Progress ticker ───────────────────────────────────────────
                n = stats["articles_scanned"]
                if n % self.tick_every == 0:
                    elapsed = time.time() - t0
                    rate    = n / elapsed if elapsed > 0 else 0
                    sys.stdout.write(
                        f"\r  scanned {n:,}  |  "
                        f"chunks {stats['chunks_produced']:,}  |  "
                        f"{rate:.0f} art/s   "
                    )
                    sys.stdout.flush()
                    if self.tick_cb:
                        self.tick_cb(stats)

        # Mark run as complete
        done_path.write_text(
            json.dumps({"completed": time.strftime("%Y-%m-%dT%H:%M:%S"), **stats}) + "\n"
        )

        self._print_summary(stats, time.time() - t0, chunks_path, skipped_path)
        return stats

    def _print_summary(self, s: dict, elapsed: float, chunks_path: Path, skipped_path: Path) -> None:
        avg = s["chars_total"] // max(s["chunks_produced"], 1)
        print(f"\n\nResults")
        print(_BAR)
        if s.get("articles_resumed"):
            print(f"Resumed from        : {s['articles_resumed']:,} articles (previous run)")
        print(f"Articles scanned    : {s['articles_scanned']:,}")
        print(f"  Redirects         : {s['redirects']:,}")
        print(f"  Namespace pages   : {s['namespaces']:,}")
        print(f"  Empty after clean : {s['empty']:,}")
        print(f"  Errors            : {s['errors']:,}")
        print(f"Articles processed  : {s['articles_processed']:,}")
        print(f"\nChunks produced     : {s['chunks_produced']:,}")
        print(f"Avg chunk size      : {avg} {self.size_label}")
        print(f"\nElements removed")
        print(f"  Infoboxes         : {s['removed_infoboxes']:,}")
        print(f"  Navboxes          : {s['removed_navboxes']:,}")
        print(f"  Images/figures    : {s['removed_images']:,}")
        print(f"  Tables            : {s['removed_tables']:,}")
        print(f"\nDone in {elapsed:.1f}s")
        print(f"  chunks  → {chunks_path}")
        print(f"  skipped → {skipped_path}")
        print(_BAR)
