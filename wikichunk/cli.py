from __future__ import annotations

import argparse
from pathlib import Path
from wikichunk.pipeline import WikiChunker


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wikichunk",
        description="wikichunk — ZIM/Wikipedia article chunker for RAG pipelines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("zim",               help="Path to ZIM file")
    parser.add_argument("--strategy",        default="section",
                        choices=["section", "flat", "paragraph"],
                        help="Chunking strategy")
    parser.add_argument("--chunk-size",      type=int, default=500,
                        help="Max chunk size in characters")
    parser.add_argument("--overlap",         type=int, default=50,
                        help="Character overlap between consecutive chunks (section/flat only)")
    parser.add_argument("--min-size",        type=int, default=50,
                        help="Minimum chunk size — smaller chunks are dropped")
    parser.add_argument("--keep-tables",     action="store_true",
                        help="Keep Wikipedia tables in output")
    parser.add_argument("--keep-captions",   action="store_true",
                        help="Keep image captions in output")
    parser.add_argument("--keep-infobox",    action="store_true",
                        help="Extract infoboxes as grouped chunks instead of discarding them")
    parser.add_argument("--out",             default=None,
                        help="Output directory (default: output/<zim-stem>/)")
    args = parser.parse_args()

    WikiChunker(
        zim_path=Path(args.zim),
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        min_chunk_size=args.min_size,
        size_label="chars",
        keep_tables=args.keep_tables,
        keep_captions=args.keep_captions,
        keep_infobox=args.keep_infobox,
        output_dir=args.out,
    ).run()


if __name__ == "__main__":
    main()
