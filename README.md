# wikichunk

ZIM/Wikipedia article chunker for RAG pipelines. Reads a `.zim` file, cleans HTML, splits articles into overlapping text chunks, and writes them to `chunks.jsonl` — ready for embedding with **wikembed**.

```
wikichunk → wikembed → wikibench
   ZIM         FAISS       Benchmarks
```

---

## Full pipeline walkthrough

```bash
# 1 — Chunk a ZIM file
wikichunk wikipedia_en_medicine.zim --strategy section --chunk-size 500 --overlap 50

# 2 — Embed chunks into a FAISS index  (run in wikembed repo)
wikembed output/wikipedia_en_medicine/section_500_50/chunks/chunks.jsonl

# 3 — Run a benchmark on your Pi  (run in wikibench repo)
wikibench run configs/my_run.yaml
```

The output directory from wikichunk (`section_500_50/chunks/`) feeds directly into wikembed as the `chunks.jsonl` argument. wikembed writes `faiss.index` and `chunks.db` one level above — no manual file moving required.

---

## Installation

```bash
pip install -e .
```

Requires Python 3.10+.

---

## Quick start

```bash
# Default: section strategy, 500-char chunks, 50-char overlap
wikichunk wikipedia_en_medicine.zim

# Flat chunking, 300 words
wikichunk wikipedia_en_medicine.zim --strategy flat --chunk-size 300

# Custom output directory
wikichunk wikipedia_en_medicine.zim --out /data/chunks
```

Or launch the Streamlit UI:

```bash
streamlit run app.py
```

---

## Output structure

```
output/
└── wikipedia_en_medicine/          ← ZIM stem
    └── section_500_50/             ← strategy_chunksize_overlap
        └── chunks/
            ├── chunks.jsonl        ← one JSON object per line
            ├── skipped.jsonl       ← articles that were skipped + reason
            └── .done               ← written on successful completion
```

Each line of `chunks.jsonl` is:

```json
{
  "article":     "Pneumonia",
  "path":        "A/Pneumonia",
  "section":     "Pathophysiology",
  "depth":       1,
  "chunk_index": 3,
  "text":        "Inflammatory exudates accumulate...",
  "char_count":  412
}
```

---

## Chunking strategies

| Strategy    | Description | Best for |
|-------------|-------------|----------|
| `section`   | Keeps Wikipedia section boundaries intact. Sections within chunk_size are kept whole; longer sections are split with a sliding window. | General use (default) |
| `flat`      | Ignores section structure. Sliding window over the full article text. | Dense retrieval, max coverage |
| `paragraph` | Splits on paragraph boundaries (double newline). Short paragraphs are merged; long ones are split. | Precise passage retrieval |

---

## CLI reference

```
wikichunk <zim> [options]

Arguments:
  zim                       Path to ZIM file

Options:
  --strategy    {section,flat,paragraph}
                            Chunking strategy (default: section)
  --chunk-size  INT         Max chunk size in characters (default: 500)
  --overlap     INT         Character overlap between consecutive chunks,
                            section/flat only (default: 50)
  --min-size    INT         Drop chunks smaller than this (default: 50)
  --keep-tables             Keep Wikipedia tables in output
  --keep-captions           Keep image captions
  --keep-infobox            Extract infoboxes as grouped chunks
  --out         DIR         Output directory
                            (default: output/<zim-stem>/)
```

---

## Resume behaviour

If a run is interrupted, re-running with the same arguments picks up where it left off — articles already written to `chunks.jsonl` are skipped. Delete `chunks/` to start fresh.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `libzim` | Read `.zim` archives |
| `beautifulsoup4` | HTML cleaning and section extraction |
| `streamlit` | Web UI (`app.py`) |
