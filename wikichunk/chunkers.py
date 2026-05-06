from __future__ import annotations

from typing import Callable
from wikichunk.models import Chunk, Section


def _sliding_window(
    text: str,
    chunk_size: int,
    overlap: int,
    size_fn: Callable[[str], int] = len,
) -> list[str]:
    """
    Split text into overlapping chunks, always splitting at word boundaries.
    size_fn measures the 'size' of a string — chars, words, or tokens.
    overlap is in the same units as chunk_size.
    """
    if not text.strip():
        return []
    words = text.split()
    results = []
    i = 0

    while i < len(words):
        # Greedily add words until the next word would exceed chunk_size
        j = i + 1
        while j < len(words) and size_fn(" ".join(words[i : j + 1])) <= chunk_size:
            j += 1
        chunk = " ".join(words[i:j])
        if chunk:
            results.append(chunk)
        if j >= len(words):
            break
        # Step back from j to find next start such that overlap is preserved
        if overlap > 0:
            next_i = j
            for k in range(j - 1, i, -1):
                if size_fn(" ".join(words[k:j])) >= overlap:
                    next_i = k
                    break
            i = max(next_i, i + 1)  # always advance to prevent infinite loop
        else:
            i = j

    return [r for r in results if r]


class SectionChunker:
    """
    Default. Keeps Wikipedia section boundaries intact.
    Sections within chunk_size are kept whole.
    Longer sections are split with a sliding window.
    """

    def __init__(
        self,
        chunk_size:     int                  = 500,
        chunk_overlap:  int                  = 50,
        min_chunk_size: int                  = 50,
        size_fn:        Callable[[str], int] = len,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.size_fn       = size_fn

    def chunk(self, path: str, title: str, sections: list[Section]) -> list[Chunk]:
        results = []
        idx = 0
        for sec in sections:
            text = sec.text.strip()
            if not text or self.size_fn(text) < self.min_chunk_size:
                continue
            if self.size_fn(text) <= self.chunk_size:
                results.append(Chunk(
                    article=title, path=path, section=sec.title, depth=sec.depth,
                    chunk_index=idx, text=text, char_count=len(text),
                ))
                idx += 1
            else:
                for window in _sliding_window(text, self.chunk_size, self.chunk_overlap, self.size_fn):
                    if self.size_fn(window) >= self.min_chunk_size:
                        results.append(Chunk(
                            article=title, path=path, section=sec.title, depth=sec.depth,
                            chunk_index=idx, text=window, char_count=len(window),
                        ))
                        idx += 1
        return results


class FlatChunker:
    """
    Ignores section structure entirely.
    Sliding window over the full article text as one string.
    """

    def __init__(
        self,
        chunk_size:     int                  = 500,
        chunk_overlap:  int                  = 50,
        min_chunk_size: int                  = 50,
        size_fn:        Callable[[str], int] = len,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.size_fn       = size_fn

    def chunk(self, path: str, title: str, sections: list[Section]) -> list[Chunk]:
        full_text = "\n\n".join(s.text.strip() for s in sections if s.text.strip())
        if not full_text:
            return []
        results = []
        for idx, window in enumerate(
            _sliding_window(full_text, self.chunk_size, self.chunk_overlap, self.size_fn)
        ):
            if self.size_fn(window) >= self.min_chunk_size:
                results.append(Chunk(
                    article=title, path=path, section="", depth=0,
                    chunk_index=idx, text=window, char_count=len(window),
                ))
        return results


class ParagraphChunker:
    """
    Splits on paragraph boundaries (double newline).
    Short paragraphs are merged with the previous one.
    Long paragraphs are split with a sliding window.
    """

    def __init__(
        self,
        max_chunk_size: int                  = 500,
        min_chunk_size: int                  = 50,
        size_fn:        Callable[[str], int] = len,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.size_fn        = size_fn

    def chunk(self, path: str, title: str, sections: list[Section]) -> list[Chunk]:
        results = []
        idx         = 0
        pending     = ""
        pending_sec = ("Lead", 0)

        def _emit(text: str, sec_title: str, depth: int) -> None:
            nonlocal idx
            text = text.strip()
            if text and self.size_fn(text) >= self.min_chunk_size:
                results.append(Chunk(
                    article=title, path=path, section=sec_title, depth=depth,
                    chunk_index=idx, text=text, char_count=len(text),
                ))
                idx += 1

        for sec in sections:
            paras = [p.strip() for p in sec.text.split("\n\n") if p.strip()]
            for para in paras:
                if self.size_fn(para) > self.max_chunk_size:
                    if pending:
                        _emit(pending, *pending_sec)
                        pending = ""
                    overlap = max(1, self.max_chunk_size // 10)
                    for window in _sliding_window(para, self.max_chunk_size, overlap, self.size_fn):
                        _emit(window, sec.title, sec.depth)
                elif self.size_fn(pending) + self.size_fn(para) > self.max_chunk_size:
                    if pending:
                        _emit(pending, *pending_sec)
                    pending     = para
                    pending_sec = (sec.title, sec.depth)
                else:
                    pending     = (pending + "\n\n" + para).strip() if pending else para
                    pending_sec = (sec.title, sec.depth)

        if pending:
            _emit(pending, *pending_sec)

        return results
