from dataclasses import dataclass


@dataclass
class Chunk:
    article:     str
    path:        str
    section:     str
    depth:       int
    chunk_index: int
    text:        str
    char_count:  int

    def to_dict(self) -> dict:
        return {
            "article":     self.article,
            "path":        self.path,
            "section":     self.section,
            "depth":       self.depth,
            "chunk_index": self.chunk_index,
            "text":        self.text,
            "char_count":  self.char_count,
        }


@dataclass
class SkippedItem:
    type:   str   # "article"
    path:   str
    title:  str
    reason: str   # "redirect" | "namespace" | "empty_after_cleaning" | "error:..."

    def to_dict(self) -> dict:
        return {
            "type":   self.type,
            "path":   self.path,
            "title":  self.title,
            "reason": self.reason,
        }


@dataclass
class Section:
    title: str
    depth: int
    text:  str
