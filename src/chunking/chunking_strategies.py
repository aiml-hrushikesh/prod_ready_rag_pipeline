from abc import abstractmethod
from typing import List

from src.config import settings

from .base import BaseChunker


class SimpleChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        overlap: int = settings.DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """Simple sliding window character-based chunking."""
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= text_len:
                break
            start += self.chunk_size - self.overlap
            if self.chunk_size <= self.overlap:
                start += 1
        return chunks


class AdvancedChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        overlap: int = settings.DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """Advanced chunking using recursive split/merge by separators (paragraphs, sentences, words)."""
        if not text:
            return []

        separators = ["\n\n", "\n", ". ", " ", ""]

        def split_text(text_to_split: str, separators_list: List[str]) -> List[str]:
            if not separators_list:
                return [text_to_split]
            separator = separators_list[0]
            if len(text_to_split) <= self.chunk_size:
                return [text_to_split]

            if separator == "":
                return [
                    text_to_split[i : i + self.chunk_size]
                    for i in range(0, len(text_to_split), self.chunk_size)
                ]

            splits = text_to_split.split(separator)
            final_splits = []
            for i, split in enumerate(splits):
                if i < len(splits) - 1:
                    final_splits.append(split + separator)
                else:
                    final_splits.append(split)

            output = []
            for split in final_splits:
                if len(split) > self.chunk_size:
                    output.extend(split_text(split, separators_list[1:]))
                else:
                    output.append(split)
            return output

        raw_splits = split_text(text, separators)

        chunks = []
        current_chunk_parts: List[str] = []
        current_length = 0

        for split in raw_splits:
            if current_length + len(split) <= self.chunk_size:
                current_chunk_parts.append(split)
                current_length += len(split)
            else:
                if current_chunk_parts:
                    chunks.append("".join(current_chunk_parts))

                overlap_parts = []
                overlap_len = 0
                for part in reversed(current_chunk_parts):
                    if overlap_len + len(part) <= self.overlap:
                        overlap_parts.insert(0, part)
                        overlap_len += len(part)
                    else:
                        break
                current_chunk_parts = overlap_parts + [split]
                current_length = sum(len(p) for p in current_chunk_parts)

        if current_chunk_parts:
            chunks.append("".join(current_chunk_parts))

        return chunks

