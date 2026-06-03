from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ContentChunk:
    title: str
    body: str
    source: str = ""
    index: int = 0
    metadata: dict = field(default_factory=dict)


def ingest_text(text: str, source: str = "") -> list[ContentChunk]:
    """Split text into titled chunks by markdown headings or blank-line paragraphs."""
    chunks: list[ContentChunk] = []

    # Try splitting on markdown headings (# / ## / ###)
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(text))

    if matches:
        for i, match in enumerate(matches):
            title = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            chunks.append(ContentChunk(title=title, body=body, source=source, index=i))
    else:
        # Fall back: split on double newlines (paragraphs)
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        for i, para in enumerate(paragraphs):
            lines = para.splitlines()
            title = lines[0][:80]
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else para
            chunks.append(ContentChunk(title=title, body=body, source=source, index=i))

    # If nothing produced (e.g. single-line text), wrap it
    if not chunks and text.strip():
        chunks.append(ContentChunk(title=text.strip()[:80], body=text.strip(), source=source, index=0))

    return chunks


def ingest_file(path: str | Path) -> list[ContentChunk]:
    """Read a .txt or .md file and return its chunks."""
    p = Path(path)
    if p.suffix.lower() not in {".txt", ".md", ""}:
        raise ValueError(f"Unsupported file type: {p.suffix!r} (expected .txt or .md)")
    text = p.read_text(encoding="utf-8")
    return ingest_text(text, source=str(p))
