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


def _paragraph_chunk(text: str, source: str, index: int, kind: str) -> ContentChunk:
    lines = text.splitlines()
    title = lines[0][:80]
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else text
    return ContentChunk(
        title=title,
        body=body,
        source=source,
        index=index,
        metadata={"kind": kind, "generation_text": text},
    )


def _markdown_headings(text: str) -> list[tuple[int, int, str]]:
    headings: list[tuple[int, int, str]] = []
    fence_character = ""
    fence_length = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped_line = line.rstrip("\r\n")
        if fence_character:
            closing_fence = re.match(
                rf"^[ \t]{{0,3}}({re.escape(fence_character)}{{{fence_length},}})[ \t]*$",
                stripped_line,
            )
            if closing_fence:
                fence_character = ""
                fence_length = 0
            offset += len(line)
            continue
        opening_fence = re.match(r"^[ \t]{0,3}(`{3,}|~{3,})", stripped_line)
        if opening_fence:
            marker = opening_fence.group(1)
            if not fence_character:
                fence_character = marker[0]
                fence_length = len(marker)
            offset += len(line)
            continue
        heading = re.match(r"^(#{1,6})[ \t]+(.+)$", stripped_line)
        if heading:
            headings.append((offset, offset + len(line), heading.group(2).strip()))
        offset += len(line)
    return headings


def ingest_text(text: str, source: str = "") -> list[ContentChunk]:
    """Split text into titled chunks by markdown headings or blank-line paragraphs."""
    chunks: list[ContentChunk] = []

    matches = _markdown_headings(text)

    if matches:
        preamble = text[: matches[0][0]].strip()
        if preamble:
            chunks.append(_paragraph_chunk(preamble, source, 0, "preamble"))
        for i, (_, heading_end, title) in enumerate(matches):
            start = heading_end
            end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            generation_text = "\n".join(part for part in (title, body) if part)
            chunks.append(
                ContentChunk(
                    title=title,
                    body=body,
                    source=source,
                    index=len(chunks),
                    metadata={"kind": "heading", "generation_text": generation_text},
                )
            )
    else:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        chunks.extend(
            _paragraph_chunk(paragraph, source, index, "paragraph")
            for index, paragraph in enumerate(paragraphs)
        )

    if not chunks and text.strip():
        chunks.append(_paragraph_chunk(text.strip(), source, 0, "paragraph"))

    return chunks


def ingest_file(path: str | Path) -> list[ContentChunk]:
    """Read a .txt or .md file and return its chunks."""
    p = Path(path)
    if p.suffix.lower() not in {".txt", ".md", ""}:
        raise ValueError(f"Unsupported file type: {p.suffix!r} (expected .txt or .md)")
    text = p.read_text(encoding="utf-8")
    return ingest_text(text, source=str(p))
