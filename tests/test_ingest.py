"""
Adversarial tests for learner.ingest — independent authorship.

No unit under test is mocked; real code paths are exercised end-to-end.
"""
from __future__ import annotations

import pytest

from learner.ingest import ContentChunk, ingest_file, ingest_text


# ── empty and whitespace-only inputs ─────────────────────────────────────────


class TestEmptyAndWhitespaceInputs:

    def test_empty_string_returns_empty_list(self):
        assert ingest_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert ingest_text("   \n  \t  ") == []

    def test_newlines_only_returns_empty_list(self):
        assert ingest_text("\n\n\n") == []


# ── markdown heading splitting ────────────────────────────────────────────────


class TestMarkdownHeadingSplit:

    def test_h2_headings_produce_titled_chunks(self):
        text = "## Alpha\nContent A.\n\n## Beta\nContent B."
        chunks = ingest_text(text)
        assert len(chunks) == 2
        assert chunks[0].title == "Alpha"
        assert chunks[1].title == "Beta"

    def test_all_heading_levels_match(self):
        text = "# Level1\nbody1\n\n## Level2\nbody2\n\n### Level3\nbody3"
        chunks = ingest_text(text)
        assert len(chunks) == 3
        assert chunks[0].title == "Level1"
        assert chunks[1].title == "Level2"
        assert chunks[2].title == "Level3"

    def test_chunk_body_does_not_contain_heading_line(self):
        text = "## The Title\nBody content here."
        chunks = ingest_text(text)
        assert len(chunks) == 1
        assert "## The Title" not in chunks[0].body
        assert chunks[0].body == "Body content here."

    def test_heading_title_has_no_hash_prefix(self):
        text = "## My Section\nsome body"
        chunk = ingest_text(text)[0]
        assert not chunk.title.startswith("#")
        assert chunk.title == "My Section"

    def test_last_heading_captures_trailing_text(self):
        text = "## First\ncontent A\n## Last\ncontent B\ncontent C"
        chunks = ingest_text(text)
        assert "content B" in chunks[1].body
        assert "content C" in chunks[1].body

    def test_heading_with_no_body_has_empty_body(self):
        text = "## Lonely\n\n## NextSection\nhas content"
        chunks = ingest_text(text)
        assert chunks[0].body == ""

    def test_index_sequential_starting_at_zero_for_headings(self):
        text = "## A\nbody\n## B\nbody\n## C\nbody"
        chunks = ingest_text(text)
        assert [c.index for c in chunks] == [0, 1, 2]

    def test_source_propagated_to_all_heading_chunks(self):
        text = "## X\nfoo\n## Y\nbar"
        chunks = ingest_text(text, source="my_source")
        assert all(c.source == "my_source" for c in chunks)

    def test_heading_body_stripped_of_leading_trailing_whitespace(self):
        text = "## Title\n\n   body with spaces   \n\n## Next\nbody"
        chunks = ingest_text(text)
        assert chunks[0].body == "body with spaces"

    def test_h6_heading_matches(self):
        text = "###### Deep\ndeep content"
        chunks = ingest_text(text)
        assert len(chunks) == 1
        assert chunks[0].title == "Deep"


# ── paragraph fallback splitting ──────────────────────────────────────────────


class TestParagraphFallback:

    def test_no_headings_uses_blank_line_split(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = ingest_text(text)
        assert len(chunks) == 3

    def test_paragraph_chunks_have_sequential_indexes(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = ingest_text(text)
        assert [c.index for c in chunks] == [0, 1, 2]

    def test_paragraph_source_propagated(self):
        text = "Para one.\n\nPara two."
        chunks = ingest_text(text, source="book.txt")
        assert all(c.source == "book.txt" for c in chunks)

    def test_paragraph_first_line_becomes_title(self):
        text = "Title line\nSecond line\nThird line"
        chunks = ingest_text(text)
        assert chunks[0].title == "Title line"

    def test_paragraph_remaining_lines_form_body(self):
        text = "Title line\nSecond line\nThird line"
        chunks = ingest_text(text)
        assert "Second line" in chunks[0].body
        assert "Third line" in chunks[0].body

    def test_single_paragraph_produces_one_chunk(self):
        text = "This is a single paragraph with no blank lines and no headings."
        chunks = ingest_text(text)
        assert len(chunks) == 1
        assert isinstance(chunks[0], ContentChunk)

    def test_single_line_text_produces_one_chunk(self):
        text = "Just one line of text."
        chunks = ingest_text(text)
        assert len(chunks) == 1

    def test_multiple_blank_lines_treated_as_paragraph_separator(self):
        text = "Para A.\n\n\n\nPara B."
        chunks = ingest_text(text)
        assert len(chunks) == 2

    def test_paragraph_title_truncated_at_eighty_chars(self):
        long_line = "A" * 100
        text = f"{long_line}\nrest of paragraph"
        chunks = ingest_text(text)
        assert len(chunks[0].title) == 80


# ── ContentChunk structure ────────────────────────────────────────────────────


class TestContentChunkStructure:

    def test_chunks_are_content_chunk_instances(self):
        chunks = ingest_text("## Section\ncontent")
        assert all(isinstance(c, ContentChunk) for c in chunks)

    def test_index_starts_at_zero(self):
        chunks = ingest_text("## First\nbody")
        assert chunks[0].index == 0

    def test_default_source_is_empty_string(self):
        chunks = ingest_text("## Title\nbody")
        assert chunks[0].source == ""

    def test_source_is_string_on_each_chunk(self):
        chunks = ingest_text("Para one.\n\nPara two.", source="src.md")
        for chunk in chunks:
            assert isinstance(chunk.source, str)
            assert chunk.source == "src.md"


# ── ingest_file ───────────────────────────────────────────────────────────────


class TestIngestFile:

    def test_ingest_txt_file_returns_chunks(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("## Section One\nSome content.\n\n## Section Two\nMore content.", encoding="utf-8")
        chunks = ingest_file(f)
        assert len(chunks) == 2

    def test_ingest_md_file_returns_chunks(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("## Introduction\nHello.\n\n## Summary\nBye.", encoding="utf-8")
        chunks = ingest_file(f)
        assert len(chunks) == 2

    def test_ingest_file_sets_source_to_file_path(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("Some text here.", encoding="utf-8")
        chunks = ingest_file(f)
        assert all(c.source == str(f) for c in chunks)

    def test_ingest_file_raises_value_error_for_pdf(self, tmp_path):
        f = tmp_path / "report.pdf"
        f.write_text("fake pdf content", encoding="utf-8")
        with pytest.raises(ValueError):
            ingest_file(f)

    def test_ingest_file_raises_value_error_for_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3", encoding="utf-8")
        with pytest.raises(ValueError):
            ingest_file(f)

    def test_ingest_file_raises_value_error_for_py(self, tmp_path):
        f = tmp_path / "script.py"
        f.write_text("print('hello')", encoding="utf-8")
        with pytest.raises(ValueError):
            ingest_file(f)

    def test_ingest_file_accepts_string_path(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("Hello world.", encoding="utf-8")
        chunks = ingest_file(str(f))
        assert len(chunks) >= 1

    def test_ingest_file_accepts_path_object(self, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("## Topic\nContent here.", encoding="utf-8")
        chunks = ingest_file(f)
        assert len(chunks) == 1

    def test_ingest_file_md_heading_body_correct(self, tmp_path):
        f = tmp_path / "lesson.md"
        f.write_text("## Key Concept\nThe explanation.", encoding="utf-8")
        chunks = ingest_file(f)
        assert chunks[0].title == "Key Concept"
        assert chunks[0].body == "The explanation."
        assert "## Key Concept" not in chunks[0].body
