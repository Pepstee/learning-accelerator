from __future__ import annotations

import argparse
import json
import pathlib

import pytest

import learner.cli as cli_module
from learner.cli import build_parser, main
from learner.llm import MockBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_bundle(data_dir: pathlib.Path, topic: str, bundle: dict) -> pathlib.Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{topic}.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return path


def _minimal_bundle(n_cards: int = 2, n_questions: int = 1) -> dict:
    return {
        "summary": "Test summary for the topic.",
        "cards": [
            {"front": f"Front {i}", "back": f"Back {i}"}
            for i in range(n_cards)
        ],
        "questions": [
            {
                "stem": f"Question {i}?",
                "choices": ["A", "B", "C", "D"],
                "answer_index": 0,
                "explanation": "Because A.",
            }
            for i in range(n_questions)
        ],
    }


def _write_session(data_dir: pathlib.Path, topic: str, quality: int = 5) -> None:
    sessions_dir = data_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "session_id": "test-session-001",
        "started_at": "2024-01-01T10:00:00",
        "duration_seconds": 60.0,
        "ratings": [
            {"card_front": "Front 0", "topic": topic, "quality": quality}
        ],
    }
    (sessions_dir / "20240101T100000.json").write_text(
        json.dumps(record), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

def test_build_parser_returns_argument_parser():
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_build_parser_has_all_subcommands():
    parser = build_parser()
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    choices = set(subparsers_action.choices.keys())
    expected = {"ingest", "summary", "flashcards", "practice", "exam", "study-plan", "analytics"}
    assert expected.issubset(choices)


def test_build_parser_mock_flag_parsed():
    parser = build_parser()
    args = parser.parse_args(["--mock", "ingest", "file.txt"])
    assert args.mock is True


def test_build_parser_mock_flag_default_false():
    parser = build_parser()
    args = parser.parse_args(["ingest", "file.txt"])
    assert args.mock is False


def test_build_parser_data_dir_default():
    parser = build_parser()
    args = parser.parse_args(["ingest", "file.txt"])
    assert args.data_dir == pathlib.Path.home() / ".learner"


def test_build_parser_data_dir_override(tmp_path):
    parser = build_parser()
    args = parser.parse_args(["--data-dir", str(tmp_path), "ingest", "file.txt"])
    assert args.data_dir == tmp_path


# ---------------------------------------------------------------------------
# main with no subcommand
# ---------------------------------------------------------------------------

def test_main_no_args_exits_zero():
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 0


def test_main_no_args_prints_help(capsys):
    with pytest.raises(SystemExit):
        main([])
    out = capsys.readouterr().out
    # argparse help always contains the prog name and "usage"
    assert "learner" in out or "usage" in out.lower()


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

def test_ingest_writes_json_bundle(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("Spaced repetition is a learning technique.", encoding="utf-8")
    main(["--mock", "--data-dir", str(tmp_path), "ingest", str(source)])
    bundle_path = tmp_path / "notes.json"
    assert bundle_path.exists()
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert "summary" in data
    assert "cards" in data
    assert "questions" in data


def test_ingest_prints_summary_line(tmp_path, capsys):
    source = tmp_path / "topic.txt"
    source.write_text("Python is a programming language.", encoding="utf-8")
    main(["--mock", "--data-dir", str(tmp_path), "ingest", str(source)])
    out = capsys.readouterr().out
    assert "Summary:" in out


def test_ingest_prints_cards_line(tmp_path, capsys):
    source = tmp_path / "topic.txt"
    source.write_text("Python is a programming language.", encoding="utf-8")
    main(["--mock", "--data-dir", str(tmp_path), "ingest", str(source)])
    out = capsys.readouterr().out
    assert "Cards:" in out


def test_ingest_prints_questions_line(tmp_path, capsys):
    source = tmp_path / "topic.txt"
    source.write_text("Python is a programming language.", encoding="utf-8")
    main(["--mock", "--data-dir", str(tmp_path), "ingest", str(source)])
    out = capsys.readouterr().out
    assert "Questions:" in out


def test_ingest_saves_bundle_using_source_stem(tmp_path):
    source = tmp_path / "my_notes.txt"
    source.write_text("Some content here.", encoding="utf-8")
    main(["--mock", "--data-dir", str(tmp_path), "ingest", str(source)])
    assert (tmp_path / "my_notes.json").exists()


def test_ingest_missing_file_exits_1(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        main(["--mock", "--data-dir", str(tmp_path), "ingest", str(tmp_path / "ghost.txt")])
    assert exc_info.value.code == 1


def test_ingest_missing_file_reports_error_on_stderr(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--mock", "--data-dir", str(tmp_path), "ingest", str(tmp_path / "ghost.txt")])
    err = capsys.readouterr().err
    assert "error" in err.lower()


def test_ingest_creates_data_dir_if_absent(tmp_path):
    new_dir = tmp_path / "deep" / "nested"
    source = tmp_path / "file.txt"
    source.write_text("content", encoding="utf-8")
    main(["--mock", "--data-dir", str(new_dir), "ingest", str(source)])
    assert new_dir.exists()
    assert (new_dir / "file.json").exists()


class _RecordingBackend:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_ingest_uses_one_strict_backend_generation(tmp_path, monkeypatch):
    source = tmp_path / "atomic.txt"
    source.write_text("One source must produce one coherent bundle.", encoding="utf-8")
    expected = _minimal_bundle(n_cards=2, n_questions=1)
    backend = _RecordingBackend(json.dumps(expected))
    monkeypatch.setattr(cli_module, "_backend", lambda args: backend)

    main(["--data-dir", str(tmp_path), "ingest", str(source)])

    assert len(backend.prompts) == 1
    assert "One source must produce one coherent bundle." in backend.prompts[0]
    assert json.loads((tmp_path / "atomic.json").read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        json.dumps({"summary": "incomplete", "cards": []}),
    ],
)
def test_ingest_refuses_invalid_bundle_without_partial_write(
    tmp_path, monkeypatch, capsys, response
):
    source = tmp_path / "invalid.txt"
    source.write_text("Invalid output must not become a bundle.", encoding="utf-8")
    backend = _RecordingBackend(response)
    monkeypatch.setattr(cli_module, "_backend", lambda args: backend)

    with pytest.raises(SystemExit) as exc_info:
        main(["--data-dir", str(tmp_path), "ingest", str(source)])

    assert exc_info.value.code == 1
    assert len(backend.prompts) == 1
    assert not (tmp_path / "invalid.json").exists()
    assert "processing failed" in capsys.readouterr().err


def test_ingest_mock_path_calls_backend_once(tmp_path, monkeypatch):
    source = tmp_path / "offline.txt"
    source.write_text("Offline ingestion remains deterministic.", encoding="utf-8")
    calls: list[str] = []
    original = MockBackend.complete

    def recording_complete(self, prompt: str) -> str:
        calls.append(prompt)
        return original(self, prompt)

    monkeypatch.setattr(MockBackend, "complete", recording_complete)

    main(["--mock", "--data-dir", str(tmp_path), "ingest", str(source)])

    assert len(calls) == 1
    data = json.loads((tmp_path / "offline.json").read_text(encoding="utf-8"))
    assert data == json.loads(MockBackend._CONTENT_JSON)


def test_generate_routes_markdown_sections_as_distinct_chunks(tmp_path, monkeypatch):
    source = tmp_path / "sections.md"
    source.write_text("# Alpha\nA body.\n\n## Beta\nB body.", encoding="utf-8")
    backend = object()
    calls: list[tuple[str, list[str], object]] = []

    monkeypatch.setattr(cli_module, "_backend", lambda args: backend)
    monkeypatch.setattr(
        cli_module,
        "generate_summary",
        lambda chunks, actual: calls.append(("summary", chunks, actual)) or "summary",
    )
    monkeypatch.setattr(
        cli_module,
        "generate_flashcards",
        lambda chunks, actual: calls.append(("cards", chunks, actual)) or [],
    )
    monkeypatch.setattr(
        cli_module,
        "generate_questions",
        lambda chunks, actual: calls.append(("questions", chunks, actual)) or [],
    )

    main(["--data-dir", str(tmp_path), "generate", str(source)])

    expected = ["Alpha\nA body.", "Beta\nB body."]
    assert calls == [
        ("summary", expected, backend),
        ("cards", expected, backend),
        ("questions", expected, backend),
    ]


def test_generate_preserves_preamble_and_fenced_literal_heading(tmp_path, monkeypatch):
    source = tmp_path / "literals.md"
    source.write_text(
        "Before.\n\n```md\n# Literal example\n```\n\n# Section\nSame\nSame",
        encoding="utf-8",
    )
    seen: list[list[str]] = []
    monkeypatch.setattr(cli_module, "_backend", lambda args: object())
    monkeypatch.setattr(
        cli_module,
        "generate_summary",
        lambda chunks, backend: seen.append(chunks) or "summary",
    )
    monkeypatch.setattr(cli_module, "generate_flashcards", lambda chunks, backend: [])
    monkeypatch.setattr(cli_module, "generate_questions", lambda chunks, backend: [])

    main(["--data-dir", str(tmp_path), "generate", str(source)])

    assert seen == [
        ["Before.\n\n```md\n# Literal example\n```", "Section\nSame\nSame"]
    ]


def test_generate_routes_paragraphs_without_duplicating_single_line(tmp_path, monkeypatch):
    source = tmp_path / "paragraphs.txt"
    source.write_text("First line\ncontinues.\n\nSecond paragraph.", encoding="utf-8")
    seen: list[list[str]] = []
    monkeypatch.setattr(cli_module, "_backend", lambda args: object())
    monkeypatch.setattr(
        cli_module,
        "generate_summary",
        lambda chunks, backend: seen.append(chunks) or "summary",
    )
    monkeypatch.setattr(cli_module, "generate_flashcards", lambda chunks, backend: [])
    monkeypatch.setattr(cli_module, "generate_questions", lambda chunks, backend: [])

    main(["--data-dir", str(tmp_path), "generate", str(source)])

    assert seen == [["First line\ncontinues.", "Second paragraph."]]


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_generate_preserves_empty_source_as_one_exact_chunk(
    tmp_path, monkeypatch, text
):
    source = tmp_path / "empty.txt"
    source.write_text(text, encoding="utf-8")
    seen: list[list[str]] = []
    monkeypatch.setattr(cli_module, "_backend", lambda args: object())
    monkeypatch.setattr(
        cli_module,
        "generate_summary",
        lambda chunks, backend: seen.append(chunks) or "summary",
    )
    monkeypatch.setattr(cli_module, "generate_flashcards", lambda chunks, backend: [])
    monkeypatch.setattr(cli_module, "generate_questions", lambda chunks, backend: [])

    main(["--data-dir", str(tmp_path), "generate", str(source)])

    assert seen == [[text]]


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_summary_prints_bundle_summary(tmp_path, capsys):
    _write_bundle(tmp_path, "mytopic", _minimal_bundle())
    main(["--mock", "--data-dir", str(tmp_path), "summary", "mytopic"])
    out = capsys.readouterr().out
    assert "Test summary for the topic." in out


def test_summary_missing_topic_exits_1(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        main(["--mock", "--data-dir", str(tmp_path), "summary", "nonexistent"])
    assert exc_info.value.code == 1


def test_summary_missing_topic_prints_error_to_stderr(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--mock", "--data-dir", str(tmp_path), "summary", "nonexistent"])
    err = capsys.readouterr().err
    assert "nonexistent" in err


def test_summary_corrupt_bundle_exits_1(tmp_path):
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{ not valid json !!!", encoding="utf-8")
    with pytest.raises(SystemExit) as exc_info:
        main(["--mock", "--data-dir", str(tmp_path), "summary", "corrupt"])
    assert exc_info.value.code == 1


def test_summary_missing_key_falls_back_gracefully(tmp_path, capsys):
    # Bundle without 'summary' key should not crash — get() returns fallback
    bundle = {"cards": [], "questions": []}
    _write_bundle(tmp_path, "partial", bundle)
    main(["--mock", "--data-dir", str(tmp_path), "summary", "partial"])
    out = capsys.readouterr().out
    assert "(no summary available)" in out


# ---------------------------------------------------------------------------
# flashcards
# ---------------------------------------------------------------------------

def test_flashcards_lists_all_cards(tmp_path, capsys):
    _write_bundle(tmp_path, "topic", _minimal_bundle(n_cards=3))
    main(["--mock", "--data-dir", str(tmp_path), "flashcards", "topic"])
    out = capsys.readouterr().out
    for i in range(3):
        assert f"Front {i}" in out
        assert f"Back {i}" in out


def test_flashcards_numbers_cards_from_one(tmp_path, capsys):
    _write_bundle(tmp_path, "topic", _minimal_bundle(n_cards=2))
    main(["--mock", "--data-dir", str(tmp_path), "flashcards", "topic"])
    out = capsys.readouterr().out
    assert "[1]" in out
    assert "[2]" in out


def test_flashcards_empty_cards_prints_message(tmp_path, capsys):
    bundle = {"summary": "empty", "cards": [], "questions": []}
    _write_bundle(tmp_path, "emptytopic", bundle)
    main(["--mock", "--data-dir", str(tmp_path), "flashcards", "emptytopic"])
    out = capsys.readouterr().out
    assert "No flashcards" in out


def test_flashcards_missing_topic_exits_1(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        main(["--mock", "--data-dir", str(tmp_path), "flashcards", "ghost"])
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# analytics
# ---------------------------------------------------------------------------

def test_analytics_no_history_prints_no_session_message(tmp_path, capsys):
    main(["--mock", "--data-dir", str(tmp_path), "analytics"])
    out = capsys.readouterr().out
    assert "No session history" in out


def test_analytics_with_history_shows_session_count(tmp_path, capsys):
    _write_session(tmp_path, "python", quality=4)
    main(["--mock", "--data-dir", str(tmp_path), "analytics"])
    out = capsys.readouterr().out
    assert "Sessions:     1" in out


def test_analytics_counts_every_persisted_session(tmp_path, capsys):
    _write_session(tmp_path, "python", quality=4)
    session_path = tmp_path / "sessions" / "20240101T100000.json"
    session_path.rename(session_path.with_name("20240101T100000-python.json"))
    _write_session(tmp_path, "algebra", quality=2)
    main(["--mock", "--data-dir", str(tmp_path), "analytics"])
    out = capsys.readouterr().out
    assert "Sessions:     2" in out
    assert "Total reviews: 2" in out


def test_analytics_with_history_shows_total_reviews(tmp_path, capsys):
    _write_session(tmp_path, "python", quality=4)
    main(["--mock", "--data-dir", str(tmp_path), "analytics"])
    out = capsys.readouterr().out
    assert "Total reviews" in out


def test_analytics_with_high_error_rate_shows_weak_area(tmp_path, capsys):
    # quality=0 → error_rate=1.0 (100%) → listed as a weak area
    _write_session(tmp_path, "calculus", quality=0)
    main(["--mock", "--data-dir", str(tmp_path), "analytics"])
    out = capsys.readouterr().out
    assert "calculus" in out


def test_analytics_with_perfect_scores_shows_no_weak_areas(tmp_path, capsys):
    # quality=5 → error_rate=0.0; weak_areas still computed but error rate is 0%
    _write_session(tmp_path, "algebra", quality=5)
    main(["--mock", "--data-dir", str(tmp_path), "analytics"])
    out = capsys.readouterr().out
    # Should show session info without crashing
    assert "Sessions:" in out


# ---------------------------------------------------------------------------
# study-plan
# ---------------------------------------------------------------------------

def test_study_plan_no_history_prints_no_weak_areas(tmp_path, capsys):
    main(["--mock", "--data-dir", str(tmp_path), "study-plan"])
    out = capsys.readouterr().out
    assert "No weak areas" in out


def test_study_plan_with_weak_areas_produces_plan(tmp_path, capsys):
    # quality=0 → high error rate → study plan should be generated via mock backend
    _write_session(tmp_path, "math", quality=0)
    main(["--mock", "--data-dir", str(tmp_path), "study-plan"])
    out = capsys.readouterr().out
    assert len(out.strip()) > 0


def test_study_plan_with_weak_areas_does_not_exit_nonzero(tmp_path):
    _write_session(tmp_path, "physics", quality=0)
    # Should complete without raising SystemExit
    main(["--mock", "--data-dir", str(tmp_path), "study-plan"])
