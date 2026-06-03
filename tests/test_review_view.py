"""
Adversarial tests for learner.review_view — independent authorship.

Strategy: exercise _load_questions, _save_session_record, _display_question,
_prompt_answer, and ReviewView.run() against real filesystem paths (tmp_path)
and captured I/O (capsys / monkeypatch).  No LLM subprocess calls; the unit
under test is never mocked; stdin is only patched via monkeypatch.
"""
from __future__ import annotations

import datetime
import json
import pathlib
import uuid

import pytest

from learner.models import Card, CardRating, Question, SessionRecord
from learner.session import ReviewSession
from learner.review_view import (
    ReviewView,
    _display_question,
    _load_questions,
    _prompt_answer,
    _save_session_record,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _session(tmp_path: pathlib.Path) -> ReviewSession:
    return ReviewSession(tmp_path)


def _write_bundle(
    data_dir: pathlib.Path,
    topic: str,
    questions: list[dict],
    cards: list[dict] | None = None,
) -> None:
    bundle = {"summary": "test", "cards": cards or [], "questions": questions}
    (data_dir / f"{topic}.json").write_text(json.dumps(bundle), encoding="utf-8")


def _q(
    stem: str = "What is 2+2?",
    choices: list[str] | None = None,
    answer_index: int = 0,
    explanation: str = "Because math.",
) -> Question:
    return Question(
        stem=stem,
        choices=choices or ["4", "3", "5"],
        answer_index=answer_index,
        explanation=explanation,
    )


def _record(
    started_at: datetime.datetime | None = None,
    ratings: list[CardRating] | None = None,
    duration_seconds: float = 0.0,
) -> SessionRecord:
    return SessionRecord(
        session_id="test-session-id",
        started_at=started_at or datetime.datetime(2024, 6, 1, 10, 0, 0),
        ratings=ratings or [],
        duration_seconds=duration_seconds,
    )


# ── _load_questions ───────────────────────────────────────────────────────────

class TestLoadQuestions:
    def test_empty_data_dir_returns_empty_list(self, tmp_path: pathlib.Path):
        result = _load_questions(_session(tmp_path))
        assert result == []

    def test_skips_corrupt_json_bundle(self, tmp_path: pathlib.Path):
        (tmp_path / "bad.json").write_text("{not valid json}", encoding="utf-8")
        result = _load_questions(_session(tmp_path))
        assert result == []

    def test_corrupt_skipped_valid_loaded(self, tmp_path: pathlib.Path):
        (tmp_path / "bad.json").write_text("{not valid json}", encoding="utf-8")
        _write_bundle(tmp_path, "good", questions=[
            {"stem": "Q?", "choices": ["A", "B"], "answer_index": 0, "explanation": "A"}
        ])
        result = _load_questions(_session(tmp_path))
        assert len(result) == 1
        assert result[0][0].stem == "Q?"

    def test_returns_question_objects_for_valid_bundle(self, tmp_path: pathlib.Path):
        _write_bundle(tmp_path, "math", questions=[
            {"stem": "2+2?", "choices": ["3", "4", "5"], "answer_index": 1, "explanation": "4"}
        ])
        result = _load_questions(_session(tmp_path))
        assert len(result) == 1
        q, topic = result[0]
        assert isinstance(q, Question)
        assert q.stem == "2+2?"
        assert q.choices == ["3", "4", "5"]
        assert q.answer_index == 1
        assert q.explanation == "4"

    def test_topic_name_taken_from_bundle_stem(self, tmp_path: pathlib.Path):
        _write_bundle(tmp_path, "organic_chemistry", questions=[
            {"stem": "What is benzene?", "choices": ["C6H6", "H2O"], "answer_index": 0, "explanation": "C6H6"}
        ])
        result = _load_questions(_session(tmp_path))
        assert result[0][1] == "organic_chemistry"

    def test_question_entries_missing_keys_silently_skipped(self, tmp_path: pathlib.Path):
        bundle = {
            "questions": [
                {"stem": "Good Q?", "choices": ["A", "B"], "answer_index": 0, "explanation": "A"},
                {"stem": "Bad Q?"},  # missing 'choices', 'answer_index', 'explanation'
            ]
        }
        (tmp_path / "mixed.json").write_text(json.dumps(bundle), encoding="utf-8")
        result = _load_questions(_session(tmp_path))
        assert len(result) == 1
        assert result[0][0].stem == "Good Q?"

    def test_bundle_without_questions_key_contributes_nothing(self, tmp_path: pathlib.Path):
        (tmp_path / "nocards.json").write_text(json.dumps({"summary": "x", "cards": []}), encoding="utf-8")
        result = _load_questions(_session(tmp_path))
        assert result == []

    def test_multiple_bundles_all_questions_loaded(self, tmp_path: pathlib.Path):
        _write_bundle(tmp_path, "math", questions=[
            {"stem": "1+1?", "choices": ["1", "2"], "answer_index": 1, "explanation": "2"}
        ])
        _write_bundle(tmp_path, "science", questions=[
            {"stem": "H2O?", "choices": ["water", "gas"], "answer_index": 0, "explanation": "water"}
        ])
        result = _load_questions(_session(tmp_path))
        stems = {q.stem for q, _ in result}
        assert stems == {"1+1?", "H2O?"}

    def test_empty_questions_list_in_bundle_returns_empty(self, tmp_path: pathlib.Path):
        _write_bundle(tmp_path, "empty_qs", questions=[])
        result = _load_questions(_session(tmp_path))
        assert result == []


# ── _save_session_record ──────────────────────────────────────────────────────

class TestSaveSessionRecord:
    def test_writes_parseable_json_under_sessions(self, tmp_path: pathlib.Path):
        sess = _session(tmp_path)
        _save_session_record(sess, _record())
        files = list((tmp_path / "sessions").glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["session_id"] == "test-session-id"

    def test_creates_sessions_directory_if_absent(self, tmp_path: pathlib.Path):
        assert not (tmp_path / "sessions").exists()
        _save_session_record(_session(tmp_path), _record())
        assert (tmp_path / "sessions").is_dir()

    def test_filename_encodes_started_at_timestamp(self, tmp_path: pathlib.Path):
        sess = _session(tmp_path)
        _save_session_record(sess, _record(started_at=datetime.datetime(2024, 6, 1, 10, 0, 0)))
        files = list((tmp_path / "sessions").glob("*.json"))
        assert files[0].name == "20240601T100000.json"

    def test_ratings_serialised_in_json(self, tmp_path: pathlib.Path):
        ratings = [CardRating(card_front="Q?", topic="math", quality=4)]
        rec = SessionRecord(
            session_id="sid",
            started_at=datetime.datetime(2024, 1, 1),
            ratings=ratings,
            duration_seconds=30.0,
        )
        _save_session_record(_session(tmp_path), rec)
        files = list((tmp_path / "sessions").glob("*.json"))
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["ratings"][0]["card_front"] == "Q?"
        assert data["ratings"][0]["quality"] == 4
        assert data["ratings"][0]["topic"] == "math"

    def test_duration_seconds_round_trips(self, tmp_path: pathlib.Path):
        rec = SessionRecord(
            session_id="s",
            started_at=datetime.datetime(2024, 1, 1),
            ratings=[],
            duration_seconds=123.45,
        )
        _save_session_record(_session(tmp_path), rec)
        files = list((tmp_path / "sessions").glob("*.json"))
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["duration_seconds"] == pytest.approx(123.45)

    def test_started_at_written_as_iso_string(self, tmp_path: pathlib.Path):
        dt = datetime.datetime(2025, 3, 15, 8, 30, 0)
        rec = SessionRecord(session_id="s", started_at=dt, ratings=[], duration_seconds=0.0)
        _save_session_record(_session(tmp_path), rec)
        files = list((tmp_path / "sessions").glob("*.json"))
        data = json.loads(files[0].read_text(encoding="utf-8"))
        # Must round-trip through fromisoformat
        assert datetime.datetime.fromisoformat(data["started_at"]) == dt


# ── _display_question ─────────────────────────────────────────────────────────

class TestDisplayQuestion:
    def test_stem_appears_in_stdout(self, capsys):
        q = _q(stem="What is the speed of light?", choices=["3e8 m/s", "1e5 m/s"])
        _display_question(1, 5, q)
        assert "What is the speed of light?" in capsys.readouterr().out

    def test_all_choices_appear_in_stdout(self, capsys):
        q = _q(choices=["Paris", "London", "Berlin"])
        _display_question(1, 3, q)
        out = capsys.readouterr().out
        assert "Paris" in out
        assert "London" in out
        assert "Berlin" in out

    def test_choices_labeled_alphabetically(self, capsys):
        q = _q(choices=["Alpha", "Beta", "Gamma"])
        _display_question(2, 10, q)
        out = capsys.readouterr().out
        assert "A) Alpha" in out
        assert "B) Beta" in out
        assert "C) Gamma" in out

    def test_idx_and_total_appear_in_output(self, capsys):
        _display_question(3, 7, _q())
        assert "[3/7]" in capsys.readouterr().out

    def test_fourth_choice_labeled_D(self, capsys):
        q = _q(choices=["W", "X", "Y", "Z"])
        _display_question(1, 1, q)
        out = capsys.readouterr().out
        assert "D) Z" in out


# ── _prompt_answer ────────────────────────────────────────────────────────────

class TestPromptAnswer:
    def test_correct_letter_returns_true(self, monkeypatch):
        q = _q(choices=["Yes", "No", "Maybe"], answer_index=0)
        monkeypatch.setattr("builtins.input", lambda _: "A")
        assert _prompt_answer(q) is True

    def test_wrong_letter_returns_false(self, monkeypatch):
        q = _q(choices=["Yes", "No"], answer_index=1)
        # answer_index=1 → 'B' is correct; user presses 'A'
        monkeypatch.setattr("builtins.input", lambda _: "A")
        assert _prompt_answer(q) is False

    def test_lowercase_input_accepted(self, monkeypatch):
        q = _q(choices=["Yes", "No"], answer_index=0)
        monkeypatch.setattr("builtins.input", lambda _: "a")
        assert _prompt_answer(q) is True

    def test_invalid_then_valid_loops_and_succeeds(self, monkeypatch, capsys):
        q = _q(choices=["Alpha", "Beta"], answer_index=0)
        answers = iter(["Z", "9", "A"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))
        result = _prompt_answer(q)
        assert result is True
        assert "Enter one of" in capsys.readouterr().out

    def test_last_choice_index_correct(self, monkeypatch):
        q = _q(choices=["W", "X", "Y", "Z"], answer_index=3)
        monkeypatch.setattr("builtins.input", lambda _: "D")
        assert _prompt_answer(q) is True

    def test_correct_middle_choice(self, monkeypatch):
        q = _q(choices=["A opt", "B opt", "C opt"], answer_index=1)
        monkeypatch.setattr("builtins.input", lambda _: "B")
        assert _prompt_answer(q) is True


# ── ReviewView.run() ──────────────────────────────────────────────────────────

class TestReviewViewRunEmpty:
    def test_returns_session_record_when_no_cards_no_questions(self, tmp_path: pathlib.Path):
        record = ReviewView().run(_session(tmp_path))
        assert isinstance(record, SessionRecord)

    def test_returns_empty_ratings_when_nothing_due(self, tmp_path: pathlib.Path):
        record = ReviewView().run(_session(tmp_path))
        assert record.ratings == []

    def test_duration_is_zero_when_nothing_due(self, tmp_path: pathlib.Path):
        record = ReviewView().run(_session(tmp_path))
        assert record.duration_seconds == pytest.approx(0.0)

    def test_session_id_is_valid_uuid(self, tmp_path: pathlib.Path):
        record = ReviewView().run(_session(tmp_path))
        uuid.UUID(record.session_id)  # raises ValueError if malformed

    def test_prints_nothing_due_message(self, tmp_path: pathlib.Path, capsys):
        ReviewView().run(_session(tmp_path))
        assert "Nothing due for review." in capsys.readouterr().out

    def test_no_session_file_written_when_nothing_due(self, tmp_path: pathlib.Path):
        ReviewView().run(_session(tmp_path))
        sessions_dir = tmp_path / "sessions"
        files = list(sessions_dir.glob("*.json")) if sessions_dir.exists() else []
        assert files == []


class TestReviewViewRunQuestionsOnly:
    def _run_with_answer(self, tmp_path: pathlib.Path, monkeypatch, answer: str) -> SessionRecord:
        _write_bundle(tmp_path, "math", questions=[
            {"stem": "2+2?", "choices": ["3", "4", "5"], "answer_index": 1, "explanation": "4"}
        ])
        # answer_index=1 → 'B' is correct
        monkeypatch.setattr("builtins.input", lambda _: answer)
        return ReviewView().run(_session(tmp_path))

    def test_creates_session_record(self, tmp_path: pathlib.Path, monkeypatch):
        record = self._run_with_answer(tmp_path, monkeypatch, "B")
        assert isinstance(record, SessionRecord)

    def test_writes_session_file_under_sessions_dir(self, tmp_path: pathlib.Path, monkeypatch):
        self._run_with_answer(tmp_path, monkeypatch, "B")
        files = list((tmp_path / "sessions").glob("*.json"))
        assert len(files) == 1

    def test_correct_answer_quality_5(self, tmp_path: pathlib.Path, monkeypatch):
        record = self._run_with_answer(tmp_path, monkeypatch, "B")
        assert len(record.ratings) == 1
        assert record.ratings[0].quality == 5

    def test_wrong_answer_quality_2(self, tmp_path: pathlib.Path, monkeypatch):
        record = self._run_with_answer(tmp_path, monkeypatch, "A")
        assert record.ratings[0].quality == 2

    def test_rating_card_front_is_question_stem(self, tmp_path: pathlib.Path, monkeypatch):
        record = self._run_with_answer(tmp_path, monkeypatch, "B")
        assert record.ratings[0].card_front == "2+2?"

    def test_rating_topic_matches_bundle_stem(self, tmp_path: pathlib.Path, monkeypatch):
        record = self._run_with_answer(tmp_path, monkeypatch, "B")
        assert record.ratings[0].topic == "math"

    def test_session_file_is_parseable_json(self, tmp_path: pathlib.Path, monkeypatch):
        self._run_with_answer(tmp_path, monkeypatch, "B")
        files = list((tmp_path / "sessions").glob("*.json"))
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert "session_id" in data
        assert "ratings" in data
        assert "duration_seconds" in data

    def test_multiple_questions_all_rated(self, tmp_path: pathlib.Path, monkeypatch):
        _write_bundle(tmp_path, "mixed", questions=[
            {"stem": "Q1?", "choices": ["A", "B"], "answer_index": 0, "explanation": "A"},
            {"stem": "Q2?", "choices": ["X", "Y"], "answer_index": 1, "explanation": "Y"},
        ])
        answers = iter(["A", "B"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))
        record = ReviewView().run(_session(tmp_path))
        assert len(record.ratings) == 2

    def test_no_due_cards_means_no_srs_state_saved(self, tmp_path: pathlib.Path, monkeypatch):
        _write_bundle(tmp_path, "math", questions=[
            {"stem": "2+2?", "choices": ["3", "4"], "answer_index": 1, "explanation": "4"}
        ])
        monkeypatch.setattr("builtins.input", lambda _: "B")
        ReviewView().run(_session(tmp_path))
        assert not (tmp_path / "srs").exists()
