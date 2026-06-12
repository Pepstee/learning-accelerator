"""
Integration tests: full ingest→generate→analytics→CLI→review_view round-trip.

Covers:
- ingest → generate → analytics chain with MockLLM
- CLI subcommands via subprocess (argparse CLI; subprocess = CliRunner equivalent)
- review_view render path without interactive input
- acceptance.sh is executable and contains a real invocation
- No assessment-automation symbols in the learner package
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import subprocess
import sys
import uuid

import pytest

from learner.analytics import compute_weak_areas, generate_study_plan
from learner.cli import main
from learner.generator import generate_flashcards, generate_questions, generate_summary
from learner.ingest import ingest_text
from learner.llm import MockLLM
from learner.models import Card, CardRating, SessionRecord
from learner.review_view import (
    ReviewView,
    _display_question,
    _load_questions,
    _save_session_record,
)
from learner.session import ReviewSession, load_session_history

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_bundle(data_dir: pathlib.Path, topic: str, bundle: dict) -> pathlib.Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    p = data_dir / f"{topic}.json"
    p.write_text(json.dumps(bundle), encoding="utf-8")
    return p


def _minimal_bundle(n_cards: int = 1, n_questions: int = 1) -> dict:
    return {
        "summary": "Integration test summary.",
        "cards": [{"front": f"Front {i}", "back": f"Back {i}"} for i in range(n_cards)],
        "questions": [
            {
                "stem": f"Question {i}?",
                "choices": ["Opt A", "Opt B", "Opt C", "Opt D"],
                "answer_index": 0,
                "explanation": "A is correct.",
            }
            for i in range(n_questions)
        ],
    }


def _write_session_record(data_dir: pathlib.Path, topic: str, quality: int = 5) -> None:
    sessions_dir = data_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "session_id": str(uuid.uuid4()),
        "started_at": "2024-01-01T10:00:00",
        "duration_seconds": 60.0,
        "ratings": [{"card_front": "Q1", "topic": topic, "quality": quality}],
    }
    (sessions_dir / "20240101T100000.json").write_text(json.dumps(record), encoding="utf-8")


def _make_past_due_card(session: ReviewSession, front: str, back: str) -> None:
    """Persist an SRS state with a past due date so the card appears in the due list."""
    session._save_srs_state(Card(
        front=front,
        back=back,
        due=datetime.datetime(2000, 1, 1),
        interval=1.0,
        ease=2.5,
    ))


# ===========================================================================
# 1. ingest → generate → analytics chain with MockLLM
# ===========================================================================

class TestIngestGenerateAnalyticsChain:

    def test_mock_llm_summary_is_non_empty_string(self):
        backend = MockLLM()
        summary = generate_summary(["Photosynthesis converts light to energy."], backend)
        assert isinstance(summary, str)
        assert len(summary.strip()) > 0

    def test_mock_llm_flashcards_have_non_empty_front_and_back(self):
        backend = MockLLM()
        cards = generate_flashcards(["Any text"], backend)
        assert len(cards) >= 1
        for c in cards:
            assert c.front
            assert c.back

    def test_mock_llm_flashcards_have_sm2_initial_state(self):
        backend = MockLLM()
        cards = generate_flashcards(["Any text"], backend)
        for card in cards:
            assert card.interval == 1.0
            assert card.ease == 2.5
            assert card.due > datetime.datetime.utcnow()

    def test_mock_llm_questions_have_exactly_four_choices(self):
        backend = MockLLM()
        questions = generate_questions(["Any text"], backend)
        assert len(questions) >= 1
        for q in questions:
            assert len(q.choices) == 4

    def test_mock_llm_question_answer_index_is_in_bounds(self):
        backend = MockLLM()
        for q in generate_questions(["Any text"], backend):
            assert 0 <= q.answer_index < len(q.choices)

    def test_mock_llm_summary_is_deterministic_across_inputs(self):
        backend = MockLLM()
        assert generate_summary(["input A"], backend) == generate_summary(["input B"], backend)

    def test_ingest_text_splits_on_markdown_headings(self):
        text = "# Section One\nBody one.\n\n# Section Two\nBody two."
        chunks = ingest_text(text)
        assert len(chunks) == 2
        assert chunks[0].title == "Section One"
        assert chunks[1].title == "Section Two"

    def test_ingest_text_falls_back_to_paragraphs_without_headings(self):
        text = "First paragraph.\n\nSecond paragraph."
        chunks = ingest_text(text)
        assert len(chunks) == 2

    def test_analytics_zero_error_rate_for_perfect_score(self):
        records = [SessionRecord(
            session_id="s1",
            started_at=datetime.datetime(2024, 1, 1),
            ratings=[CardRating("q1", "bio", 5)],
            duration_seconds=10.0,
        )]
        wa = compute_weak_areas(records)
        assert wa[0].error_rate == 0.0

    def test_analytics_full_error_rate_for_zero_score(self):
        records = [SessionRecord(
            session_id="s1",
            started_at=datetime.datetime(2024, 1, 1),
            ratings=[CardRating("q1", "bio", 0)],
            duration_seconds=10.0,
        )]
        wa = compute_weak_areas(records)
        assert wa[0].error_rate == 1.0

    def test_analytics_partial_error_rate(self):
        records = [SessionRecord(
            session_id="s1",
            started_at=datetime.datetime(2024, 1, 1),
            ratings=[CardRating("q1", "chem", 0), CardRating("q2", "chem", 5)],
            duration_seconds=10.0,
        )]
        wa = compute_weak_areas(records)
        # mean=2.5, error_rate = 1 - 2.5/5 = 0.5
        assert abs(wa[0].error_rate - 0.5) < 1e-9

    def test_analytics_sorted_by_error_rate_descending(self):
        records = [SessionRecord(
            session_id="s1",
            started_at=datetime.datetime(2024, 1, 1),
            ratings=[CardRating("q1", "easy", 5), CardRating("q2", "hard", 0)],
            duration_seconds=10.0,
        )]
        wa = compute_weak_areas(records)
        assert wa[0].topic == "hard"
        assert wa[1].topic == "easy"

    def test_analytics_aggregates_multiple_sessions(self):
        records = [
            SessionRecord("s1", datetime.datetime(2024, 1, 1),
                          [CardRating("q1", "math", 1)], 10.0),
            SessionRecord("s2", datetime.datetime(2024, 1, 2),
                          [CardRating("q2", "math", 3)], 10.0),
        ]
        wa = compute_weak_areas(records)
        assert wa[0].question_count == 2
        # mean = (1+3)/2 = 2.0, error_rate = 1 - 2/5 = 0.6
        assert abs(wa[0].error_rate - 0.6) < 1e-9

    def test_study_plan_advice_includes_all_weak_topics(self):
        records = [SessionRecord(
            session_id="s1",
            started_at=datetime.datetime(2024, 1, 1),
            ratings=[CardRating("q1", "algebra", 1), CardRating("q2", "geometry", 2)],
            duration_seconds=10.0,
        )]
        wa = compute_weak_areas(records)
        plan = generate_study_plan(wa)
        assert "algebra" in plan.advice
        assert "geometry" in plan.advice

    def test_study_plan_empty_weak_areas_returns_encouraging_advice(self):
        plan = generate_study_plan([])
        assert "great" in plan.advice.lower() or "no weak" in plan.advice.lower()

    def test_cli_ingest_creates_bundle_with_all_keys(self, tmp_path, capsys):
        source = tmp_path / "bio.txt"
        source.write_text("Cells are the basic unit of life.", encoding="utf-8")
        main(["--mock", "--data-dir", str(tmp_path), "ingest", str(source)])
        data = json.loads((tmp_path / "bio.json").read_text())
        assert "summary" in data
        assert "cards" in data
        assert "questions" in data

    def test_cli_ingest_then_analytics_shows_no_history_message(self, tmp_path, capsys):
        source = tmp_path / "chem.txt"
        source.write_text("Chemistry studies matter.", encoding="utf-8")
        main(["--mock", "--data-dir", str(tmp_path), "ingest", str(source)])
        capsys.readouterr()
        main(["--mock", "--data-dir", str(tmp_path), "analytics"])
        out = capsys.readouterr().out
        assert "No session history" in out

    def test_cli_ingest_then_session_then_analytics(self, tmp_path, capsys):
        source = tmp_path / "phys.txt"
        source.write_text("Newton's laws describe motion.", encoding="utf-8")
        main(["--mock", "--data-dir", str(tmp_path), "ingest", str(source)])
        capsys.readouterr()
        _write_session_record(tmp_path, "phys", quality=1)
        main(["--mock", "--data-dir", str(tmp_path), "analytics"])
        out = capsys.readouterr().out
        assert "Sessions:" in out

    def test_cli_ingest_then_flashcards_shows_qa_pairs(self, tmp_path, capsys):
        source = tmp_path / "notes.txt"
        source.write_text("Water boils at 100 degrees Celsius.", encoding="utf-8")
        main(["--mock", "--data-dir", str(tmp_path), "ingest", str(source)])
        capsys.readouterr()
        main(["--mock", "--data-dir", str(tmp_path), "flashcards", "notes"])
        out = capsys.readouterr().out
        assert "Q:" in out
        assert "A:" in out

    def test_session_history_persisted_and_reloaded(self, tmp_path):
        _write_session_record(tmp_path, "history_topic", quality=3)
        history = load_session_history(tmp_path)
        assert len(history) == 1
        assert history[0].ratings[0].topic == "history_topic"
        assert history[0].ratings[0].quality == 3

    def test_ingest_bundle_readable_by_review_session(self, tmp_path):
        source = tmp_path / "mat.txt"
        source.write_text("Matrices are rectangular arrays.", encoding="utf-8")
        main(["--mock", "--data-dir", str(tmp_path), "ingest", str(source)])
        session = ReviewSession(data_dir=tmp_path)
        # Bundle was just created; cards are due tomorrow (interval=1) so due list is empty
        due = session._load_due_cards()
        assert isinstance(due, list)


# ===========================================================================
# 2. CLI subcommands via subprocess (argparse-based)
# ===========================================================================

class TestCLISubcommandsViaSubprocess:
    """
    Uses subprocess.run for process-level isolation — the argparse equivalent of
    click.testing.CliRunner.
    """

    def _run(self, *args, data_dir: pathlib.Path, timeout: int = 30):
        env = {**os.environ, "LEARNER_MOCK": "1"}
        cmd = [sys.executable, "-m", "learner", "--data-dir", str(data_dir)] + list(args)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            env=env,
            timeout=timeout,
        )

    # summary (= "summarise")
    def test_summary_exits_zero(self, tmp_path):
        _write_bundle(tmp_path, "topic", _minimal_bundle())
        assert self._run("summary", "topic", data_dir=tmp_path).returncode == 0

    def test_summary_prints_bundle_summary(self, tmp_path):
        _write_bundle(tmp_path, "topic", _minimal_bundle())
        r = self._run("summary", "topic", data_dir=tmp_path)
        assert "Integration test summary." in r.stdout

    def test_summary_missing_topic_exits_nonzero(self, tmp_path):
        assert self._run("summary", "ghost", data_dir=tmp_path).returncode != 0

    def test_summary_missing_topic_writes_error_to_stderr(self, tmp_path):
        r = self._run("summary", "ghost", data_dir=tmp_path)
        assert "ghost" in r.stderr or "error" in r.stderr.lower()

    # flashcards
    def test_flashcards_exits_zero(self, tmp_path):
        _write_bundle(tmp_path, "cards", _minimal_bundle(n_cards=2))
        assert self._run("flashcards", "cards", data_dir=tmp_path).returncode == 0

    def test_flashcards_lists_all_cards(self, tmp_path):
        _write_bundle(tmp_path, "cards", _minimal_bundle(n_cards=2))
        r = self._run("flashcards", "cards", data_dir=tmp_path)
        assert "Front 0" in r.stdout
        assert "Front 1" in r.stdout

    def test_flashcards_empty_bundle_prints_no_flashcards_message(self, tmp_path):
        _write_bundle(tmp_path, "empty", {"summary": "", "cards": [], "questions": []})
        r = self._run("flashcards", "empty", data_dir=tmp_path)
        assert r.returncode == 0
        assert "No flashcards" in r.stdout

    def test_flashcards_missing_topic_exits_nonzero(self, tmp_path):
        assert self._run("flashcards", "ghost", data_dir=tmp_path).returncode != 0

    # study-plan
    def test_study_plan_no_history_exits_zero(self, tmp_path):
        assert self._run("study-plan", data_dir=tmp_path).returncode == 0

    def test_study_plan_no_history_says_no_weak_areas(self, tmp_path):
        assert "No weak areas" in self._run("study-plan", data_dir=tmp_path).stdout

    def test_study_plan_with_weak_areas_mentions_topic(self, tmp_path):
        _write_session_record(tmp_path, "calculus", quality=0)
        r = self._run("study-plan", data_dir=tmp_path)
        assert r.returncode == 0
        assert "calculus" in r.stdout

    # analytics
    def test_analytics_no_history_exits_zero(self, tmp_path):
        assert self._run("analytics", data_dir=tmp_path).returncode == 0

    def test_analytics_no_history_shows_message(self, tmp_path):
        assert "No session history" in self._run("analytics", data_dir=tmp_path).stdout

    def test_analytics_with_history_shows_session_count(self, tmp_path):
        _write_session_record(tmp_path, "science", quality=3)
        r = self._run("analytics", data_dir=tmp_path)
        assert r.returncode == 0
        assert "Sessions:" in r.stdout

    # ingest
    def test_ingest_creates_json_bundle(self, tmp_path):
        source = tmp_path / "material.txt"
        source.write_text("Mitosis is cell division.", encoding="utf-8")
        r = self._run("ingest", str(source), data_dir=tmp_path)
        assert r.returncode == 0
        assert (tmp_path / "material.json").exists()

    def test_ingest_output_has_summary_cards_questions_lines(self, tmp_path):
        source = tmp_path / "notes.txt"
        source.write_text("The water cycle.", encoding="utf-8")
        r = self._run("ingest", str(source), data_dir=tmp_path)
        assert "Summary:" in r.stdout
        assert "Cards:" in r.stdout
        assert "Questions:" in r.stdout

    def test_ingest_missing_file_exits_nonzero(self, tmp_path):
        assert self._run("ingest", str(tmp_path / "ghost.txt"), data_dir=tmp_path).returncode != 0

    def test_no_subcommand_exits_zero(self, tmp_path):
        assert self._run(data_dir=tmp_path).returncode == 0


# ===========================================================================
# 3. review_view render path without interactive input
# ===========================================================================

class TestReviewViewRenderPath:

    def test_empty_session_returns_record_with_no_ratings(self, tmp_path):
        session = ReviewSession(data_dir=tmp_path)
        record = ReviewView().run(session)
        assert record.ratings == []
        assert record.duration_seconds == 0.0

    def test_empty_session_prints_nothing_due(self, tmp_path, capsys):
        ReviewView().run(ReviewSession(data_dir=tmp_path))
        assert "Nothing due" in capsys.readouterr().out

    def test_empty_session_record_has_valid_uuid(self, tmp_path):
        record = ReviewView().run(ReviewSession(data_dir=tmp_path))
        uuid.UUID(record.session_id)  # raises ValueError if invalid

    def test_future_due_card_excluded_from_session(self, tmp_path):
        _write_bundle(tmp_path, "future", {
            "summary": "s",
            "cards": [{"front": "Future Q", "back": "Future A"}],
            "questions": [],
        })
        session = ReviewSession(data_dir=tmp_path)
        session._save_srs_state(Card(
            front="Future Q",
            back="Future A",
            due=datetime.datetime.utcnow() + datetime.timedelta(days=365),
            interval=365.0,
            ease=2.5,
        ))
        assert ReviewView().run(session).ratings == []

    def test_display_question_prints_stem(self, capsys):
        from learner.models import Question
        q = Question("Speed of light?", ["1 km/s", "300 km/s", "300,000 km/s", "3M km/s"], 2, "~300k km/s")
        _display_question(2, 5, q)
        assert "Speed of light?" in capsys.readouterr().out

    def test_display_question_prints_a_through_d(self, capsys):
        from learner.models import Question
        q = Question("Test?", ["W", "X", "Y", "Z"], 0, "W.")
        _display_question(1, 1, q)
        out = capsys.readouterr().out
        for letter in ["A)", "B)", "C)", "D)"]:
            assert letter in out

    def test_display_question_includes_index_and_total(self, capsys):
        from learner.models import Question
        q = Question("Q?", ["A", "B", "C", "D"], 0, "A.")
        _display_question(3, 7, q)
        assert "[3/7]" in capsys.readouterr().out

    def test_load_questions_returns_question_topic_tuples(self, tmp_path):
        _write_bundle(tmp_path, "chem", {
            "summary": "s", "cards": [],
            "questions": [{
                "stem": "What is H2O?",
                "choices": ["Water", "Oxygen", "Hydrogen", "Helium"],
                "answer_index": 0,
                "explanation": "H2O is water.",
            }],
        })
        session = ReviewSession(data_dir=tmp_path)
        qs = _load_questions(session)
        assert len(qs) == 1
        q, topic = qs[0]
        assert q.stem == "What is H2O?"
        assert topic == "chem"

    def test_load_questions_skips_entries_with_missing_keys(self, tmp_path):
        _write_bundle(tmp_path, "mixed", {
            "summary": "s", "cards": [],
            "questions": [
                {"stem": "Good?", "choices": ["A", "B", "C", "D"], "answer_index": 0, "explanation": "A."},
                {"stem": "Bad — no choices"},
            ],
        })
        qs = _load_questions(ReviewSession(data_dir=tmp_path))
        assert len(qs) == 1

    def test_load_questions_skips_corrupt_json_bundle(self, tmp_path):
        (tmp_path / "corrupt.json").write_text("{ invalid", encoding="utf-8")
        assert _load_questions(ReviewSession(data_dir=tmp_path)) == []

    def test_load_questions_aggregates_across_multiple_bundles(self, tmp_path):
        for topic in ("a", "b"):
            _write_bundle(tmp_path, topic, {
                "summary": "s", "cards": [],
                "questions": [{
                    "stem": f"Q from {topic}?",
                    "choices": ["A", "B", "C", "D"],
                    "answer_index": 0,
                    "explanation": "A.",
                }],
            })
        qs = _load_questions(ReviewSession(data_dir=tmp_path))
        assert len(qs) == 2
        topics = {t for _, t in qs}
        assert topics == {"a", "b"}

    def test_save_session_record_writes_file_in_sessions_dir(self, tmp_path):
        session = ReviewSession(data_dir=tmp_path)
        record = SessionRecord(
            session_id="test-id-42",
            started_at=datetime.datetime(2024, 6, 1, 12, 0, 0),
            ratings=[CardRating("Term A", "bio", 4)],
            duration_seconds=90.0,
        )
        _save_session_record(session, record)
        files = list((tmp_path / "sessions").glob("*.json"))
        assert len(files) == 1
        saved = json.loads(files[0].read_text())
        assert saved["session_id"] == "test-id-42"

    def test_save_session_record_persists_all_ratings(self, tmp_path):
        session = ReviewSession(data_dir=tmp_path)
        record = SessionRecord(
            session_id="s1",
            started_at=datetime.datetime(2024, 1, 1, 8, 0, 0),
            ratings=[CardRating("Q1", "t1", 3), CardRating("Q2", "t2", 5)],
            duration_seconds=30.0,
        )
        _save_session_record(session, record)
        saved = json.loads(list((tmp_path / "sessions").glob("*.json"))[0].read_text())
        assert len(saved["ratings"]) == 2
        assert saved["ratings"][0]["quality"] == 3
        assert saved["ratings"][1]["quality"] == 5

    def test_run_with_due_card_records_quality(self, tmp_path, monkeypatch, capsys):
        _write_bundle(tmp_path, "bio", {
            "summary": "s",
            "cards": [{"front": "What is a gene?", "back": "Unit of heredity."}],
            "questions": [],
        })
        session = ReviewSession(data_dir=tmp_path)
        _make_past_due_card(session, "What is a gene?", "Unit of heredity.")

        monkeypatch.setattr("learner.review_view._getch", lambda: " ")
        monkeypatch.setattr("builtins.input", lambda _="": "4")

        record = ReviewView().run(session)
        capsys.readouterr()
        assert len(record.ratings) == 1
        assert record.ratings[0].card_front == "What is a gene?"
        assert record.ratings[0].quality == 4

    def test_run_with_due_card_invalid_then_valid_rating(self, tmp_path, monkeypatch, capsys):
        _write_bundle(tmp_path, "cells", {
            "summary": "s",
            "cards": [{"front": "What is mitosis?", "back": "Cell division."}],
            "questions": [],
        })
        session = ReviewSession(data_dir=tmp_path)
        _make_past_due_card(session, "What is mitosis?", "Cell division.")

        monkeypatch.setattr("learner.review_view._getch", lambda: " ")
        inputs = iter(["9", "3"])  # invalid first, then valid
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))

        record = ReviewView().run(session)
        capsys.readouterr()
        assert record.ratings[0].quality == 3

    def test_run_with_correct_question_answer_assigns_quality_5(self, tmp_path, monkeypatch, capsys):
        _write_bundle(tmp_path, "geo", {
            "summary": "s", "cards": [],
            "questions": [{
                "stem": "Capital of Italy?",
                "choices": ["Paris", "Rome", "Madrid", "Berlin"],
                "answer_index": 1,
                "explanation": "Rome.",
            }],
        })
        session = ReviewSession(data_dir=tmp_path)
        monkeypatch.setattr("builtins.input", lambda _="": "B")  # index 1 = correct

        record = ReviewView().run(session)
        capsys.readouterr()
        assert record.ratings[0].quality == 5

    def test_run_with_wrong_question_answer_assigns_quality_2(self, tmp_path, monkeypatch, capsys):
        _write_bundle(tmp_path, "geo2", {
            "summary": "s", "cards": [],
            "questions": [{
                "stem": "Capital of Spain?",
                "choices": ["Paris", "Rome", "Madrid", "Berlin"],
                "answer_index": 2,
                "explanation": "Madrid.",
            }],
        })
        session = ReviewSession(data_dir=tmp_path)
        monkeypatch.setattr("builtins.input", lambda _="": "A")  # wrong

        record = ReviewView().run(session)
        capsys.readouterr()
        assert record.ratings[0].quality == 2

    def test_run_with_question_saves_session_file(self, tmp_path, monkeypatch, capsys):
        _write_bundle(tmp_path, "mol", {
            "summary": "s", "cards": [],
            "questions": [{
                "stem": "What is DNA?",
                "choices": ["Protein", "Lipid", "RNA", "Deoxyribonucleic acid"],
                "answer_index": 3,
                "explanation": "DNA = Deoxyribonucleic acid.",
            }],
        })
        session = ReviewSession(data_dir=tmp_path)
        monkeypatch.setattr("builtins.input", lambda _="": "D")  # correct

        ReviewView().run(session)
        capsys.readouterr()
        assert len(list((tmp_path / "sessions").glob("*.json"))) == 1

    def test_run_with_due_card_updates_srs_state(self, tmp_path, monkeypatch, capsys):
        _write_bundle(tmp_path, "srs_test", {
            "summary": "s",
            "cards": [{"front": "SRS Q", "back": "SRS A"}],
            "questions": [],
        })
        session = ReviewSession(data_dir=tmp_path)
        _make_past_due_card(session, "SRS Q", "SRS A")

        monkeypatch.setattr("learner.review_view._getch", lambda: " ")
        monkeypatch.setattr("builtins.input", lambda _="": "5")

        ReviewView().run(session)
        capsys.readouterr()
        # Reloading SRS state should show an updated due date (not year 2000 any more)
        updated = session._load_srs_state(Card(front="SRS Q", back="SRS A"))
        assert updated.due > datetime.datetime(2000, 1, 2)


# ===========================================================================
# 4. acceptance.sh smoke-test
# ===========================================================================

_ACCEPTANCE_SCRIPT = _PROJECT_ROOT / "acceptance.sh"


class TestAcceptanceScript:

    def test_script_exists(self):
        assert _ACCEPTANCE_SCRIPT.exists(), "acceptance.sh not found at project root"

    def test_script_is_executable(self):
        assert os.access(_ACCEPTANCE_SCRIPT, os.X_OK), (
            "acceptance.sh is not executable — run: chmod +x acceptance.sh"
        )

    def test_script_starts_with_shebang(self):
        assert _ACCEPTANCE_SCRIPT.read_text(encoding="utf-8").startswith("#!")

    def test_script_contains_learner_invocation(self):
        assert "learner" in _ACCEPTANCE_SCRIPT.read_text(encoding="utf-8")

    def test_script_contains_ingest_command(self):
        assert "ingest" in _ACCEPTANCE_SCRIPT.read_text(encoding="utf-8")

    def test_script_uses_learner_mock_env_var(self):
        assert "LEARNER_MOCK" in _ACCEPTANCE_SCRIPT.read_text(encoding="utf-8")

    def test_script_references_sample_material(self):
        assert "sample_material" in _ACCEPTANCE_SCRIPT.read_text(encoding="utf-8")

    def test_script_checks_for_non_empty_summary(self):
        content = _ACCEPTANCE_SCRIPT.read_text(encoding="utf-8")
        # Script must validate summary output is present
        assert "SUMMARY" in content or "Summary" in content

    def test_script_runs_to_completion(self):
        result = subprocess.run(
            ["bash", str(_ACCEPTANCE_SCRIPT)],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=60,
        )
        assert result.returncode == 0, (
            f"acceptance.sh failed (exit {result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_script_output_contains_ok_marker(self):
        result = subprocess.run(
            ["bash", str(_ACCEPTANCE_SCRIPT)],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=60,
        )
        assert "acceptance: OK" in result.stdout


# ===========================================================================
# 5. No assessment-automation symbols
# ===========================================================================

def _learner_sources() -> str:
    """Concatenated source of every .py file in the learner package."""
    src_dir = _PROJECT_ROOT / "learner"
    return "\n".join(f.read_text(encoding="utf-8") for f in sorted(src_dir.glob("*.py")))


class TestNoAssessmentAutomationSymbols:
    """
    Verifies the learner package has no symbols that would bypass genuine user
    interaction during a review session.
    """

    def test_no_autograde_symbol(self):
        assert "autograde" not in _learner_sources().lower()

    def test_no_auto_correct_symbol(self):
        assert "auto_correct" not in _learner_sources()

    def test_no_bypass_review_symbol(self):
        assert "bypass_review" not in _learner_sources()

    def test_no_skip_review_symbol(self):
        assert "skip_review" not in _learner_sources()

    def test_no_auto_rate_symbol(self):
        assert "auto_rate" not in _learner_sources()

    def test_review_view_calls_user_input_for_ratings(self):
        rv = (_PROJECT_ROOT / "learner" / "review_view.py").read_text(encoding="utf-8")
        assert "_prompt_rating" in rv

    def test_review_view_calls_user_input_for_answers(self):
        rv = (_PROJECT_ROOT / "learner" / "review_view.py").read_text(encoding="utf-8")
        assert "_prompt_answer" in rv

    def test_review_view_uses_builtin_input(self):
        rv = (_PROJECT_ROOT / "learner" / "review_view.py").read_text(encoding="utf-8")
        assert "input(" in rv

    def test_review_view_does_not_import_llm_backend(self):
        # Rating must come from the user, not an LLM
        rv = (_PROJECT_ROOT / "learner" / "review_view.py").read_text(encoding="utf-8")
        assert "MockLLM" not in rv
        assert "ClaudeCliBackend" not in rv
        assert "LLMBackend" not in rv
