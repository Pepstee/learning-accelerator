"""
Adversarial tests for learner.session — independent authorship.

Strategy: exercise ReviewSession._load_due_cards, _save_srs_state,
_load_srs_state, and load_session_history against real filesystem paths
provided by pytest's tmp_path fixture.  No LLM subprocess calls; no mocking
of the units under test.  run() is exercised only for the zero-cards path
(no stdin needed).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pathlib

import pytest

from learner.models import Card, CardRating, SessionRecord
from learner.session import ReviewSession, load_session_history


# ── helpers ──────────────────────────────────────────────────────────────────

def _write_bundle(data_dir: pathlib.Path, topic: str, cards: list[dict]) -> None:
    bundle = {"summary": "test bundle", "cards": cards, "questions": []}
    (data_dir / f"{topic}.json").write_text(json.dumps(bundle), encoding="utf-8")


def _write_srs_state(
    srs_dir: pathlib.Path,
    front: str,
    due: datetime.datetime,
    interval: float = 6.0,
    ease: float = 2.5,
) -> None:
    srs_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(front.encode()).hexdigest()[:16]
    (srs_dir / f"{key}.json").write_text(
        json.dumps({
            "front": front,
            "back": "n/a",
            "due": due.isoformat(),
            "interval": interval,
            "ease": ease,
        }),
        encoding="utf-8",
    )


def _write_session_record(sessions_dir: pathlib.Path, filename: str, data: dict) -> None:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / filename).write_text(json.dumps(data), encoding="utf-8")


def _make_session_dict(
    session_id: str = "test-id",
    started_at: str = "2024-06-01T10:00:00",
    duration_seconds: float = 60.0,
    ratings: list[dict] | None = None,
) -> dict:
    return {
        "session_id": session_id,
        "started_at": started_at,
        "duration_seconds": duration_seconds,
        "ratings": ratings or [],
    }


# ── _load_due_cards ──────────────────────────────────────────────────────────

class TestLoadDueCards:
    def test_new_card_with_no_srs_state_is_not_due(self, tmp_path: pathlib.Path):
        # _load_due_cards snapshots `now` BEFORE instantiating Cards.  A brand-new
        # Card with no SRS state defaults due=utcnow() at creation (after the
        # snapshot), so it does NOT satisfy due <= now and is excluded.
        _write_bundle(tmp_path, "math", [{"front": "2+2?", "back": "4"}])
        session = ReviewSession(tmp_path)
        due = session._load_due_cards()
        assert due == []

    def test_card_with_explicit_past_srs_state_is_due(self, tmp_path: pathlib.Path):
        _write_bundle(tmp_path, "math", [{"front": "2+2?", "back": "4"}])
        past = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        _write_srs_state(tmp_path / "srs", "2+2?", past)
        session = ReviewSession(tmp_path)
        due = session._load_due_cards()
        assert len(due) == 1
        assert due[0][0].front == "2+2?"
        assert due[0][1] == "math"

    def test_card_with_past_due_date_is_included(self, tmp_path: pathlib.Path):
        _write_bundle(tmp_path, "science", [{"front": "What is H2O?", "back": "water"}])
        past = datetime.datetime.utcnow() - datetime.timedelta(days=5)
        _write_srs_state(tmp_path / "srs", "What is H2O?", past, interval=5.0)
        session = ReviewSession(tmp_path)
        due = session._load_due_cards()
        assert any(c.front == "What is H2O?" for c, _ in due)

    def test_card_with_future_due_date_is_excluded(self, tmp_path: pathlib.Path):
        _write_bundle(tmp_path, "history", [{"front": "WW1 start?", "back": "1914"}])
        future = datetime.datetime.utcnow() + datetime.timedelta(days=10)
        _write_srs_state(tmp_path / "srs", "WW1 start?", future, interval=10.0)
        session = ReviewSession(tmp_path)
        due = session._load_due_cards()
        assert due == []

    def test_mixed_cards_only_due_returned(self, tmp_path: pathlib.Path):
        _write_bundle(tmp_path, "biology", [
            {"front": "What is DNA?", "back": "nucleic acid"},
            {"front": "What is RNA?", "back": "ribonucleic acid"},
        ])
        past = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        future = datetime.datetime.utcnow() + datetime.timedelta(days=7)
        _write_srs_state(tmp_path / "srs", "What is DNA?", past, interval=1.0)
        _write_srs_state(tmp_path / "srs", "What is RNA?", future, interval=7.0)
        session = ReviewSession(tmp_path)
        due = session._load_due_cards()
        assert len(due) == 1
        assert due[0][0].front == "What is DNA?"

    def test_empty_data_dir_returns_empty_list(self, tmp_path: pathlib.Path):
        session = ReviewSession(tmp_path)
        assert session._load_due_cards() == []

    def test_corrupt_json_bundle_is_silently_skipped(self, tmp_path: pathlib.Path):
        (tmp_path / "broken.json").write_text("{not valid", encoding="utf-8")
        _write_bundle(tmp_path, "good", [{"front": "Q?", "back": "A"}])
        past = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        _write_srs_state(tmp_path / "srs", "Q?", past)
        session = ReviewSession(tmp_path)
        due = session._load_due_cards()
        # Only the valid bundle contributes; corrupt file skipped silently
        assert len(due) == 1
        assert due[0][0].front == "Q?"

    def test_bundle_without_cards_key_contributes_nothing(self, tmp_path: pathlib.Path):
        (tmp_path / "nocards.json").write_text(json.dumps({"summary": "x"}), encoding="utf-8")
        session = ReviewSession(tmp_path)
        assert session._load_due_cards() == []

    def test_topic_name_taken_from_bundle_stem(self, tmp_path: pathlib.Path):
        _write_bundle(tmp_path, "organic_chemistry", [{"front": "What is benzene?", "back": "C6H6"}])
        past = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        _write_srs_state(tmp_path / "srs", "What is benzene?", past)
        session = ReviewSession(tmp_path)
        due = session._load_due_cards()
        assert len(due) == 1
        assert due[0][1] == "organic_chemistry"

    def test_multiple_bundles_all_loaded(self, tmp_path: pathlib.Path):
        _write_bundle(tmp_path, "math", [{"front": "Q1", "back": "A1"}])
        _write_bundle(tmp_path, "physics", [{"front": "Q2", "back": "A2"}])
        past = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        _write_srs_state(tmp_path / "srs", "Q1", past)
        _write_srs_state(tmp_path / "srs", "Q2", past)
        session = ReviewSession(tmp_path)
        due = session._load_due_cards()
        topics = {t for _, t in due}
        assert "math" in topics
        assert "physics" in topics


# ── _save_srs_state / _load_srs_state ───────────────────────────────────────

class TestSrsStatePersistence:
    def test_save_then_load_recovers_interval(self, tmp_path: pathlib.Path):
        session = ReviewSession(tmp_path)
        future = datetime.datetime.utcnow() + datetime.timedelta(days=14)
        card = Card(front="Capital of France?", back="Paris", due=future, interval=14.0, ease=2.6)
        session._save_srs_state(card)
        recovered = session._load_srs_state(Card(front="Capital of France?", back="Paris"))
        assert recovered.interval == pytest.approx(14.0)

    def test_save_then_load_recovers_ease(self, tmp_path: pathlib.Path):
        session = ReviewSession(tmp_path)
        future = datetime.datetime.utcnow() + datetime.timedelta(days=7)
        card = Card(front="Speed of light?", back="3e8 m/s", due=future, interval=7.0, ease=2.8)
        session._save_srs_state(card)
        recovered = session._load_srs_state(Card(front="Speed of light?", back="3e8 m/s"))
        assert recovered.ease == pytest.approx(2.8)

    def test_save_then_load_recovers_exact_due_date(self, tmp_path: pathlib.Path):
        session = ReviewSession(tmp_path)
        target = datetime.datetime(2025, 8, 20, 15, 30, 0)
        card = Card(front="Round trip test", back="ans", due=target, interval=6.0, ease=2.5)
        session._save_srs_state(card)
        recovered = session._load_srs_state(Card(front="Round trip test", back="ans"))
        assert recovered.due == target

    def test_load_nonexistent_state_returns_original_card(self, tmp_path: pathlib.Path):
        session = ReviewSession(tmp_path)
        card = Card(front="Brand new card", back="No state yet")
        result = session._load_srs_state(card)
        assert result is card

    def test_save_creates_srs_subdirectory(self, tmp_path: pathlib.Path):
        session = ReviewSession(tmp_path)
        session._save_srs_state(Card(front="any", back="any"))
        assert (tmp_path / "srs").is_dir()

    def test_overwrite_persists_latest_values(self, tmp_path: pathlib.Path):
        session = ReviewSession(tmp_path)
        front = "Planck constant?"
        t1 = datetime.datetime.utcnow() + datetime.timedelta(days=6)
        session._save_srs_state(Card(front=front, back="h", due=t1, interval=6.0, ease=2.5))
        t2 = datetime.datetime.utcnow() + datetime.timedelta(days=15)
        session._save_srs_state(Card(front=front, back="h", due=t2, interval=15.0, ease=2.6))
        recovered = session._load_srs_state(Card(front=front, back="h"))
        assert recovered.interval == pytest.approx(15.0)
        assert recovered.ease == pytest.approx(2.6)

    def test_different_cards_do_not_collide(self, tmp_path: pathlib.Path):
        session = ReviewSession(tmp_path)
        t_a = datetime.datetime.utcnow() + datetime.timedelta(days=3)
        t_b = datetime.datetime.utcnow() + datetime.timedelta(days=9)
        session._save_srs_state(Card(front="Card A", back="ans", due=t_a, interval=3.0, ease=2.5))
        session._save_srs_state(Card(front="Card B", back="ans", due=t_b, interval=9.0, ease=2.5))
        r_a = session._load_srs_state(Card(front="Card A", back="ans"))
        r_b = session._load_srs_state(Card(front="Card B", back="ans"))
        assert r_a.interval == pytest.approx(3.0)
        assert r_b.interval == pytest.approx(9.0)

    def test_front_text_is_preserved_in_loaded_card(self, tmp_path: pathlib.Path):
        session = ReviewSession(tmp_path)
        future = datetime.datetime.utcnow() + datetime.timedelta(days=1)
        session._save_srs_state(Card(front="Test front", back="Test back", due=future, interval=1.0, ease=2.5))
        recovered = session._load_srs_state(Card(front="Test front", back="Test back"))
        assert recovered.front == "Test front"
        assert recovered.back == "Test back"


# ── _card_key ────────────────────────────────────────────────────────────────

class TestCardKey:
    def test_same_input_gives_same_key(self, tmp_path: pathlib.Path):
        s = ReviewSession(tmp_path)
        assert s._card_key("hello world") == s._card_key("hello world")

    def test_different_inputs_give_different_keys(self, tmp_path: pathlib.Path):
        s = ReviewSession(tmp_path)
        assert s._card_key("abc") != s._card_key("xyz")

    def test_key_length_is_sixteen(self, tmp_path: pathlib.Path):
        s = ReviewSession(tmp_path)
        assert len(s._card_key("any string")) == 16

    def test_key_matches_sha1_prefix(self, tmp_path: pathlib.Path):
        s = ReviewSession(tmp_path)
        front = "test card"
        expected = hashlib.sha1(front.encode()).hexdigest()[:16]
        assert s._card_key(front) == expected


# ── load_session_history ─────────────────────────────────────────────────────

class TestLoadSessionHistory:
    def test_no_sessions_dir_returns_empty(self, tmp_path: pathlib.Path):
        assert load_session_history(tmp_path) == []

    def test_empty_sessions_dir_returns_empty(self, tmp_path: pathlib.Path):
        (tmp_path / "sessions").mkdir()
        assert load_session_history(tmp_path) == []

    def test_single_session_parsed_correctly(self, tmp_path: pathlib.Path):
        data = _make_session_dict(
            session_id="abc-123",
            started_at="2024-03-10T08:30:00",
            duration_seconds=90.0,
            ratings=[{"card_front": "Q1", "topic": "math", "quality": 4}],
        )
        _write_session_record(tmp_path / "sessions", "20240310T083000.json", data)
        history = load_session_history(tmp_path)
        assert len(history) == 1
        rec = history[0]
        assert rec.session_id == "abc-123"
        assert rec.duration_seconds == pytest.approx(90.0)
        assert len(rec.ratings) == 1
        assert rec.ratings[0].quality == 4
        assert rec.ratings[0].topic == "math"

    def test_corrupt_session_file_is_skipped(self, tmp_path: pathlib.Path):
        (tmp_path / "sessions").mkdir()
        (tmp_path / "sessions" / "bad.json").write_text("{bad json", encoding="utf-8")
        assert load_session_history(tmp_path) == []

    def test_multiple_sessions_all_loaded(self, tmp_path: pathlib.Path):
        for i in range(3):
            data = _make_session_dict(session_id=f"id-{i}", duration_seconds=float(i * 30))
            _write_session_record(
                tmp_path / "sessions",
                f"2024010{i+1}T100000.json",
                data,
            )
        history = load_session_history(tmp_path)
        assert len(history) == 3

    def test_ratings_card_fronts_preserved(self, tmp_path: pathlib.Path):
        data = _make_session_dict(
            ratings=[
                {"card_front": "What is DNA?", "topic": "bio", "quality": 3},
                {"card_front": "What is RNA?", "topic": "bio", "quality": 5},
            ]
        )
        _write_session_record(tmp_path / "sessions", "20240101T000000.json", data)
        history = load_session_history(tmp_path)
        fronts = [r.card_front for r in history[0].ratings]
        assert "What is DNA?" in fronts
        assert "What is RNA?" in fronts

    def test_session_with_no_ratings_key_loads_empty_list(self, tmp_path: pathlib.Path):
        data = {
            "session_id": "no-ratings",
            "started_at": "2024-01-01T00:00:00",
            "duration_seconds": 0.0,
            # "ratings" key deliberately omitted
        }
        _write_session_record(tmp_path / "sessions", "20240101T000000.json", data)
        history = load_session_history(tmp_path)
        assert len(history) == 1
        assert history[0].ratings == []

    def test_started_at_parsed_to_datetime(self, tmp_path: pathlib.Path):
        data = _make_session_dict(started_at="2024-06-15T14:22:00")
        _write_session_record(tmp_path / "sessions", "20240615T142200.json", data)
        history = load_session_history(tmp_path)
        assert history[0].started_at == datetime.datetime(2024, 6, 15, 14, 22, 0)

    def test_valid_and_corrupt_files_coexist(self, tmp_path: pathlib.Path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "corrupt.json").write_text("!!!", encoding="utf-8")
        data = _make_session_dict(session_id="good")
        _write_session_record(sessions_dir, "20240101T000000.json", data)
        history = load_session_history(tmp_path)
        assert len(history) == 1
        assert history[0].session_id == "good"

    def test_round_trip_due_date_via_srs_state(self, tmp_path: pathlib.Path):
        """SRS state written by _save_srs_state and loaded back is identical."""
        session = ReviewSession(tmp_path)
        precise = datetime.datetime(2025, 12, 31, 23, 59, 59)
        card = Card(front="Round trip?", back="yes", due=precise, interval=30.0, ease=2.5)
        session._save_srs_state(card)
        recovered = session._load_srs_state(Card(front="Round trip?", back="yes"))
        assert recovered.due == precise
        assert recovered.interval == pytest.approx(30.0)


# ── run() with no due cards (no stdin required) ──────────────────────────────

class TestRunNoDueCards:
    def test_empty_data_dir_returns_session_record(self, tmp_path: pathlib.Path):
        session = ReviewSession(tmp_path)
        record = session.run()
        assert isinstance(record, SessionRecord)

    def test_no_due_cards_gives_empty_ratings(self, tmp_path: pathlib.Path):
        session = ReviewSession(tmp_path)
        record = session.run()
        assert record.ratings == []

    def test_no_due_cards_gives_zero_duration(self, tmp_path: pathlib.Path):
        session = ReviewSession(tmp_path)
        record = session.run()
        assert record.duration_seconds == pytest.approx(0.0)

    def test_no_due_cards_does_not_write_session_file(self, tmp_path: pathlib.Path):
        session = ReviewSession(tmp_path)
        session.run()
        sessions_dir = tmp_path / "sessions"
        # sessions dir is created but no .json files written when no cards reviewed
        if sessions_dir.exists():
            assert list(sessions_dir.glob("*.json")) == []
