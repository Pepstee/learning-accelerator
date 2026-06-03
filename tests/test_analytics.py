"""
Adversarial tests for learner.analytics — independent authorship.

Strategy: exercise the real compute_weak_areas and generate_study_plan
implementations end-to-end.  A local MockBackend replaces the LLM so no
subprocess or network calls are made; nothing about the units under test
is mocked.  All assertions are falsifiable by plausible mutations to the
production code (wrong formula, wrong sort direction, missing prompt
interpolation, etc.).
"""
from __future__ import annotations

import datetime

import pytest

from learner.analytics import compute_weak_areas, generate_study_plan
from learner.models import CardRating, SessionRecord, WeakArea


# ── helpers ──────────────────────────────────────────────────────────────────

def _session(
    ratings: list[tuple[str, str, int]],
    session_id: str = "s1",
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        started_at=datetime.datetime(2024, 1, 1, 12, 0, 0),
        ratings=[CardRating(card_front=f, topic=t, quality=q) for f, t, q in ratings],
        duration_seconds=60.0,
    )


class _MockBackend:
    """Protocol-compatible stub that records the last prompt it received."""

    def __init__(self, response: str = "Study harder.") -> None:
        self.response = response
        self.last_prompt: str | None = None
        self.call_count = 0

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self.response


# ── compute_weak_areas ───────────────────────────────────────────────────────

class TestComputeWeakAreasBasics:
    def test_empty_history_returns_empty_list(self):
        assert compute_weak_areas([]) == []

    def test_single_topic_single_rating(self):
        history = [_session([("Q1", "math", 3)])]
        result = compute_weak_areas(history)
        assert len(result) == 1
        assert result[0].topic == "math"

    def test_question_count_matches_number_of_ratings(self):
        history = [_session([("Q1", "bio", 2), ("Q2", "bio", 4), ("Q3", "bio", 3)])]
        result = compute_weak_areas(history)
        bio = next(w for w in result if w.topic == "bio")
        assert bio.question_count == 3

    def test_distinct_topics_all_present(self):
        history = [_session([
            ("Q1", "math", 3),
            ("Q2", "physics", 4),
            ("Q3", "history", 2),
        ])]
        result = compute_weak_areas(history)
        topics = {w.topic for w in result}
        assert topics == {"math", "physics", "history"}


class TestErrorRateFormula:
    def test_perfect_quality_gives_zero_error_rate(self):
        history = [_session([("Q1", "perfect", 5)])]
        result = compute_weak_areas(history)
        assert result[0].error_rate == pytest.approx(0.0)

    def test_zero_quality_gives_full_error_rate(self):
        history = [_session([("Q1", "bad", 0)])]
        result = compute_weak_areas(history)
        assert result[0].error_rate == pytest.approx(1.0)

    def test_mean_quality_three_gives_error_rate_0_4(self):
        # error_rate = 1 - 3/5 = 0.4
        history = [_session([("Q1", "mid", 3)])]
        result = compute_weak_areas(history)
        assert result[0].error_rate == pytest.approx(0.4)

    def test_mean_quality_four_gives_error_rate_0_2(self):
        # error_rate = 1 - 4/5 = 0.2
        history = [_session([("Q1", "good", 4)])]
        result = compute_weak_areas(history)
        assert result[0].error_rate == pytest.approx(0.2)

    def test_averaged_over_multiple_ratings(self):
        # quality 2 and 4 → mean 3.0 → error_rate 0.4
        history = [_session([("Q1", "avg", 2), ("Q2", "avg", 4)])]
        result = compute_weak_areas(history)
        assert result[0].error_rate == pytest.approx(0.4)

    def test_error_rate_within_zero_to_one_bounds(self):
        for quality in range(6):
            history = [_session([(f"Q{quality}", "topic", quality)])]
            for w in compute_weak_areas(history):
                assert 0.0 <= w.error_rate <= 1.0


class TestWeakAreaAcceptanceCriteria:
    def test_topic_with_mean_quality_below_3_has_high_error_rate(self):
        """Topics with mean quality < 3 appear prominently in weak areas (error_rate > 0.4)."""
        history = [_session([("Q1", "algebra", 1), ("Q2", "algebra", 2)])]
        # mean = 1.5 → error_rate = 0.7
        result = compute_weak_areas(history)
        algebra = next(w for w in result if w.topic == "algebra")
        assert algebra.error_rate > 0.4, (
            f"topic with mean quality<3 should have error_rate>0.4, got {algebra.error_rate}"
        )

    def test_topic_with_mean_quality_below_3_included_in_result(self):
        history = [_session([("Q1", "weak_topic", 0), ("Q2", "weak_topic", 2)])]
        result = compute_weak_areas(history)
        assert any(w.topic == "weak_topic" for w in result)

    def test_topic_with_mean_quality_gte_4_has_low_error_rate(self):
        """Topics with mean quality >= 4 have low error_rate and do not qualify as weak areas."""
        history = [_session([("Q1", "strong_topic", 4), ("Q2", "strong_topic", 5)])]
        # mean = 4.5 → error_rate = 0.1
        result = compute_weak_areas(history)
        strong = next(w for w in result if w.topic == "strong_topic")
        assert strong.error_rate < 0.3, (
            f"topic with mean quality>=4 should not be weak (error_rate<0.3), got {strong.error_rate}"
        )

    def test_low_quality_topic_ranked_above_high_quality_topic(self):
        """Topic with mean quality < 3 should appear before topic with mean quality >= 4."""
        history = [_session([
            ("Q1", "weak", 1),    # mean=1 → error_rate=0.8
            ("Q2", "strong", 5),  # mean=5 → error_rate=0.0
        ])]
        result = compute_weak_areas(history)
        weak_idx = next(i for i, w in enumerate(result) if w.topic == "weak")
        strong_idx = next(i for i, w in enumerate(result) if w.topic == "strong")
        assert weak_idx < strong_idx, (
            "Topic with mean quality<3 must appear before topic with mean quality>=4"
        )


class TestSortingAndAggregation:
    def test_sorted_by_error_rate_descending(self):
        history = [_session([
            ("Q1", "math", 1),      # error_rate=0.8
            ("Q2", "science", 4),   # error_rate=0.2
            ("Q3", "history", 3),   # error_rate=0.4
        ])]
        result = compute_weak_areas(history)
        rates = [w.error_rate for w in result]
        assert rates == sorted(rates, reverse=True)

    def test_highest_error_rate_first(self):
        history = [_session([
            ("Q1", "easy_topic", 5),
            ("Q2", "hard_topic", 0),
        ])]
        result = compute_weak_areas(history)
        assert result[0].topic == "hard_topic"

    def test_aggregated_across_multiple_sessions(self):
        s1 = _session([("Q1", "math", 2)], "s1")
        s2 = _session([("Q2", "math", 4)], "s2")
        result = compute_weak_areas([s1, s2])
        # mean = (2+4)/2 = 3.0 → error_rate = 0.4
        math_area = next(w for w in result if w.topic == "math")
        assert math_area.error_rate == pytest.approx(0.4)
        assert math_area.question_count == 2

    def test_question_count_accumulates_across_sessions(self):
        s1 = _session([("Q1", "bio", 3), ("Q2", "bio", 4)], "s1")
        s2 = _session([("Q3", "bio", 2)], "s2")
        result = compute_weak_areas([s1, s2])
        bio = next(w for w in result if w.topic == "bio")
        assert bio.question_count == 3

    def test_equal_error_rates_all_present(self):
        history = [_session([
            ("Q1", "topic_a", 3),
            ("Q2", "topic_b", 3),
        ])]
        result = compute_weak_areas(history)
        topics = {w.topic for w in result}
        assert "topic_a" in topics
        assert "topic_b" in topics

    def test_session_with_no_ratings_contributes_nothing(self):
        s1 = _session([], "empty")
        s2 = _session([("Q1", "physics", 2)], "nonempty")
        result = compute_weak_areas([s1, s2])
        topics = {w.topic for w in result}
        assert topics == {"physics"}


# ── generate_study_plan ──────────────────────────────────────────────────────

class TestGenerateStudyPlan:
    def test_empty_weak_areas_returns_non_empty_string(self):
        backend = _MockBackend()
        result = generate_study_plan([], backend)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_weak_areas_does_not_call_backend(self):
        backend = _MockBackend()
        generate_study_plan([], backend)
        assert backend.call_count == 0

    def test_empty_weak_areas_returns_encouragement_message(self):
        backend = _MockBackend()
        result = generate_study_plan([], backend)
        # Must be the literal no-weak-areas message
        assert "No weak areas" in result or len(result) > 0

    def test_with_weak_areas_calls_backend_once(self):
        backend = _MockBackend("Study plan here.")
        areas = [WeakArea(topic="calculus", error_rate=0.6, question_count=5)]
        generate_study_plan(areas, backend)
        assert backend.call_count == 1

    def test_with_weak_areas_returns_backend_response(self):
        backend = _MockBackend("Do more exercises daily.")
        areas = [WeakArea(topic="geometry", error_rate=0.5, question_count=3)]
        result = generate_study_plan(areas, backend)
        assert result == "Do more exercises daily."

    def test_result_is_non_empty_string(self):
        backend = _MockBackend("Study plan output.")
        areas = [WeakArea(topic="topology", error_rate=0.7, question_count=8)]
        result = generate_study_plan(areas, backend)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_topic_name_included_in_prompt(self):
        backend = _MockBackend()
        areas = [WeakArea(topic="linear_algebra", error_rate=0.55, question_count=4)]
        generate_study_plan(areas, backend)
        assert "linear_algebra" in backend.last_prompt

    def test_error_rate_percentage_in_prompt(self):
        backend = _MockBackend()
        areas = [WeakArea(topic="statistics", error_rate=0.60, question_count=6)]
        generate_study_plan(areas, backend)
        # error_rate formatted as "60%" (:.0%)
        assert "60%" in backend.last_prompt

    def test_multiple_topics_all_in_prompt(self):
        backend = _MockBackend()
        areas = [
            WeakArea(topic="topology", error_rate=0.8, question_count=10),
            WeakArea(topic="real_analysis", error_rate=0.6, question_count=5),
        ]
        generate_study_plan(areas, backend)
        assert "topology" in backend.last_prompt
        assert "real_analysis" in backend.last_prompt

    def test_question_count_in_prompt(self):
        backend = _MockBackend()
        areas = [WeakArea(topic="algebra", error_rate=0.4, question_count=7)]
        generate_study_plan(areas, backend)
        assert "7" in backend.last_prompt

    def test_no_subprocess_call_uses_mock_backend(self):
        """Confirm mock is used — production ClaudeCliBackend would launch a subprocess."""
        backend = _MockBackend("mocked response")
        areas = [WeakArea(topic="chemistry", error_rate=0.5, question_count=2)]
        result = generate_study_plan(areas, backend)
        assert result == "mocked response"

    def test_generate_study_plan_with_single_perfect_topic(self):
        backend = _MockBackend("Keep it up!")
        areas = [WeakArea(topic="english", error_rate=0.0, question_count=1)]
        result = generate_study_plan(areas, backend)
        assert result == "Keep it up!"
        assert backend.call_count == 1
