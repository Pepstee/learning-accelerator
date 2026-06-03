"""
Adversarial tests for learner.analytics — independent authorship.

Tests exercise the real compute_weak_areas, build_analytics_report, and
generate_study_plan implementations end-to-end.  Nothing about the units
under test is mocked; all assertions are falsifiable by plausible mutations
to the production code (wrong formula, wrong sort direction, wrong advice
format, wrong accumulation logic, etc.).
"""
from __future__ import annotations

import datetime

import pytest

from learner.analytics import build_analytics_report, compute_weak_areas, generate_study_plan
from learner.models import CardRating, SessionRecord, StudyPlan, WeakArea


# ── helpers ───────────────────────────────────────────────────────────────────

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


def _weak(topic: str, error_rate: float, count: int = 1) -> WeakArea:
    return WeakArea(topic=topic, error_rate=error_rate, question_count=count)


# ── compute_weak_areas ────────────────────────────────────────────────────────

class TestComputeWeakAreasEmpty:
    def test_empty_history_returns_empty_list(self):
        assert compute_weak_areas([]) == []

    def test_session_with_no_ratings_contributes_nothing(self):
        s1 = _session([], "empty")
        s2 = _session([("Q1", "physics", 2)], "nonempty")
        result = compute_weak_areas([s1, s2])
        assert {w.topic for w in result} == {"physics"}

    def test_all_empty_sessions_returns_empty_list(self):
        assert compute_weak_areas([_session([], "a"), _session([], "b")]) == []


class TestComputeWeakAreasSingleTopic:
    def test_single_rating_topic_and_count(self):
        result = compute_weak_areas([_session([("Q1", "math", 3)])])
        assert len(result) == 1
        assert result[0].topic == "math"
        assert result[0].question_count == 1

    def test_perfect_quality_5_gives_zero_error_rate(self):
        result = compute_weak_areas([_session([("Q1", "ace", 5)])])
        assert result[0].error_rate == pytest.approx(0.0)

    def test_zero_quality_gives_full_error_rate(self):
        result = compute_weak_areas([_session([("Q1", "fail", 0)])])
        assert result[0].error_rate == pytest.approx(1.0)

    def test_quality_3_gives_error_rate_0_4(self):
        # 1 - 3/5 = 0.4
        result = compute_weak_areas([_session([("Q1", "mid", 3)])])
        assert result[0].error_rate == pytest.approx(0.4)

    def test_quality_4_gives_error_rate_0_2(self):
        result = compute_weak_areas([_session([("Q1", "good", 4)])])
        assert result[0].error_rate == pytest.approx(0.2)

    def test_quality_1_gives_error_rate_0_8(self):
        result = compute_weak_areas([_session([("Q1", "bad", 1)])])
        assert result[0].error_rate == pytest.approx(0.8)


class TestComputeWeakAreasErrorRateBounds:
    def test_error_rate_in_0_to_1_for_every_quality_value(self):
        for q in range(6):
            result = compute_weak_areas([_session([(f"Q{q}", "t", q)])])
            assert 0.0 <= result[0].error_rate <= 1.0

    def test_average_quality_2_and_4_gives_0_4(self):
        # mean = 3 → error_rate = 0.4
        result = compute_weak_areas([_session([("Q1", "avg", 2), ("Q2", "avg", 4)])])
        assert result[0].error_rate == pytest.approx(0.4)

    def test_all_perfect_ratings_gives_zero_error_rate(self):
        ratings = [(f"Q{i}", "perfect", 5) for i in range(5)]
        result = compute_weak_areas([_session(ratings)])
        assert result[0].error_rate == pytest.approx(0.0)

    def test_all_zero_ratings_gives_full_error_rate(self):
        ratings = [(f"Q{i}", "zero", 0) for i in range(5)]
        result = compute_weak_areas([_session(ratings)])
        assert result[0].error_rate == pytest.approx(1.0)


class TestComputeWeakAreasMultipleTopics:
    def test_distinct_topics_all_present(self):
        history = [_session([("Q1", "math", 3), ("Q2", "physics", 4), ("Q3", "history", 2)])]
        result = compute_weak_areas(history)
        assert {w.topic for w in result} == {"math", "physics", "history"}

    def test_sorted_by_error_rate_descending(self):
        history = [_session([
            ("Q1", "math", 1),      # 0.8
            ("Q2", "science", 4),   # 0.2
            ("Q3", "history", 3),   # 0.4
        ])]
        rates = [w.error_rate for w in compute_weak_areas(history)]
        assert rates == sorted(rates, reverse=True)

    def test_highest_error_rate_first(self):
        history = [_session([("Q1", "easy", 5), ("Q2", "hard", 0)])]
        result = compute_weak_areas(history)
        assert result[0].topic == "hard"

    def test_lowest_error_rate_last(self):
        history = [_session([("Q1", "easy", 5), ("Q2", "hard", 0), ("Q3", "mid", 3)])]
        result = compute_weak_areas(history)
        assert result[-1].topic == "easy"

    def test_question_count_per_topic(self):
        history = [_session([("Q1", "bio", 2), ("Q2", "bio", 4), ("Q3", "bio", 3)])]
        bio = next(w for w in compute_weak_areas(history) if w.topic == "bio")
        assert bio.question_count == 3


class TestComputeWeakAreasMultipleSessions:
    def test_aggregated_across_sessions(self):
        s1 = _session([("Q1", "math", 2)], "s1")
        s2 = _session([("Q2", "math", 4)], "s2")
        result = compute_weak_areas([s1, s2])
        math = next(w for w in result if w.topic == "math")
        # mean = (2+4)/2 = 3.0 → 0.4
        assert math.error_rate == pytest.approx(0.4)

    def test_question_count_accumulates_across_sessions(self):
        s1 = _session([("Q1", "bio", 3), ("Q2", "bio", 4)], "s1")
        s2 = _session([("Q3", "bio", 2)], "s2")
        bio = next(w for w in compute_weak_areas([s1, s2]) if w.topic == "bio")
        assert bio.question_count == 3

    def test_separate_topics_from_different_sessions(self):
        s1 = _session([("Q1", "math", 1)], "s1")
        s2 = _session([("Q1", "bio", 5)], "s2")
        result = compute_weak_areas([s1, s2])
        assert {w.topic for w in result} == {"math", "bio"}
        math = next(w for w in result if w.topic == "math")
        bio = next(w for w in result if w.topic == "bio")
        assert math.error_rate > bio.error_rate


class TestComputeWeakAreasAcceptanceCriteria:
    def test_weak_topic_mean_below_3_has_error_rate_above_0_4(self):
        history = [_session([("Q1", "algebra", 1), ("Q2", "algebra", 2)])]
        algebra = next(w for w in compute_weak_areas(history) if w.topic == "algebra")
        assert algebra.error_rate > 0.4

    def test_strong_topic_mean_gte_4_has_error_rate_below_0_3(self):
        history = [_session([("Q1", "strong", 4), ("Q2", "strong", 5)])]
        strong = next(w for w in compute_weak_areas(history) if w.topic == "strong")
        assert strong.error_rate < 0.3

    def test_weak_topic_ranked_above_strong_topic(self):
        history = [_session([("Q1", "weak", 1), ("Q2", "strong", 5)])]
        result = compute_weak_areas(history)
        idx = {w.topic: i for i, w in enumerate(result)}
        assert idx["weak"] < idx["strong"]


# ── build_analytics_report ────────────────────────────────────────────────────

class TestBuildAnalyticsReport:
    def test_empty_weak_areas_gives_zero_reviews(self):
        report = build_analytics_report([])
        assert report.total_reviews == 0

    def test_empty_weak_areas_gives_empty_weak_areas_list(self):
        report = build_analytics_report([])
        assert report.weak_areas == []

    def test_sessions_is_zero(self):
        report = build_analytics_report([_weak("math", 0.4, 3)])
        assert report.sessions == 0

    def test_total_reviews_sums_question_counts(self):
        areas = [_weak("math", 0.4, 3), _weak("bio", 0.6, 7)]
        report = build_analytics_report(areas)
        assert report.total_reviews == 10

    def test_single_area_total_reviews_equals_count(self):
        report = build_analytics_report([_weak("physics", 0.5, 5)])
        assert report.total_reviews == 5

    def test_weak_areas_list_preserved(self):
        areas = [_weak("a", 0.8, 2), _weak("b", 0.2, 4)]
        report = build_analytics_report(areas)
        assert report.weak_areas == areas

    def test_weak_areas_list_is_copy_not_same_object(self):
        areas = [_weak("a", 0.8, 2)]
        report = build_analytics_report(areas)
        # Should be a new list (not the same reference)
        areas.append(_weak("b", 0.3, 1))
        assert len(report.weak_areas) == 1

    def test_large_counts_summed_correctly(self):
        areas = [_weak(f"t{i}", 0.5, 100) for i in range(10)]
        report = build_analytics_report(areas)
        assert report.total_reviews == 1000

    def test_returns_analytics_report_type(self):
        from learner.models import AnalyticsReport
        report = build_analytics_report([])
        assert isinstance(report, AnalyticsReport)


# ── generate_study_plan ───────────────────────────────────────────────────────

class TestGenerateStudyPlanEmpty:
    def test_empty_weak_areas_returns_study_plan(self):
        result = generate_study_plan([])
        assert isinstance(result, StudyPlan)

    def test_empty_weak_areas_advice_contains_no_weak_areas(self):
        result = generate_study_plan([])
        assert "No weak areas" in result.advice

    def test_empty_weak_areas_advice_non_empty(self):
        result = generate_study_plan([])
        assert len(result.advice) > 0

    def test_empty_weak_areas_list_is_empty(self):
        result = generate_study_plan([])
        assert result.weak_areas == []

    def test_empty_advice_encouraging_message(self):
        # The advice should be the literal no-weak-areas message
        result = generate_study_plan([])
        assert result.advice == "No weak areas identified. Keep up the great work!"


class TestGenerateStudyPlanSingleTopic:
    def test_single_topic_returns_study_plan(self):
        result = generate_study_plan([_weak("calculus", 0.6, 5)])
        assert isinstance(result, StudyPlan)

    def test_single_topic_in_advice(self):
        result = generate_study_plan([_weak("calculus", 0.6, 5)])
        assert "calculus" in result.advice

    def test_single_topic_error_rate_in_advice_as_percent(self):
        result = generate_study_plan([_weak("algebra", 0.60, 4)])
        assert "60%" in result.advice

    def test_single_topic_question_count_in_advice(self):
        result = generate_study_plan([_weak("geometry", 0.5, 7)])
        assert "7" in result.advice

    def test_single_topic_weak_areas_list_has_one_entry(self):
        result = generate_study_plan([_weak("trig", 0.4, 3)])
        assert len(result.weak_areas) == 1
        assert result.weak_areas[0].topic == "trig"

    def test_zero_error_rate_topic_included_if_present(self):
        # A topic that has perfect accuracy is still included if passed in
        result = generate_study_plan([_weak("english", 0.0, 1)])
        assert "english" in result.advice
        assert "0%" in result.advice

    def test_full_error_rate_topic_formatted_as_100_percent(self):
        result = generate_study_plan([_weak("disaster", 1.0, 2)])
        assert "100%" in result.advice


class TestGenerateStudyPlanMultipleTopics:
    def test_all_topics_present_in_advice(self):
        areas = [_weak("topology", 0.8, 10), _weak("real_analysis", 0.6, 5)]
        result = generate_study_plan(areas)
        assert "topology" in result.advice
        assert "real_analysis" in result.advice

    def test_weak_areas_sorted_by_error_rate_descending(self):
        areas = [
            _weak("easy", 0.1, 2),
            _weak("hard", 0.9, 3),
            _weak("medium", 0.5, 4),
        ]
        result = generate_study_plan(areas)
        rates = [w.error_rate for w in result.weak_areas]
        assert rates == sorted(rates, reverse=True)

    def test_highest_error_rate_topic_first_in_list(self):
        areas = [_weak("easy", 0.1, 1), _weak("hard", 0.9, 1)]
        result = generate_study_plan(areas)
        assert result.weak_areas[0].topic == "hard"

    def test_lowest_error_rate_topic_last_in_list(self):
        areas = [_weak("easy", 0.1, 1), _weak("hard", 0.9, 1), _weak("mid", 0.5, 1)]
        result = generate_study_plan(areas)
        assert result.weak_areas[-1].topic == "easy"

    def test_highest_error_rate_topic_appears_first_in_advice(self):
        areas = [_weak("easy", 0.1, 1), _weak("hard", 0.9, 1)]
        result = generate_study_plan(areas)
        assert result.advice.index("hard") < result.advice.index("easy")

    def test_input_order_does_not_affect_output_order(self):
        areas_asc = [_weak("low", 0.1, 1), _weak("high", 0.9, 1)]
        areas_desc = [_weak("high", 0.9, 1), _weak("low", 0.1, 1)]
        r_asc = generate_study_plan(areas_asc)
        r_desc = generate_study_plan(areas_desc)
        assert [w.topic for w in r_asc.weak_areas] == [w.topic for w in r_desc.weak_areas]

    def test_weak_areas_count_preserved(self):
        areas = [_weak(f"t{i}", 0.5, 1) for i in range(5)]
        result = generate_study_plan(areas)
        assert len(result.weak_areas) == 5

    def test_all_topics_at_zero_accuracy_sorted_correctly(self):
        areas = [
            _weak("alpha", 1.0, 2),
            _weak("beta", 1.0, 3),
            _weak("gamma", 1.0, 1),
        ]
        result = generate_study_plan(areas)
        rates = [w.error_rate for w in result.weak_areas]
        assert all(r == 1.0 for r in rates)
        assert len(result.weak_areas) == 3


class TestGenerateStudyPlanAcceptanceCriteria:
    def test_lower_accuracy_appears_earlier_in_study_plan(self):
        """Lower accuracy (higher error_rate) must appear earlier — the core acceptance criterion."""
        areas = [
            _weak("mastered", 0.0, 5),   # perfect
            _weak("struggling", 0.8, 5), # very weak
            _weak("learning", 0.4, 5),   # moderate
        ]
        result = generate_study_plan(areas)
        idx = {w.topic: i for i, w in enumerate(result.weak_areas)}
        assert idx["struggling"] < idx["learning"] < idx["mastered"]

    def test_100_percent_accuracy_no_weak_areas_message(self):
        """When all topics have perfect scores, generate_study_plan([]) returns encouragement."""
        # The caller filters by accuracy; if they pass [], we get the no-weak-areas message
        result = generate_study_plan([])
        assert "No weak areas" in result.advice
        assert result.weak_areas == []

    def test_all_topics_at_zero_accuracy_all_appear_in_advice(self):
        areas = [_weak("a", 1.0, 1), _weak("b", 1.0, 1), _weak("c", 1.0, 1)]
        result = generate_study_plan(areas)
        for a in areas:
            assert a.topic in result.advice

    def test_study_plan_advice_non_empty_for_non_empty_input(self):
        result = generate_study_plan([_weak("x", 0.5, 3)])
        assert len(result.advice) > 0

    def test_study_plan_returned_for_single_topic(self):
        result = generate_study_plan([_weak("only_topic", 0.3, 2)])
        assert isinstance(result, StudyPlan)
        assert result.weak_areas[0].topic == "only_topic"
