"""
Adversarial tests for learner.models — independent authorship.

Strategy: exercise the real model dataclasses end-to-end.  No production
code is mocked.  Every assertion is falsifiable by a plausible mutation
(changed default, removed alias, wrong type annotation, missing field,
broken equality, wrong ordering invariant, etc.).

Serialisation round-trips use dataclasses.asdict to validate structural
fidelity without requiring explicit to_dict methods on the models.
"""
from __future__ import annotations

import dataclasses
import datetime
import json

import pytest

from learner.models import (
    AnalyticsReport,
    Card,
    CardRating,
    Flashcard,
    Question,
    SessionRecord,
    StudyPlan,
    StudySession,
    WeakArea,
)


# ── Card / Flashcard alias ────────────────────────────────────────────────────

class TestCardAlias:
    """Card must remain a backwards-compatible alias for Flashcard."""

    def test_card_is_flashcard(self):
        assert Card is Flashcard

    def test_card_instance_is_flashcard_instance(self):
        assert isinstance(Card(front="Q", back="A"), Flashcard)

    def test_flashcard_instance_is_card_instance(self):
        assert isinstance(Flashcard(front="Q", back="A"), Card)

    def test_card_and_flashcard_share_class_name(self):
        assert Card.__name__ == Flashcard.__name__


# ── Flashcard construction ────────────────────────────────────────────────────

class TestFlashcardConstruction:
    def test_front_stored_exactly(self):
        card = Flashcard(front="What is 2+2?", back="4")
        assert card.front == "What is 2+2?"

    def test_back_stored_exactly(self):
        card = Flashcard(front="Q", back="The answer is 4")
        assert card.back == "The answer is 4"

    def test_explicit_interval_stored(self):
        card = Flashcard(front="Q", back="A", interval=7.0)
        assert card.interval == pytest.approx(7.0)

    def test_explicit_ease_stored(self):
        card = Flashcard(front="Q", back="A", ease=3.1)
        assert card.ease == pytest.approx(3.1)

    def test_explicit_due_stored(self):
        due = datetime.datetime(2025, 12, 31, 0, 0, 0)
        card = Flashcard(front="Q", back="A", due=due)
        assert card.due == due


class TestFlashcardDefaults:
    def test_default_interval_is_one(self):
        assert Flashcard(front="Q", back="A").interval == pytest.approx(1.0)

    def test_default_ease_is_2_5(self):
        assert Flashcard(front="Q", back="A").ease == pytest.approx(2.5)

    def test_default_due_is_a_datetime(self):
        assert isinstance(Flashcard(front="Q", back="A").due, datetime.datetime)

    def test_default_due_is_recent(self):
        before = datetime.datetime.utcnow()
        card = Flashcard(front="Q", back="A")
        after = datetime.datetime.utcnow()
        assert before <= card.due <= after

    def test_default_factory_called_per_instance(self):
        # Two separate Flashcard instances must have independent due objects,
        # not the same reference (field(default_factory=...) not a class-level constant).
        c1 = Flashcard(front="Q1", back="A1")
        c2 = Flashcard(front="Q2", back="A2")
        assert c1.due is not c2.due


class TestFlashcardEquality:
    def _card(self, **kw):
        base = dict(
            front="Q", back="A",
            due=datetime.datetime(2025, 1, 1),
            interval=1.0, ease=2.5,
        )
        base.update(kw)
        return Flashcard(**base)

    def test_equal_fields_give_equal_cards(self):
        assert self._card() == self._card()

    def test_different_front_gives_unequal(self):
        assert self._card(front="Q1") != self._card(front="Q2")

    def test_different_back_gives_unequal(self):
        assert self._card(back="A1") != self._card(back="A2")

    def test_different_interval_gives_unequal(self):
        assert self._card(interval=1.0) != self._card(interval=6.0)

    def test_different_ease_gives_unequal(self):
        assert self._card(ease=2.5) != self._card(ease=2.8)

    def test_different_due_gives_unequal(self):
        d1 = datetime.datetime(2025, 1, 1)
        d2 = datetime.datetime(2025, 6, 1)
        assert self._card(due=d1) != self._card(due=d2)

    def test_value_equality_with_distinct_objects(self):
        c1 = self._card()
        c2 = self._card()
        assert c1 is not c2
        assert c1 == c2


# ── Flashcard round-trip ──────────────────────────────────────────────────────

class TestFlashcardRoundTrip:
    def test_asdict_has_exactly_five_fields(self):
        d = dataclasses.asdict(Flashcard(front="Q", back="A"))
        assert set(d.keys()) == {"front", "back", "due", "interval", "ease"}

    def test_front_back_survive_asdict(self):
        d = dataclasses.asdict(Flashcard(front="E=mc²?", back="Einstein's equation"))
        assert d["front"] == "E=mc²?"
        assert d["back"] == "Einstein's equation"

    def test_interval_survives_asdict(self):
        d = dataclasses.asdict(Flashcard(front="Q", back="A", interval=21.5))
        assert d["interval"] == pytest.approx(21.5)

    def test_ease_survives_asdict(self):
        d = dataclasses.asdict(Flashcard(front="Q", back="A", ease=3.14))
        assert d["ease"] == pytest.approx(3.14)

    def test_reconstruct_from_asdict(self):
        due = datetime.datetime(2025, 6, 15, 10, 0, 0)
        original = Flashcard(front="Capital of France?", back="Paris", due=due, interval=14.0, ease=2.6)
        reconstructed = Flashcard(**dataclasses.asdict(original))
        assert reconstructed == original


# ── Question ──────────────────────────────────────────────────────────────────

class TestQuestionConstruction:
    def _q(self, **kw):
        base = dict(
            stem="Which is correct?",
            choices=["A", "B", "C", "D"],
            answer_index=0,
            explanation="Because A.",
        )
        base.update(kw)
        return Question(**base)

    def test_stem_stored_exactly(self):
        q = self._q(stem="What is the powerhouse of the cell?")
        assert q.stem == "What is the powerhouse of the cell?"

    def test_choices_stored_exactly(self):
        choices = ["Nucleus", "Mitochondria", "Ribosome", "Vacuole"]
        assert self._q(choices=choices).choices == choices

    def test_answer_index_stored(self):
        assert self._q(answer_index=2).answer_index == 2

    def test_explanation_stored_exactly(self):
        q = self._q(explanation="Mitochondria is the powerhouse of the cell.")
        assert q.explanation == "Mitochondria is the powerhouse of the cell."

    def test_answer_index_zero_is_valid_lower_bound(self):
        assert self._q(answer_index=0).answer_index == 0

    def test_answer_index_three_is_valid_upper_bound(self):
        assert self._q(answer_index=3).answer_index == 3

    def test_choices_empty_list_stored(self):
        assert self._q(choices=[]).choices == []

    def test_choices_list_is_mutable(self):
        # Dataclass does NOT freeze the list — this is an invariant to preserve.
        q = self._q()
        q.choices.append("E")
        assert len(q.choices) == 5


class TestQuestionEquality:
    def _q(self, **kw):
        base = dict(
            stem="Q?", choices=["A", "B", "C", "D"], answer_index=1, explanation="B.",
        )
        base.update(kw)
        return Question(**base)

    def test_identical_questions_are_equal(self):
        assert self._q() == self._q()

    def test_different_stem_gives_unequal(self):
        assert self._q(stem="Q1?") != self._q(stem="Q2?")

    def test_different_answer_index_gives_unequal(self):
        assert self._q(answer_index=0) != self._q(answer_index=2)


class TestQuestionRoundTrip:
    def test_asdict_has_all_fields(self):
        d = dataclasses.asdict(Question(stem="Q?", choices=["A","B","C","D"], answer_index=1, explanation="B."))
        assert set(d.keys()) == {"stem", "choices", "answer_index", "explanation"}

    def test_reconstruct_from_asdict(self):
        original = Question(
            stem="What is photosynthesis?",
            choices=["Process A", "Process B", "Process C", "Process D"],
            answer_index=2,
            explanation="It is Process C.",
        )
        reconstructed = Question(**dataclasses.asdict(original))
        assert reconstructed == original

    def test_choices_list_preserved_in_asdict(self):
        choices = ["Alpha", "Beta", "Gamma", "Delta"]
        q = Question(stem="Q?", choices=choices, answer_index=0, explanation="Alpha.")
        d = dataclasses.asdict(q)
        assert d["choices"] == choices


# ── StudySession ──────────────────────────────────────────────────────────────

class TestStudySessionDefaults:
    def test_default_cards_reviewed_is_zero(self):
        assert StudySession().cards_reviewed == 0

    def test_default_correct_is_zero(self):
        assert StudySession().correct == 0

    def test_default_duration_seconds_is_zero(self):
        assert StudySession().duration_seconds == pytest.approx(0.0)

    def test_default_started_at_is_a_datetime(self):
        assert isinstance(StudySession().started_at, datetime.datetime)

    def test_default_started_at_is_recent(self):
        before = datetime.datetime.utcnow()
        session = StudySession()
        after = datetime.datetime.utcnow()
        assert before <= session.started_at <= after

    def test_two_sessions_have_independent_started_at_objects(self):
        s1 = StudySession()
        s2 = StudySession()
        assert s1.started_at is not s2.started_at


class TestStudySessionConstruction:
    def test_explicit_cards_reviewed_stored(self):
        assert StudySession(cards_reviewed=10).cards_reviewed == 10

    def test_explicit_correct_stored(self):
        assert StudySession(correct=7).correct == 7

    def test_explicit_duration_stored(self):
        assert StudySession(duration_seconds=300.5).duration_seconds == pytest.approx(300.5)

    def test_cards_reviewed_can_exceed_correct(self):
        session = StudySession(cards_reviewed=10, correct=7)
        assert session.cards_reviewed > session.correct

    def test_fields_are_mutable(self):
        session = StudySession()
        session.cards_reviewed = 5
        assert session.cards_reviewed == 5


# ── WeakArea ──────────────────────────────────────────────────────────────────

class TestWeakAreaConstruction:
    def test_topic_stored_exactly(self):
        wa = WeakArea(topic="linear_algebra", error_rate=0.5, question_count=10)
        assert wa.topic == "linear_algebra"

    def test_error_rate_stored(self):
        wa = WeakArea(topic="x", error_rate=0.75, question_count=4)
        assert wa.error_rate == pytest.approx(0.75)

    def test_question_count_stored(self):
        wa = WeakArea(topic="calculus", error_rate=0.3, question_count=15)
        assert wa.question_count == 15


class TestWeakAreaErrorRateBounds:
    """error_rate is documented as 0.0–1.0; boundary values must be stored exactly."""

    def test_error_rate_zero_lower_boundary(self):
        wa = WeakArea(topic="perfect", error_rate=0.0, question_count=1)
        assert wa.error_rate == pytest.approx(0.0)

    def test_error_rate_one_upper_boundary(self):
        wa = WeakArea(topic="failing", error_rate=1.0, question_count=1)
        assert wa.error_rate == pytest.approx(1.0)

    def test_error_rate_midpoint_stored_precisely(self):
        wa = WeakArea(topic="mid", error_rate=0.5, question_count=2)
        assert wa.error_rate == pytest.approx(0.5)

    def test_error_rate_float_precision_preserved(self):
        # Must not be rounded or truncated.
        wa = WeakArea(topic="x", error_rate=0.123456789, question_count=1)
        assert wa.error_rate == pytest.approx(0.123456789)

    @pytest.mark.parametrize("rate", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    def test_error_rate_sampled_values_stored_correctly(self, rate: float):
        wa = WeakArea(topic="t", error_rate=rate, question_count=1)
        assert wa.error_rate == pytest.approx(rate)


class TestWeakAreaEquality:
    def test_identical_weak_areas_are_equal(self):
        w1 = WeakArea(topic="math", error_rate=0.6, question_count=5)
        w2 = WeakArea(topic="math", error_rate=0.6, question_count=5)
        assert w1 == w2

    def test_different_topic_gives_unequal(self):
        assert (
            WeakArea(topic="math", error_rate=0.6, question_count=5)
            != WeakArea(topic="physics", error_rate=0.6, question_count=5)
        )

    def test_different_error_rate_gives_unequal(self):
        assert (
            WeakArea(topic="math", error_rate=0.6, question_count=5)
            != WeakArea(topic="math", error_rate=0.7, question_count=5)
        )

    def test_different_question_count_gives_unequal(self):
        assert (
            WeakArea(topic="math", error_rate=0.6, question_count=5)
            != WeakArea(topic="math", error_rate=0.6, question_count=10)
        )


class TestWeakAreaRoundTrip:
    def test_asdict_has_exactly_three_fields(self):
        d = dataclasses.asdict(WeakArea(topic="bio", error_rate=0.45, question_count=8))
        assert set(d.keys()) == {"topic", "error_rate", "question_count"}

    def test_reconstruct_from_asdict(self):
        original = WeakArea(topic="chemistry", error_rate=0.33, question_count=6)
        reconstructed = WeakArea(**dataclasses.asdict(original))
        assert reconstructed == original

    def test_json_round_trip_preserves_all_values(self):
        wa = WeakArea(topic="history", error_rate=0.55, question_count=12)
        d = json.loads(json.dumps(dataclasses.asdict(wa)))
        reconstructed = WeakArea(**d)
        assert reconstructed.topic == wa.topic
        assert reconstructed.error_rate == pytest.approx(wa.error_rate)
        assert reconstructed.question_count == wa.question_count


# ── StudyPlan ─────────────────────────────────────────────────────────────────

class TestStudyPlanConstruction:
    def test_weak_areas_stored(self):
        areas = [WeakArea(topic="math", error_rate=0.5, question_count=3)]
        plan = StudyPlan(weak_areas=areas, advice="Study more.")
        assert plan.weak_areas == areas

    def test_advice_stored_exactly(self):
        plan = StudyPlan(weak_areas=[], advice="Review chapter 3 twice a week.")
        assert plan.advice == "Review chapter 3 twice a week."

    def test_default_generated_at_is_recent(self):
        before = datetime.datetime.utcnow()
        plan = StudyPlan(weak_areas=[], advice="x")
        after = datetime.datetime.utcnow()
        assert before <= plan.generated_at <= after

    def test_explicit_generated_at_stored(self):
        ts = datetime.datetime(2025, 3, 15, 12, 0, 0)
        plan = StudyPlan(weak_areas=[], advice="x", generated_at=ts)
        assert plan.generated_at == ts

    def test_empty_weak_areas_stored(self):
        plan = StudyPlan(weak_areas=[], advice="All good!")
        assert plan.weak_areas == []

    def test_multiple_weak_areas_count_preserved(self):
        areas = [
            WeakArea(topic="math", error_rate=0.7, question_count=4),
            WeakArea(topic="science", error_rate=0.5, question_count=3),
            WeakArea(topic="history", error_rate=0.3, question_count=2),
        ]
        plan = StudyPlan(weak_areas=areas, advice="Focus on math first.")
        assert len(plan.weak_areas) == 3

    def test_two_plans_have_independent_generated_at_objects(self):
        p1 = StudyPlan(weak_areas=[], advice="x")
        p2 = StudyPlan(weak_areas=[], advice="y")
        assert p1.generated_at is not p2.generated_at


class TestStudyPlanWeakAreaInvariants:
    """Invariants on StudyPlan.weak_areas: type, ordering, mutability."""

    def test_all_items_are_weak_area_instances(self):
        areas = [
            WeakArea(topic="a", error_rate=0.2, question_count=1),
            WeakArea(topic="b", error_rate=0.4, question_count=2),
        ]
        plan = StudyPlan(weak_areas=areas, advice="x")
        for item in plan.weak_areas:
            assert isinstance(item, WeakArea)

    def test_ordering_of_weak_areas_is_preserved(self):
        # StudyPlan does not sort; it must store in the order given.
        areas = [
            WeakArea(topic="first", error_rate=0.9, question_count=1),
            WeakArea(topic="second", error_rate=0.1, question_count=1),
        ]
        plan = StudyPlan(weak_areas=areas, advice="x")
        assert plan.weak_areas[0].topic == "first"
        assert plan.weak_areas[1].topic == "second"

    def test_plan_stores_list_reference_not_defensive_copy(self):
        # DataClass does not copy the list on init.  Mutations to the original
        # list are visible through plan.weak_areas (same object).
        areas = [WeakArea(topic="x", error_rate=0.5, question_count=1)]
        plan = StudyPlan(weak_areas=areas, advice="advice")
        areas.append(WeakArea(topic="y", error_rate=0.3, question_count=2))
        assert len(plan.weak_areas) == 2

    def test_error_rates_in_plan_are_in_valid_range(self):
        areas = [
            WeakArea(topic="a", error_rate=0.0, question_count=1),
            WeakArea(topic="b", error_rate=1.0, question_count=1),
            WeakArea(topic="c", error_rate=0.5, question_count=1),
        ]
        plan = StudyPlan(weak_areas=areas, advice="x")
        for wa in plan.weak_areas:
            assert 0.0 <= wa.error_rate <= 1.0

    def test_question_counts_in_plan_are_non_negative(self):
        areas = [
            WeakArea(topic="t1", error_rate=0.5, question_count=10),
            WeakArea(topic="t2", error_rate=0.3, question_count=5),
        ]
        for wa in StudyPlan(weak_areas=areas, advice="x").weak_areas:
            assert wa.question_count >= 0

    def test_study_plan_equality(self):
        ts = datetime.datetime(2025, 1, 1)
        areas = [WeakArea(topic="math", error_rate=0.5, question_count=3)]
        p1 = StudyPlan(weak_areas=list(areas), advice="study", generated_at=ts)
        p2 = StudyPlan(weak_areas=list(areas), advice="study", generated_at=ts)
        assert p1 == p2

    def test_different_advice_gives_unequal_plans(self):
        ts = datetime.datetime(2025, 1, 1)
        p1 = StudyPlan(weak_areas=[], advice="study more", generated_at=ts)
        p2 = StudyPlan(weak_areas=[], advice="study less", generated_at=ts)
        assert p1 != p2

    def test_different_generated_at_gives_unequal_plans(self):
        p1 = StudyPlan(weak_areas=[], advice="x", generated_at=datetime.datetime(2025, 1, 1))
        p2 = StudyPlan(weak_areas=[], advice="x", generated_at=datetime.datetime(2025, 6, 1))
        assert p1 != p2


class TestStudyPlanRoundTrip:
    def test_asdict_has_all_fields(self):
        plan = StudyPlan(weak_areas=[], advice="Study daily.")
        d = dataclasses.asdict(plan)
        assert set(d.keys()) == {"weak_areas", "advice", "generated_at"}

    def test_nested_weak_areas_are_dicts_in_asdict(self):
        areas = [WeakArea(topic="math", error_rate=0.6, question_count=3)]
        plan = StudyPlan(weak_areas=areas, advice="x")
        d = dataclasses.asdict(plan)
        assert isinstance(d["weak_areas"], list)
        assert isinstance(d["weak_areas"][0], dict)

    def test_nested_weak_area_fields_preserved(self):
        areas = [WeakArea(topic="biology", error_rate=0.42, question_count=7)]
        d = dataclasses.asdict(StudyPlan(weak_areas=areas, advice="x"))
        wa_dict = d["weak_areas"][0]
        assert wa_dict["topic"] == "biology"
        assert wa_dict["error_rate"] == pytest.approx(0.42)
        assert wa_dict["question_count"] == 7

    def test_reconstruct_from_asdict(self):
        ts = datetime.datetime(2025, 7, 1, 9, 0, 0)
        areas = [WeakArea(topic="physics", error_rate=0.8, question_count=2)]
        original = StudyPlan(weak_areas=areas, advice="Review kinematics.", generated_at=ts)
        d = dataclasses.asdict(original)
        reconstructed = StudyPlan(
            weak_areas=[WeakArea(**wa) for wa in d["weak_areas"]],
            advice=d["advice"],
            generated_at=d["generated_at"],
        )
        assert reconstructed == original


# ── AnalyticsReport ───────────────────────────────────────────────────────────

class TestAnalyticsReportConstruction:
    def test_sessions_stored(self):
        assert AnalyticsReport(sessions=5, total_reviews=100, weak_areas=[]).sessions == 5

    def test_total_reviews_stored(self):
        assert AnalyticsReport(sessions=3, total_reviews=50, weak_areas=[]).total_reviews == 50

    def test_weak_areas_stored(self):
        areas = [WeakArea(topic="x", error_rate=0.4, question_count=2)]
        r = AnalyticsReport(sessions=1, total_reviews=10, weak_areas=areas)
        assert r.weak_areas == areas

    def test_default_generated_at_is_recent(self):
        before = datetime.datetime.utcnow()
        r = AnalyticsReport(sessions=0, total_reviews=0, weak_areas=[])
        after = datetime.datetime.utcnow()
        assert before <= r.generated_at <= after

    def test_zero_sessions_is_valid(self):
        r = AnalyticsReport(sessions=0, total_reviews=0, weak_areas=[])
        assert r.sessions == 0
        assert r.total_reviews == 0

    def test_two_reports_have_independent_generated_at(self):
        r1 = AnalyticsReport(sessions=0, total_reviews=0, weak_areas=[])
        r2 = AnalyticsReport(sessions=0, total_reviews=0, weak_areas=[])
        assert r1.generated_at is not r2.generated_at


class TestAnalyticsReportRoundTrip:
    def test_asdict_has_all_fields(self):
        d = dataclasses.asdict(AnalyticsReport(sessions=2, total_reviews=30, weak_areas=[]))
        assert set(d.keys()) == {"sessions", "total_reviews", "weak_areas", "generated_at"}

    def test_reconstruct_from_asdict(self):
        ts = datetime.datetime(2025, 1, 1)
        original = AnalyticsReport(sessions=3, total_reviews=45, weak_areas=[], generated_at=ts)
        d = dataclasses.asdict(original)
        reconstructed = AnalyticsReport(
            sessions=d["sessions"],
            total_reviews=d["total_reviews"],
            weak_areas=[WeakArea(**wa) for wa in d["weak_areas"]],
            generated_at=d["generated_at"],
        )
        assert reconstructed == original


# ── CardRating ────────────────────────────────────────────────────────────────

class TestCardRatingConstruction:
    def test_card_front_stored_exactly(self):
        r = CardRating(card_front="What is 2+2?", topic="math", quality=4)
        assert r.card_front == "What is 2+2?"

    def test_topic_stored_exactly(self):
        r = CardRating(card_front="Q", topic="linear_algebra", quality=3)
        assert r.topic == "linear_algebra"

    def test_quality_stored(self):
        r = CardRating(card_front="Q", topic="science", quality=5)
        assert r.quality == 5

    @pytest.mark.parametrize("quality", [0, 1, 2, 3, 4, 5])
    def test_all_quality_values_stored_correctly(self, quality: int):
        r = CardRating(card_front="Q", topic="t", quality=quality)
        assert r.quality == quality

    def test_quality_zero_lower_bound(self):
        assert CardRating(card_front="Q", topic="t", quality=0).quality == 0

    def test_quality_five_upper_bound(self):
        assert CardRating(card_front="Q", topic="t", quality=5).quality == 5


class TestCardRatingEquality:
    def test_identical_ratings_are_equal(self):
        r1 = CardRating(card_front="Q", topic="math", quality=4)
        r2 = CardRating(card_front="Q", topic="math", quality=4)
        assert r1 == r2

    def test_different_quality_gives_unequal(self):
        assert (
            CardRating(card_front="Q", topic="math", quality=3)
            != CardRating(card_front="Q", topic="math", quality=5)
        )

    def test_different_topic_gives_unequal(self):
        assert (
            CardRating(card_front="Q", topic="math", quality=4)
            != CardRating(card_front="Q", topic="science", quality=4)
        )


class TestCardRatingRoundTrip:
    def test_asdict_has_all_fields(self):
        d = dataclasses.asdict(CardRating(card_front="Q?", topic="bio", quality=3))
        assert set(d.keys()) == {"card_front", "topic", "quality"}

    def test_reconstruct_from_asdict(self):
        original = CardRating(card_front="What is DNA?", topic="biology", quality=4)
        reconstructed = CardRating(**dataclasses.asdict(original))
        assert reconstructed == original

    def test_json_round_trip(self):
        original = CardRating(card_front="Q?", topic="chem", quality=2)
        d = json.loads(json.dumps(dataclasses.asdict(original)))
        reconstructed = CardRating(**d)
        assert reconstructed == original


# ── SessionRecord ─────────────────────────────────────────────────────────────

class TestSessionRecordConstruction:
    def _rec(self, **kw):
        base = dict(
            session_id="s1",
            started_at=datetime.datetime(2024, 1, 1),
            ratings=[],
            duration_seconds=60.0,
        )
        base.update(kw)
        return SessionRecord(**base)

    def test_session_id_stored(self):
        assert self._rec(session_id="abc-123").session_id == "abc-123"

    def test_started_at_stored(self):
        ts = datetime.datetime(2024, 6, 15, 8, 30, 0)
        assert self._rec(started_at=ts).started_at == ts

    def test_empty_ratings_stored(self):
        assert self._rec(ratings=[]).ratings == []

    def test_ratings_with_items_count(self):
        ratings = [
            CardRating(card_front="Q1", topic="math", quality=4),
            CardRating(card_front="Q2", topic="bio", quality=3),
        ]
        rec = self._rec(ratings=ratings)
        assert len(rec.ratings) == 2

    def test_ratings_contents_preserved(self):
        ratings = [CardRating(card_front="Q1", topic="math", quality=4)]
        rec = self._rec(ratings=ratings)
        assert rec.ratings[0].card_front == "Q1"
        assert rec.ratings[0].topic == "math"
        assert rec.ratings[0].quality == 4

    def test_duration_stored_precisely(self):
        assert self._rec(duration_seconds=123.456).duration_seconds == pytest.approx(123.456)

    def test_zero_duration_stored(self):
        assert self._rec(duration_seconds=0.0).duration_seconds == pytest.approx(0.0)


class TestSessionRecordEquality:
    def _rec(self):
        return SessionRecord(
            session_id="s1",
            started_at=datetime.datetime(2024, 1, 1),
            ratings=[],
            duration_seconds=60.0,
        )

    def test_identical_records_are_equal(self):
        assert self._rec() == self._rec()

    def test_different_session_id_gives_unequal(self):
        r1 = self._rec()
        r2 = self._rec()
        r2.session_id = "s2"
        assert r1 != r2

    def test_different_duration_gives_unequal(self):
        r1 = self._rec()
        r2 = self._rec()
        r2.duration_seconds = 999.0
        assert r1 != r2


class TestSessionRecordRoundTrip:
    def test_asdict_has_all_fields(self):
        rec = SessionRecord(
            session_id="id-1",
            started_at=datetime.datetime(2024, 1, 1),
            ratings=[],
            duration_seconds=60.0,
        )
        d = dataclasses.asdict(rec)
        assert set(d.keys()) == {"session_id", "started_at", "ratings", "duration_seconds"}

    def test_nested_ratings_are_dicts_in_asdict(self):
        rating = CardRating(card_front="Q", topic="bio", quality=2)
        rec = SessionRecord(
            session_id="x",
            started_at=datetime.datetime(2024, 1, 1),
            ratings=[rating],
            duration_seconds=0.0,
        )
        d = dataclasses.asdict(rec)
        assert isinstance(d["ratings"][0], dict)
        assert d["ratings"][0]["card_front"] == "Q"

    def test_reconstruct_from_asdict(self):
        ts = datetime.datetime(2025, 3, 10, 14, 0, 0)
        rating = CardRating(card_front="What is X?", topic="chem", quality=3)
        original = SessionRecord(
            session_id="sess-abc",
            started_at=ts,
            ratings=[rating],
            duration_seconds=90.0,
        )
        d = dataclasses.asdict(original)
        reconstructed = SessionRecord(
            session_id=d["session_id"],
            started_at=d["started_at"],
            ratings=[CardRating(**r) for r in d["ratings"]],
            duration_seconds=d["duration_seconds"],
        )
        assert reconstructed == original


# ── Cross-model integration ───────────────────────────────────────────────────

class TestStudyPlanFromSessionPipeline:
    """StudyPlan and WeakArea play well together in an analytics pipeline."""

    def test_plan_built_from_sorted_weak_areas_preserves_order(self):
        # Simulate output of compute_weak_areas (sorted by error_rate desc).
        areas = [
            WeakArea(topic="algebra", error_rate=0.8, question_count=4),
            WeakArea(topic="geometry", error_rate=0.5, question_count=3),
            WeakArea(topic="statistics", error_rate=0.2, question_count=2),
        ]
        plan = StudyPlan(weak_areas=areas, advice="Focus on algebra first.")
        # Order must be identical to what was passed in.
        assert [wa.topic for wa in plan.weak_areas] == ["algebra", "geometry", "statistics"]

    def test_plan_error_rates_match_original_areas(self):
        areas = [
            WeakArea(topic="a", error_rate=0.9, question_count=5),
            WeakArea(topic="b", error_rate=0.4, question_count=2),
        ]
        plan = StudyPlan(weak_areas=areas, advice="Study a first.")
        assert plan.weak_areas[0].error_rate == pytest.approx(0.9)
        assert plan.weak_areas[1].error_rate == pytest.approx(0.4)

    def test_analytics_report_uses_same_weak_area_model(self):
        areas = [WeakArea(topic="math", error_rate=0.6, question_count=3)]
        plan = StudyPlan(weak_areas=list(areas), advice="x")
        report = AnalyticsReport(sessions=1, total_reviews=3, weak_areas=list(areas))
        # Both model types share the same WeakArea dataclass.
        assert isinstance(plan.weak_areas[0], WeakArea)
        assert isinstance(report.weak_areas[0], WeakArea)
        assert plan.weak_areas[0] == report.weak_areas[0]

    def test_session_record_ratings_are_card_ratings(self):
        ratings = [CardRating(card_front="Q1", topic="bio", quality=4)]
        rec = SessionRecord(
            session_id="s1",
            started_at=datetime.datetime(2024, 1, 1),
            ratings=ratings,
            duration_seconds=30.0,
        )
        assert all(isinstance(r, CardRating) for r in rec.ratings)


class TestDefaultTimestampFactoryIsolation:
    """All timestamp defaults must be factory-based, not shared class-level constants."""

    def test_flashcard_due_factory(self):
        c1 = Flashcard(front="Q1", back="A1")
        c2 = Flashcard(front="Q2", back="A2")
        assert c1.due is not c2.due

    def test_study_session_started_at_factory(self):
        s1 = StudySession()
        s2 = StudySession()
        assert s1.started_at is not s2.started_at

    def test_study_plan_generated_at_factory(self):
        p1 = StudyPlan(weak_areas=[], advice="x")
        p2 = StudyPlan(weak_areas=[], advice="y")
        assert p1.generated_at is not p2.generated_at

    def test_analytics_report_generated_at_factory(self):
        r1 = AnalyticsReport(sessions=0, total_reviews=0, weak_areas=[])
        r2 = AnalyticsReport(sessions=0, total_reviews=0, weak_areas=[])
        assert r1.generated_at is not r2.generated_at
