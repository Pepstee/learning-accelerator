"""
Adversarial tests for learner.srs.schedule — independent authorship.

Strategy: exercise the real SM-2 implementation end-to-end.  Nothing about
the unit under test is mocked; every assertion is falsifiable by a plausible
mutation of the production code (altered threshold, wrong formula coefficient,
missing clamp, etc.).
"""
from __future__ import annotations

import datetime

import pytest

from learner.models import Card
from learner.srs import schedule


# ── helpers ─────────────────────────────────────────────────────────────────

def _card(*, interval: float = 1.0, ease: float = 2.5) -> Card:
    """Return a card that is already due (due = 1 day in the past)."""
    past = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
    return Card(front="Q", back="A", due=past, interval=interval, ease=ease)


# ── input validation ─────────────────────────────────────────────────────────

class TestRejectsOutOfRangeQuality:
    def test_negative_one_raises_value_error(self):
        with pytest.raises(ValueError, match="quality"):
            schedule(_card(), -1)

    def test_six_raises_value_error(self):
        with pytest.raises(ValueError, match="quality"):
            schedule(_card(), 6)

    def test_large_positive_raises(self):
        with pytest.raises(ValueError):
            schedule(_card(), 100)

    def test_large_negative_raises(self):
        with pytest.raises(ValueError):
            schedule(_card(), -99)

    @pytest.mark.parametrize("q", [0, 1, 2, 3, 4, 5])
    def test_boundary_values_do_not_raise(self, q: int):
        schedule(_card(), q)  # must not raise


# ── quality < 3: interval reset ──────────────────────────────────────────────

class TestQualityBelowThreeResetsInterval:
    @pytest.mark.parametrize("quality", [0, 1, 2])
    def test_interval_resets_to_one(self, quality: int):
        card = _card(interval=30.0, ease=2.5)
        result = schedule(card, quality)
        assert result.interval == pytest.approx(1.0), (
            f"quality={quality}: expected interval=1.0, got {result.interval}"
        )

    @pytest.mark.parametrize("quality", [0, 1, 2])
    def test_ease_factor_is_preserved_on_reset(self, quality: int):
        card = _card(ease=2.7)
        result = schedule(card, quality)
        assert result.ease == pytest.approx(2.7), (
            f"quality={quality}: ease should not change, got {result.ease}"
        )

    def test_blackout_resets_even_from_very_long_interval(self):
        card = _card(interval=365.0, ease=2.5)
        result = schedule(card, 0)
        assert result.interval == pytest.approx(1.0)

    def test_quality_two_resets_not_six(self):
        # 2 is < 3, so interval must be 1, NOT 6 (first-review branch value)
        card = _card(interval=1.0, ease=2.5)
        result = schedule(card, 2)
        assert result.interval == pytest.approx(1.0)
        assert result.interval != pytest.approx(6.0)

    def test_due_date_advances_by_exactly_one_day(self):
        card = _card(interval=20.0)
        before = datetime.datetime.utcnow()
        result = schedule(card, 1)
        after = datetime.datetime.utcnow()
        assert before + datetime.timedelta(days=1) <= result.due
        assert result.due <= after + datetime.timedelta(days=1)


# ── quality >= 3: interval growth ────────────────────────────────────────────

class TestQualityAtLeastThreeIncreasesInterval:
    def test_first_review_always_gives_six_days(self):
        # interval <= 1.0 → fixed 6-day jump regardless of ease
        card = _card(interval=1.0, ease=3.0)
        result = schedule(card, 3)
        assert result.interval == pytest.approx(6.0)

    def test_first_review_quality_five_still_six_days(self):
        card = _card(interval=1.0, ease=2.5)
        result = schedule(card, 5)
        assert result.interval == pytest.approx(6.0)

    def test_repeat_review_multiplies_interval_by_ease(self):
        card = _card(interval=10.0, ease=2.5)
        result = schedule(card, 5)
        # interval = card.interval * card.ease = 10 * 2.5 = 25
        assert result.interval == pytest.approx(25.0)

    def test_repeat_review_quality_three_multiplies_by_ease(self):
        card = _card(interval=8.0, ease=2.0)
        result = schedule(card, 3)
        # interval = 8 * 2.0 = 16
        assert result.interval == pytest.approx(16.0)

    def test_interval_strictly_greater_than_one_for_quality_three(self):
        # Must not reset to 1 when quality >= 3
        card = _card(interval=10.0)
        result = schedule(card, 3)
        assert result.interval > 1.0

    def test_interval_at_boundary_one_uses_six_not_times_ease(self):
        # Exactly 1.0 triggers the first-review branch (<=)
        card = _card(interval=1.0, ease=5.0)
        result = schedule(card, 4)
        assert result.interval == pytest.approx(6.0)

    def test_interval_just_above_one_uses_multiplication(self):
        card = _card(interval=1.001, ease=2.0)
        result = schedule(card, 4)
        assert result.interval == pytest.approx(1.001 * 2.0)

    def test_due_date_advances_by_interval_days(self):
        card = _card(interval=10.0, ease=2.0)
        before = datetime.datetime.utcnow()
        result = schedule(card, 5)
        after = datetime.datetime.utcnow()
        expected_interval = 10.0 * 2.0  # = 20 days
        assert result.interval == pytest.approx(expected_interval)
        assert before + datetime.timedelta(days=expected_interval) <= result.due
        assert result.due <= after + datetime.timedelta(days=expected_interval)

    def test_due_date_always_in_future(self):
        now = datetime.datetime.utcnow()
        card = _card(interval=1.0, ease=2.5)
        result = schedule(card, 3)
        assert result.due > now


# ── ease factor adjustment ───────────────────────────────────────────────────

class TestEaseFactorAdjustment:
    def test_perfect_quality_increases_ease_by_0_1(self):
        # q=5: delta = 0.1 - 0*(0.08+0*0.02) = 0.1
        card = _card(ease=2.5)
        result = schedule(card, 5)
        assert result.ease == pytest.approx(2.6)

    def test_quality_four_leaves_ease_unchanged(self):
        # q=4: delta = 0.1 - 1*(0.08+1*0.02) = 0.1 - 0.10 = 0.0
        card = _card(ease=2.5)
        result = schedule(card, 4)
        assert result.ease == pytest.approx(2.5)

    def test_quality_three_decreases_ease(self):
        # q=3: delta = 0.1 - 2*(0.08+2*0.02) = 0.1 - 0.24 = -0.14
        card = _card(ease=2.5)
        result = schedule(card, 3)
        assert result.ease == pytest.approx(2.36)

    def test_ease_never_falls_below_1_3(self):
        # card already at minimum ease; quality=3 would push it lower
        card = _card(ease=1.3)
        result = schedule(card, 3)
        assert result.ease >= 1.3

    def test_ease_clamped_exactly_to_1_3(self):
        # 1.3 - 0.14 = 1.16, must be clamped to 1.3
        card = _card(ease=1.3)
        result = schedule(card, 3)
        assert result.ease == pytest.approx(1.3)

    def test_ease_proportionally_lower_for_quality_3_than_quality_5(self):
        card3 = _card(ease=2.5)
        card5 = _card(ease=2.5)
        r3 = schedule(card3, 3)
        r5 = schedule(card5, 5)
        assert r3.ease < r5.ease

    def test_ease_with_low_quality_does_not_change_interval_computation(self):
        # interval uses the card's ORIGINAL ease, not the new ease
        card = _card(interval=10.0, ease=2.5)
        result_q3 = schedule(card, 3)
        result_q5 = schedule(card, 5)
        # Both should give the same interval (original ease used in both)
        assert result_q3.interval == pytest.approx(result_q5.interval)

    def test_ease_increases_accumulate_across_sequential_schedules(self):
        card = _card(interval=1.0, ease=2.5)
        r1 = schedule(card, 5)   # ease → 2.6
        r2 = schedule(r1, 5)     # ease → 2.7
        assert r2.ease == pytest.approx(2.7)

    def test_ease_decrease_from_quality_zero_preserved_not_applied(self):
        # quality < 3 preserves ease (no decrease even at 0)
        card = _card(ease=2.5)
        result = schedule(card, 0)
        assert result.ease == pytest.approx(2.5)


# ── card identity preserved ──────────────────────────────────────────────────

class TestCardIdentityPreserved:
    def test_front_and_back_unchanged_after_schedule(self):
        card = Card(front="What is gravity?", back="9.8 m/s²")
        result = schedule(card, 5)
        assert result.front == "What is gravity?"
        assert result.back == "9.8 m/s²"

    def test_schedule_returns_new_card_object(self):
        card = _card()
        result = schedule(card, 5)
        assert result is not card

    def test_original_card_not_mutated(self):
        card = _card(interval=1.0, ease=2.5)
        original_interval = card.interval
        original_ease = card.ease
        schedule(card, 5)
        assert card.interval == original_interval
        assert card.ease == original_ease
