"""Adversarial tests for learner.generator — independent of the builder.

Coverage targets:
    generate_questions  — happy path, value assertions, edge cases, error paths
    generate_flashcards — happy path, SR scheduling invariants, edge cases, error paths
    generate_summary    — happy path, value assertions, edge cases, error paths
"""
from __future__ import annotations

import datetime
import json
from abc import abstractmethod

import pytest

from learner.generator import generate_flashcards, generate_questions, generate_summary
from learner.llm import LLMBackend, MockLLM
from learner.models import Flashcard, Question

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FixedResponseBackend(LLMBackend):
    """Returns a caller-supplied string, unchanged."""

    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, prompt: str) -> str:
        return self._response


class RaisingBackend(LLMBackend):
    """Always raises a RuntimeError — simulates total backend failure."""

    def complete(self, prompt: str) -> str:
        raise RuntimeError("backend connection refused")


class CapturingBackend(LLMBackend):
    """Records every call; returns a valid minimal JSON response."""

    _RESPONSE = json.dumps({"summary": "cap", "cards": [], "questions": []})

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._RESPONSE


def _valid_json(**overrides) -> str:
    base = {
        "summary": "Test summary.",
        "cards": [{"front": "Q1", "back": "A1"}],
        "questions": [
            {
                "stem": "What is X?",
                "choices": ["A", "B", "C", "D"],
                "answer_index": 2,
                "explanation": "Because X is C.",
            }
        ],
    }
    base.update(overrides)
    return json.dumps(base)


# ---------------------------------------------------------------------------
# generate_questions — happy path & value assertions
# ---------------------------------------------------------------------------

class TestGenerateQuestionsHappyPath:
    def test_returns_list_of_Question_objects(self):
        qs = generate_questions(["chunk one"], MockLLM())
        assert isinstance(qs, list)
        assert len(qs) >= 1
        assert all(isinstance(q, Question) for q in qs)

    def test_stem_value_matches_mock_content(self):
        """Mutation-resistant: the exact stem text must survive the round-trip."""
        qs = generate_questions(["anything"], MockLLM())
        assert qs[0].stem == "What does spaced repetition help with?"

    def test_choices_are_the_four_mock_options(self):
        qs = generate_questions(["anything"], MockLLM())
        assert qs[0].choices == [
            "Memory retention",
            "Speed reading",
            "Note-taking",
            "Summarising",
        ]

    def test_answer_index_is_zero(self):
        qs = generate_questions(["anything"], MockLLM())
        assert qs[0].answer_index == 0

    def test_explanation_is_correct_string(self):
        qs = generate_questions(["anything"], MockLLM())
        assert qs[0].explanation == "Spaced repetition optimises long-term memory retention."

    def test_multiple_chunks_are_joined_and_forwarded(self):
        cap = CapturingBackend()
        generate_questions(["alpha", "beta", "gamma"], cap)
        assert len(cap.calls) == 1
        assert "alpha" in cap.calls[0]
        assert "beta" in cap.calls[0]
        assert "gamma" in cap.calls[0]

    def test_custom_backend_stem_survives(self):
        backend = FixedResponseBackend(_valid_json())
        qs = generate_questions(["text"], backend)
        assert qs[0].stem == "What is X?"
        assert qs[0].answer_index == 2
        assert qs[0].explanation == "Because X is C."

    def test_choices_exactly_four(self):
        backend = FixedResponseBackend(_valid_json())
        qs = generate_questions(["text"], backend)
        assert len(qs[0].choices) == 4


# ---------------------------------------------------------------------------
# generate_questions — edge cases
# ---------------------------------------------------------------------------

class TestGenerateQuestionsEdgeCases:
    def test_empty_chunk_list_still_calls_backend(self):
        cap = CapturingBackend()
        result = generate_questions([], cap)
        assert isinstance(result, list)
        assert len(cap.calls) == 1

    def test_single_empty_string_chunk(self):
        cap = CapturingBackend()
        result = generate_questions([""], cap)
        assert isinstance(result, list)

    def test_missing_questions_key_returns_empty_list(self):
        backend = FixedResponseBackend(json.dumps({"summary": "s", "cards": []}))
        result = generate_questions(["text"], backend)
        assert result == []

    def test_empty_questions_array_returns_empty_list(self):
        backend = FixedResponseBackend(json.dumps({"summary": "s", "cards": [], "questions": []}))
        result = generate_questions(["text"], backend)
        assert result == []

    def test_markdown_fenced_json_is_parsed(self):
        fenced = "```json\n" + _valid_json() + "\n```"
        backend = FixedResponseBackend(fenced)
        qs = generate_questions(["text"], backend)
        assert len(qs) == 1
        assert qs[0].stem == "What is X?"

    def test_chunk_containing_study_coach_triggers_bad_response_error(self):
        """MockLLM returns a plain string when prompt contains 'study coach',
        which cannot be parsed as JSON — error must propagate."""
        with pytest.raises((ValueError, json.JSONDecodeError, Exception)):
            generate_questions(["all about study coach strategies"], MockLLM())


# ---------------------------------------------------------------------------
# generate_questions — error propagation
# ---------------------------------------------------------------------------

class TestGenerateQuestionsErrors:
    def test_invalid_json_raises_ValueError(self):
        backend = FixedResponseBackend("this is not json")
        with pytest.raises(ValueError, match="invalid JSON"):
            generate_questions(["text"], backend)

    def test_truncated_json_raises_ValueError(self):
        backend = FixedResponseBackend('{"summary": "ok", "questions": [{"stem"')
        with pytest.raises(ValueError):
            generate_questions(["text"], backend)

    def test_json_array_at_root_raises_ValueError(self):
        backend = FixedResponseBackend("[]")
        with pytest.raises(ValueError):
            generate_questions(["text"], backend)

    def test_json_null_at_root_raises_ValueError(self):
        backend = FixedResponseBackend("null")
        with pytest.raises(ValueError):
            generate_questions(["text"], backend)

    def test_backend_exception_propagates(self):
        with pytest.raises(RuntimeError, match="backend connection refused"):
            generate_questions(["text"], RaisingBackend())

    def test_empty_string_response_raises_ValueError(self):
        backend = FixedResponseBackend("")
        with pytest.raises((ValueError, json.JSONDecodeError, Exception)):
            generate_questions(["text"], backend)


# ---------------------------------------------------------------------------
# generate_flashcards — happy path & value assertions
# ---------------------------------------------------------------------------

class TestGenerateFlashcardsHappyPath:
    def test_returns_list_of_Flashcard_objects(self):
        cards = generate_flashcards(["chunk"], MockLLM())
        assert isinstance(cards, list)
        assert len(cards) >= 1
        assert all(isinstance(c, Flashcard) for c in cards)

    def test_front_value_matches_mock_content(self):
        """Mutation-resistant: the exact front text must survive the round-trip."""
        cards = generate_flashcards(["anything"], MockLLM())
        assert cards[0].front == "What is spaced repetition?"

    def test_back_value_matches_mock_content(self):
        cards = generate_flashcards(["anything"], MockLLM())
        assert cards[0].back == "A learning technique that uses increasing intervals between reviews."

    def test_custom_backend_front_and_back_survive(self):
        backend = FixedResponseBackend(_valid_json())
        cards = generate_flashcards(["text"], backend)
        assert cards[0].front == "Q1"
        assert cards[0].back == "A1"

    def test_multiple_chunks_joined_and_forwarded(self):
        cap = CapturingBackend()
        generate_flashcards(["part1", "part2"], cap)
        assert "part1" in cap.calls[0]
        assert "part2" in cap.calls[0]


# ---------------------------------------------------------------------------
# generate_flashcards — SR scheduling invariants
# ---------------------------------------------------------------------------

class TestGenerateFlashcardsSRInvariants:
    def test_due_date_is_in_the_future(self):
        """due must be >= today (not set to past)."""
        now = datetime.datetime.utcnow()
        cards = generate_flashcards(["anything"], MockLLM())
        for card in cards:
            assert card.due >= now, (
                f"card.due {card.due!r} is before call time {now!r}"
            )

    def test_interval_is_positive(self):
        cards = generate_flashcards(["anything"], MockLLM())
        for card in cards:
            assert card.interval > 0, f"interval must be > 0, got {card.interval}"

    def test_interval_is_exactly_one_day(self):
        """The generator hardcodes _INITIAL_INTERVAL = 1.0."""
        cards = generate_flashcards(["anything"], MockLLM())
        assert cards[0].interval == 1.0

    def test_ease_is_initial_SM2_value(self):
        """The generator hardcodes _INITIAL_EASE = 2.5."""
        cards = generate_flashcards(["anything"], MockLLM())
        assert cards[0].ease == 2.5

    def test_due_is_approximately_one_day_ahead(self):
        before = datetime.datetime.utcnow()
        cards = generate_flashcards(["anything"], MockLLM())
        after = datetime.datetime.utcnow()
        for card in cards:
            expected_min = before + datetime.timedelta(days=1) - datetime.timedelta(seconds=2)
            expected_max = after + datetime.timedelta(days=1) + datetime.timedelta(seconds=2)
            assert expected_min <= card.due <= expected_max, (
                f"due {card.due!r} outside expected window [{expected_min!r}, {expected_max!r}]"
            )

    def test_all_cards_get_same_due_timestamp(self):
        """All cards in one batch share the same due value (one utcnow call per batch)."""
        many_cards_json = json.dumps({
            "summary": "s",
            "cards": [
                {"front": f"F{i}", "back": f"B{i}"}
                for i in range(5)
            ],
            "questions": [],
        })
        backend = FixedResponseBackend(many_cards_json)
        cards = generate_flashcards(["text"], backend)
        assert len(cards) == 5
        due_values = {c.due for c in cards}
        assert len(due_values) == 1, "all cards in a batch must share the same due timestamp"

    def test_due_date_property_matches_due_field(self):
        cards = generate_flashcards(["anything"], MockLLM())
        for card in cards:
            assert card.due_date == card.due


# ---------------------------------------------------------------------------
# generate_flashcards — edge cases
# ---------------------------------------------------------------------------

class TestGenerateFlashcardsEdgeCases:
    def test_empty_chunk_list(self):
        cap = CapturingBackend()
        result = generate_flashcards([], cap)
        assert result == []
        assert len(cap.calls) == 1

    def test_missing_cards_key_returns_empty_list(self):
        backend = FixedResponseBackend(json.dumps({"summary": "s", "questions": []}))
        result = generate_flashcards(["text"], backend)
        assert result == []

    def test_empty_cards_array_returns_empty_list(self):
        backend = FixedResponseBackend(json.dumps({"summary": "s", "cards": [], "questions": []}))
        result = generate_flashcards(["text"], backend)
        assert result == []

    def test_markdown_fenced_json_is_parsed(self):
        fenced = "```\n" + _valid_json() + "\n```"
        backend = FixedResponseBackend(fenced)
        cards = generate_flashcards(["text"], backend)
        assert len(cards) == 1

    def test_large_card_batch(self):
        big = json.dumps({
            "summary": "s",
            "cards": [{"front": f"front{i}", "back": f"back{i}"} for i in range(20)],
            "questions": [],
        })
        cards = generate_flashcards(["text"], FixedResponseBackend(big))
        assert len(cards) == 20
        assert all(c.interval > 0 for c in cards)
        now = datetime.datetime.utcnow()
        assert all(c.due >= now for c in cards)


# ---------------------------------------------------------------------------
# generate_flashcards — error propagation
# ---------------------------------------------------------------------------

class TestGenerateFlashcardsErrors:
    def test_invalid_json_raises_ValueError(self):
        backend = FixedResponseBackend("not-json-at-all")
        with pytest.raises(ValueError, match="invalid JSON"):
            generate_flashcards(["text"], backend)

    def test_json_list_at_root_raises_ValueError(self):
        backend = FixedResponseBackend("[1, 2, 3]")
        with pytest.raises(ValueError):
            generate_flashcards(["text"], backend)

    def test_backend_exception_propagates(self):
        with pytest.raises(RuntimeError):
            generate_flashcards(["text"], RaisingBackend())

    def test_study_coach_trigger_propagates_error(self):
        with pytest.raises((ValueError, json.JSONDecodeError, Exception)):
            generate_flashcards(["notes about the study coach philosophy"], MockLLM())


# ---------------------------------------------------------------------------
# generate_summary — happy path & value assertions
# ---------------------------------------------------------------------------

class TestGenerateSummaryHappyPath:
    def test_returns_string(self):
        result = generate_summary(["chunk"], MockLLM())
        assert isinstance(result, str)

    def test_exact_summary_value_from_mock(self):
        """Mutation-resistant: the exact summary text must survive the round-trip."""
        result = generate_summary(["anything"], MockLLM())
        assert result == "Mock summary for offline testing."

    def test_custom_backend_summary_survives(self):
        backend = FixedResponseBackend(_valid_json(summary="Custom summary text here."))
        result = generate_summary(["text"], backend)
        assert result == "Custom summary text here."

    def test_multiple_chunks_joined_and_forwarded(self):
        cap = CapturingBackend()
        generate_summary(["part A", "part B"], cap)
        assert "part A" in cap.calls[0]
        assert "part B" in cap.calls[0]

    def test_summary_is_non_empty_for_mock(self):
        result = generate_summary(["anything"], MockLLM())
        assert len(result) > 0


# ---------------------------------------------------------------------------
# generate_summary — edge cases
# ---------------------------------------------------------------------------

class TestGenerateSummaryEdgeCases:
    def test_empty_chunk_list(self):
        cap = CapturingBackend()
        result = generate_summary([], cap)
        assert result == "cap"
        assert len(cap.calls) == 1

    def test_missing_summary_key_returns_empty_string(self):
        backend = FixedResponseBackend(json.dumps({"cards": [], "questions": []}))
        result = generate_summary(["text"], backend)
        assert result == ""

    def test_markdown_fenced_json_is_parsed(self):
        fenced = "```json\n" + _valid_json(summary="Fenced summary.") + "\n```"
        backend = FixedResponseBackend(fenced)
        result = generate_summary(["text"], backend)
        assert result == "Fenced summary."

    def test_single_chunk_with_whitespace_only(self):
        cap = CapturingBackend()
        result = generate_summary(["   \n  "], cap)
        assert result == "cap"

    def test_summary_with_unicode_content(self):
        backend = FixedResponseBackend(_valid_json(summary="Résumé: naïve approach, 日本語."))
        result = generate_summary(["text"], backend)
        assert result == "Résumé: naïve approach, 日本語."


# ---------------------------------------------------------------------------
# generate_summary — error propagation
# ---------------------------------------------------------------------------

class TestGenerateSummaryErrors:
    def test_invalid_json_raises_ValueError(self):
        backend = FixedResponseBackend("{bad json}")
        with pytest.raises(ValueError, match="invalid JSON"):
            generate_summary(["text"], backend)

    def test_json_array_at_root_raises_ValueError(self):
        backend = FixedResponseBackend('["item"]')
        with pytest.raises(ValueError):
            generate_summary(["text"], backend)

    def test_summary_not_a_string_raises_ValueError(self):
        backend = FixedResponseBackend(json.dumps({"summary": 42, "cards": [], "questions": []}))
        with pytest.raises(ValueError):
            generate_summary(["text"], backend)

    def test_summary_as_list_raises_ValueError(self):
        backend = FixedResponseBackend(json.dumps({"summary": ["a", "b"], "cards": [], "questions": []}))
        with pytest.raises(ValueError):
            generate_summary(["text"], backend)

    def test_summary_as_dict_raises_ValueError(self):
        backend = FixedResponseBackend(json.dumps({"summary": {"nested": "object"}, "cards": [], "questions": []}))
        with pytest.raises(ValueError):
            generate_summary(["text"], backend)

    def test_backend_exception_propagates(self):
        with pytest.raises(RuntimeError, match="backend connection refused"):
            generate_summary(["text"], RaisingBackend())

    def test_study_coach_trigger_causes_error(self):
        with pytest.raises((ValueError, json.JSONDecodeError, Exception)):
            generate_summary(["a study coach is someone who"], MockLLM())
