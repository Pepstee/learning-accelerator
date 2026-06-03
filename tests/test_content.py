"""
Adversarial tests for learner.content.ContentProcessor — independent authorship.

Strategy: a local MockBackend (implements LLMBackend protocol) supplies canned JSON so
no subprocess or network call is ever made.  The real ContentProcessor + parsing logic
is exercised end-to-end; nothing about the unit under test is mocked.
"""
from __future__ import annotations

import json
import pytest

from learner.content import ContentBundle, ContentProcessor, _extract_json
from learner.models import Card, Question


# ── fixed payload the spec mandates: 3 cards, 2 questions ───────────────────

_VALID_PAYLOAD: dict = {
    "summary": "Cells are the fundamental units of life, carrying out essential biological processes.",
    "cards": [
        {
            "front": "What is photosynthesis?",
            "back": "The process by which plants convert sunlight into chemical energy stored as glucose.",
        },
        {
            "front": "What is osmosis?",
            "back": "Net movement of water molecules through a semipermeable membrane from high to low concentration.",
        },
        {
            "front": "What is mitosis?",
            "back": "Cell division that produces two genetically identical daughter cells from a single parent cell.",
        },
    ],
    "questions": [
        {
            "stem": "Which organelle is the primary site of photosynthesis?",
            "choices": ["Mitochondria", "Chloroplast", "Ribosome", "Nucleus"],
            "answer_index": 1,
            "explanation": "Chloroplasts contain chlorophyll and perform the light-dependent and Calvin-cycle reactions.",
        },
        {
            "stem": "What molecule encodes the instructions for cell division?",
            "choices": ["ATP", "RNA polymerase", "DNA", "Vacuole"],
            "answer_index": 2,
            "explanation": "DNA carries the genetic blueprint that governs when and how a cell divides.",
        },
    ],
}


class MockBackend:
    """
    Deterministic backend for testing — never calls subprocess or network.
    Returns a fixed JSON payload with exactly 3 cards and 2 questions unless
    overridden via the constructor (for error-path tests).
    """

    def __init__(self, response: str | None = None) -> None:
        self._response = response if response is not None else json.dumps(_VALID_PAYLOAD)

    def complete(self, prompt: str) -> str:  # noqa: ARG002
        return self._response


# ── helpers ──────────────────────────────────────────────────────────────────

def _processor(response: str | None = None) -> ContentProcessor:
    return ContentProcessor(MockBackend(response))


def _bundle(response: str | None = None) -> ContentBundle:
    return _processor(response).process("Dummy input text for testing.")


# ── MockBackend contract ─────────────────────────────────────────────────────

class TestMockBackendContract:
    """MockBackend itself must expose the spec-mandated fixed payload."""

    def test_complete_returns_string(self):
        backend = MockBackend()
        result = backend.complete("any prompt")
        assert isinstance(result, str)

    def test_default_response_is_valid_json(self):
        backend = MockBackend()
        data = json.loads(backend.complete("x"))
        assert isinstance(data, dict)

    def test_default_payload_has_three_cards(self):
        data = json.loads(MockBackend().complete("x"))
        assert len(data["cards"]) == 3

    def test_default_payload_has_two_questions(self):
        data = json.loads(MockBackend().complete("x"))
        assert len(data["questions"]) == 2

    def test_default_payload_has_summary(self):
        data = json.loads(MockBackend().complete("x"))
        assert "summary" in data
        assert isinstance(data["summary"], str) and data["summary"].strip()

    def test_custom_response_overrides_default(self):
        backend = MockBackend('{"summary": "x", "cards": [], "questions": []}')
        assert '"summary": "x"' in backend.complete("y")

    def test_complete_ignores_prompt_content(self):
        backend = MockBackend()
        assert backend.complete("prompt A") == backend.complete("prompt B")

    def test_no_subprocess_or_network_involved(self):
        # If complete() tried to spawn a process it would fail in the test env;
        # this trivially passes only because MockBackend avoids subprocess entirely.
        import inspect
        src = inspect.getsource(MockBackend.complete)
        assert "subprocess" not in src
        assert "urllib" not in src
        assert "requests" not in src


# ── happy-path: ContentBundle structure ─────────────────────────────────────

class TestHappyPathBundle:
    """ContentProcessor must produce a well-formed ContentBundle from the spec payload."""

    def setup_method(self):
        self.bundle = _bundle()

    def test_returns_content_bundle_instance(self):
        assert isinstance(self.bundle, ContentBundle)

    # summary
    def test_summary_is_a_string(self):
        assert isinstance(self.bundle.summary, str)

    def test_summary_is_non_empty(self):
        assert len(self.bundle.summary) > 0

    def test_summary_is_not_just_whitespace(self):
        assert self.bundle.summary.strip() != ""

    def test_summary_content_matches_payload(self):
        assert self.bundle.summary == _VALID_PAYLOAD["summary"]

    # cards
    def test_cards_count_equals_three(self):
        assert len(self.bundle.cards) == 3

    def test_cards_are_card_instances(self):
        for card in self.bundle.cards:
            assert isinstance(card, Card)

    def test_each_card_front_is_non_empty_string(self):
        for i, card in enumerate(self.bundle.cards):
            assert isinstance(card.front, str), f"cards[{i}].front not a str"
            assert card.front.strip(), f"cards[{i}].front is empty"

    def test_each_card_back_is_non_empty_string(self):
        for i, card in enumerate(self.bundle.cards):
            assert isinstance(card.back, str), f"cards[{i}].back not a str"
            assert card.back.strip(), f"cards[{i}].back is empty"

    def test_card_fronts_match_payload(self):
        for i, card in enumerate(self.bundle.cards):
            assert card.front == _VALID_PAYLOAD["cards"][i]["front"]

    def test_card_backs_match_payload(self):
        for i, card in enumerate(self.bundle.cards):
            assert card.back == _VALID_PAYLOAD["cards"][i]["back"]

    # questions
    def test_questions_count_equals_two(self):
        assert len(self.bundle.questions) == 2

    def test_questions_are_question_instances(self):
        for q in self.bundle.questions:
            assert isinstance(q, Question)

    def test_each_question_has_exactly_four_choices(self):
        for i, q in enumerate(self.bundle.questions):
            assert len(q.choices) == 4, f"questions[{i}] has {len(q.choices)} choices"

    def test_each_choice_is_a_non_empty_string(self):
        for i, q in enumerate(self.bundle.questions):
            for j, choice in enumerate(q.choices):
                assert isinstance(choice, str), f"q[{i}].choices[{j}] not str"
                assert choice.strip(), f"q[{i}].choices[{j}] is empty"

    def test_answer_index_is_integer(self):
        for i, q in enumerate(self.bundle.questions):
            assert isinstance(q.answer_index, int), f"questions[{i}].answer_index not int"

    def test_answer_index_lower_bound(self):
        for i, q in enumerate(self.bundle.questions):
            assert q.answer_index >= 0, f"questions[{i}].answer_index < 0"

    def test_answer_index_upper_bound(self):
        for i, q in enumerate(self.bundle.questions):
            assert q.answer_index <= 3, f"questions[{i}].answer_index > 3"

    def test_each_question_stem_is_non_empty_string(self):
        for i, q in enumerate(self.bundle.questions):
            assert isinstance(q.stem, str) and q.stem.strip(), f"questions[{i}].stem empty"

    def test_each_question_explanation_is_non_empty_string(self):
        for i, q in enumerate(self.bundle.questions):
            assert isinstance(q.explanation, str), f"questions[{i}].explanation not str"
            assert q.explanation.strip(), f"questions[{i}].explanation is empty"


# ── to_dict round-trip ───────────────────────────────────────────────────────

class TestContentBundleToDict:
    """ContentBundle.to_dict must round-trip to the same structure as the input payload."""

    def setup_method(self):
        self.d = _bundle().to_dict()

    def test_to_dict_has_summary_key(self):
        assert "summary" in self.d

    def test_to_dict_has_cards_key(self):
        assert "cards" in self.d

    def test_to_dict_has_questions_key(self):
        assert "questions" in self.d

    def test_to_dict_summary_matches(self):
        assert self.d["summary"] == _VALID_PAYLOAD["summary"]

    def test_to_dict_cards_count(self):
        assert len(self.d["cards"]) == 3

    def test_to_dict_card_has_front_and_back(self):
        for card in self.d["cards"]:
            assert "front" in card and "back" in card

    def test_to_dict_questions_count(self):
        assert len(self.d["questions"]) == 2

    def test_to_dict_question_has_required_keys(self):
        required = {"stem", "choices", "answer_index", "explanation"}
        for q in self.d["questions"]:
            assert required <= q.keys()

    def test_to_dict_is_json_serialisable(self):
        # Must not raise
        json.dumps(self.d)


# ── invalid JSON → ValueError ────────────────────────────────────────────────

class TestInvalidJsonRaisesValueError:
    """ContentProcessor must raise ValueError (not a bare json.JSONDecodeError) for bad output."""

    def test_garbage_string(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            _bundle("this is not json {{{{")

    def test_truncated_json(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            _bundle('{"summary": "hi", "cards":')

    def test_empty_string(self):
        with pytest.raises(ValueError):
            _bundle("")

    def test_only_whitespace(self):
        with pytest.raises(ValueError):
            _bundle("   \n\t  ")

    def test_json_number(self):
        # A bare number is valid JSON but not an object
        with pytest.raises(ValueError):
            _bundle("42")

    def test_json_null(self):
        with pytest.raises(ValueError):
            _bundle("null")

    def test_json_array_raises(self):
        with pytest.raises(ValueError, match="object"):
            _bundle("[1, 2, 3]")

    def test_json_boolean_raises(self):
        with pytest.raises(ValueError):
            _bundle("true")

    def test_error_message_includes_raw_output(self):
        bad = "definitely-not-json"
        with pytest.raises(ValueError) as exc_info:
            _bundle(bad)
        assert bad in str(exc_info.value)


# ── missing top-level keys ───────────────────────────────────────────────────

class TestMissingTopLevelKeys:

    def _payload_without(self, *keys: str) -> str:
        d = dict(_VALID_PAYLOAD)
        for k in keys:
            d.pop(k)
        return json.dumps(d)

    def test_missing_summary(self):
        with pytest.raises(ValueError, match="missing required keys"):
            _bundle(self._payload_without("summary"))

    def test_missing_cards(self):
        with pytest.raises(ValueError, match="missing required keys"):
            _bundle(self._payload_without("cards"))

    def test_missing_questions(self):
        with pytest.raises(ValueError, match="missing required keys"):
            _bundle(self._payload_without("questions"))

    def test_missing_all_three(self):
        with pytest.raises(ValueError, match="missing required keys"):
            _bundle('{"unrelated": 1}')

    def test_summary_not_a_string(self):
        d = dict(_VALID_PAYLOAD)
        d["summary"] = 42
        with pytest.raises(ValueError):
            _bundle(json.dumps(d))

    def test_summary_not_a_string_list(self):
        d = dict(_VALID_PAYLOAD)
        d["summary"] = ["line1", "line2"]
        with pytest.raises(ValueError):
            _bundle(json.dumps(d))


# ── card validation ──────────────────────────────────────────────────────────

class TestCardValidation:

    def _with_cards(self, cards) -> str:
        d = dict(_VALID_PAYLOAD)
        d["cards"] = cards
        return json.dumps(d)

    def test_cards_not_a_list_raises(self):
        with pytest.raises(ValueError, match="list"):
            _bundle(self._with_cards("not-a-list"))

    def test_cards_dict_raises(self):
        with pytest.raises(ValueError, match="list"):
            _bundle(self._with_cards({"key": "value"}))

    def test_card_not_a_dict_raises(self):
        with pytest.raises(ValueError, match="object"):
            _bundle(self._with_cards(["not-a-dict"]))

    def test_card_missing_front_raises(self):
        with pytest.raises(ValueError, match="front"):
            _bundle(self._with_cards([{"back": "b"}]))

    def test_card_missing_back_raises(self):
        with pytest.raises(ValueError, match="back"):
            _bundle(self._with_cards([{"front": "f"}]))

    def test_card_front_not_string_raises(self):
        with pytest.raises(ValueError):
            _bundle(self._with_cards([{"front": 99, "back": "b"}]))

    def test_card_back_not_string_raises(self):
        with pytest.raises(ValueError):
            _bundle(self._with_cards([{"front": "f", "back": None}]))

    def test_card_back_integer_raises(self):
        with pytest.raises(ValueError):
            _bundle(self._with_cards([{"front": "f", "back": 0}]))

    def test_empty_cards_list_is_accepted(self):
        # ContentProcessor imposes no minimum card count
        bundle = _bundle(self._with_cards([]))
        assert bundle.cards == []


# ── question validation ───────────────────────────────────────────────────────

class TestQuestionValidation:

    def _with_questions(self, questions) -> str:
        d = dict(_VALID_PAYLOAD)
        d["questions"] = questions
        return json.dumps(d)

    def _good_q(self, **overrides) -> dict:
        base = {
            "stem": "What is X?",
            "choices": ["A", "B", "C", "D"],
            "answer_index": 0,
            "explanation": "Because A.",
        }
        base.update(overrides)
        return base

    def test_questions_not_a_list_raises(self):
        with pytest.raises(ValueError, match="list"):
            _bundle(self._with_questions("bad"))

    def test_question_not_a_dict_raises(self):
        with pytest.raises(ValueError, match="object"):
            _bundle(self._with_questions(["not-a-dict"]))

    def test_question_missing_stem_raises(self):
        q = self._good_q()
        del q["stem"]
        with pytest.raises(ValueError, match="stem"):
            _bundle(self._with_questions([q]))

    def test_question_missing_choices_raises(self):
        q = self._good_q()
        del q["choices"]
        with pytest.raises(ValueError, match="choices"):
            _bundle(self._with_questions([q]))

    def test_question_missing_answer_index_raises(self):
        q = self._good_q()
        del q["answer_index"]
        with pytest.raises(ValueError, match="answer_index"):
            _bundle(self._with_questions([q]))

    def test_question_missing_explanation_raises(self):
        q = self._good_q()
        del q["explanation"]
        with pytest.raises(ValueError, match="explanation"):
            _bundle(self._with_questions([q]))

    def test_choices_not_a_list_raises(self):
        with pytest.raises(ValueError):
            _bundle(self._with_questions([self._good_q(choices="ABCD")]))

    def test_choices_too_few_raises(self):
        with pytest.raises(ValueError, match="4"):
            _bundle(self._with_questions([self._good_q(choices=["A", "B", "C"])]))

    def test_choices_too_many_raises(self):
        with pytest.raises(ValueError, match="4"):
            _bundle(self._with_questions([self._good_q(choices=["A", "B", "C", "D", "E"])]))

    def test_choices_element_not_string_raises(self):
        with pytest.raises(ValueError):
            _bundle(self._with_questions([self._good_q(choices=["A", "B", "C", 4])]))

    def test_answer_index_negative_raises(self):
        with pytest.raises(ValueError, match="answer_index"):
            _bundle(self._with_questions([self._good_q(answer_index=-1)]))

    def test_answer_index_four_raises(self):
        with pytest.raises(ValueError, match="answer_index"):
            _bundle(self._with_questions([self._good_q(answer_index=4)]))

    def test_answer_index_string_raises(self):
        with pytest.raises(ValueError):
            _bundle(self._with_questions([self._good_q(answer_index="0")]))

    def test_answer_index_float_raises(self):
        # JSON floats like 1.0 could slip through — must be rejected
        with pytest.raises(ValueError):
            _bundle(self._with_questions([self._good_q(answer_index=1.0)]))

    def test_stem_not_string_raises(self):
        with pytest.raises(ValueError):
            _bundle(self._with_questions([self._good_q(stem=123)]))

    def test_explanation_not_string_raises(self):
        with pytest.raises(ValueError):
            _bundle(self._with_questions([self._good_q(explanation=True)]))

    def test_answer_index_zero_is_valid(self):
        bundle = _bundle(self._with_questions([self._good_q(answer_index=0)]))
        assert bundle.questions[0].answer_index == 0

    def test_answer_index_three_is_valid(self):
        bundle = _bundle(self._with_questions([self._good_q(answer_index=3)]))
        assert bundle.questions[0].answer_index == 3

    def test_empty_questions_list_is_accepted(self):
        bundle = _bundle(self._with_questions([]))
        assert bundle.questions == []


# ── markdown fence stripping (_extract_json) ─────────────────────────────────

class TestExtractJson:
    """_extract_json must strip markdown fences and insignificant whitespace."""

    def test_plain_json_returned_verbatim(self):
        raw = '{"key": "value"}'
        assert _extract_json(raw) == raw

    def test_leading_trailing_whitespace_stripped(self):
        raw = '  {"key": "value"}  '
        assert _extract_json(raw) == '{"key": "value"}'

    def test_json_code_fence_stripped(self):
        raw = "```json\n{\"key\": \"value\"}\n```"
        assert _extract_json(raw) == '{"key": "value"}'

    def test_generic_code_fence_stripped(self):
        raw = "```\n{\"key\": \"value\"}\n```"
        assert _extract_json(raw) == '{"key": "value"}'

    def test_fence_with_internal_whitespace(self):
        raw = "```json\n  {\"a\": 1}  \n```"
        result = _extract_json(raw)
        # Must be parseable as JSON
        assert json.loads(result) == {"a": 1}

    def test_no_fence_multiline_json_preserved(self):
        raw = '{\n  "key": "value"\n}'
        result = _extract_json(raw)
        assert json.loads(result) == {"key": "value"}


# ── markdown-fenced JSON through ContentProcessor ────────────────────────────

class TestMarkdownFenceEndToEnd:
    """ContentProcessor must handle LLM output that wraps JSON in markdown fences."""

    def test_json_fenced_block_produces_valid_bundle(self):
        payload = json.dumps(_VALID_PAYLOAD)
        fenced = f"```json\n{payload}\n```"
        bundle = _bundle(fenced)
        assert len(bundle.cards) == 3
        assert len(bundle.questions) == 2
        assert isinstance(bundle.summary, str) and bundle.summary.strip()

    def test_generic_fenced_block_produces_valid_bundle(self):
        payload = json.dumps(_VALID_PAYLOAD)
        fenced = f"```\n{payload}\n```"
        bundle = _bundle(fenced)
        assert len(bundle.cards) == 3

    def test_fenced_invalid_json_still_raises(self):
        fenced = "```json\nnot-json\n```"
        with pytest.raises(ValueError, match="invalid JSON"):
            _bundle(fenced)


# ── prompt forwarding ─────────────────────────────────────────────────────────

class TestPromptForwarding:
    """ContentProcessor must pass the formatted prompt to the backend (not raw text)."""

    class _CapturingBackend:
        """Records whatever prompt it receives, then returns valid JSON."""

        def __init__(self):
            self.received: list[str] = []

        def complete(self, prompt: str) -> str:
            self.received.append(prompt)
            return json.dumps(_VALID_PAYLOAD)

    def test_prompt_contains_input_text(self):
        backend = self._CapturingBackend()
        processor = ContentProcessor(backend)
        processor.process("My special input text")
        assert len(backend.received) == 1
        assert "My special input text" in backend.received[0]

    def test_backend_called_exactly_once_per_process(self):
        backend = self._CapturingBackend()
        processor = ContentProcessor(backend)
        processor.process("hello")
        assert len(backend.received) == 1

    def test_different_texts_produce_different_prompts(self):
        backend = self._CapturingBackend()
        processor = ContentProcessor(backend)
        processor.process("text alpha")
        processor.process("text beta")
        assert backend.received[0] != backend.received[1]
