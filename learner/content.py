from __future__ import annotations

import json
import re
from dataclasses import dataclass

from learner.llm import LLMBackend
from learner.models import Card, Question

_PROMPT = """\
You are a study-material processor. Given the text below, produce a JSON object with exactly three keys:

1. "summary" — a concise paragraph summarising the key points (string).
2. "cards" — a list of flashcard objects, each with "front" (question/term) and "back" \
(answer/definition) string fields. Aim for 5–15 cards.
3. "questions" — a list of multiple-choice question objects, each with:
   - "stem": the question text (string)
   - "choices": list of exactly 4 answer strings
   - "answer_index": integer index (0–3) of the correct choice
   - "explanation": brief explanation of why the answer is correct (string)
   Aim for 3–8 questions.

Respond with ONLY the raw JSON object — no markdown fences, no preamble, no trailing text.

TEXT:
{text}
"""


@dataclass
class ContentBundle:
    summary: str
    cards: list[Card]
    questions: list[Question]

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "cards": [{"front": c.front, "back": c.back} for c in self.cards],
            "questions": [
                {
                    "stem": q.stem,
                    "choices": q.choices,
                    "answer_index": q.answer_index,
                    "explanation": q.explanation,
                }
                for q in self.questions
            ],
        }


def _extract_json(raw: str) -> str:
    stripped = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", stripped)
    if match:
        return match.group(1)
    return stripped


class ContentProcessor:
    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    def process(self, text: str) -> ContentBundle:
        raw = self._backend.complete(_PROMPT.format(text=text))
        return self._parse(raw)

    def _parse(self, raw: str) -> ContentBundle:
        json_str = _extract_json(raw)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned invalid JSON: {exc}\nRaw output:\n{raw}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object, got {type(data).__name__}")

        missing = {"summary", "cards", "questions"} - data.keys()
        if missing:
            raise ValueError(f"LLM response missing required keys: {missing}")

        if not isinstance(data["summary"], str):
            raise ValueError(
                f"'summary' must be a string, got {type(data['summary']).__name__}"
            )

        return ContentBundle(
            summary=data["summary"],
            cards=self._parse_cards(data["cards"]),
            questions=self._parse_questions(data["questions"]),
        )

    def _parse_cards(self, raw: object) -> list[Card]:
        if not isinstance(raw, list):
            raise ValueError(f"'cards' must be a list, got {type(raw).__name__}")
        cards: list[Card] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"cards[{i}] must be an object")
            for key in ("front", "back"):
                if key not in item:
                    raise ValueError(f"cards[{i}] missing '{key}'")
                if not isinstance(item[key], str):
                    raise ValueError(f"cards[{i}]['{key}'] must be a string")
            cards.append(Card(front=item["front"], back=item["back"]))
        return cards

    def _parse_questions(self, raw: object) -> list[Question]:
        if not isinstance(raw, list):
            raise ValueError(f"'questions' must be a list, got {type(raw).__name__}")
        questions: list[Question] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"questions[{i}] must be an object")
            for key in ("stem", "choices", "answer_index", "explanation"):
                if key not in item:
                    raise ValueError(f"questions[{i}] missing '{key}'")
            if not isinstance(item["stem"], str):
                raise ValueError(f"questions[{i}]['stem'] must be a string")
            choices = item["choices"]
            if not isinstance(choices, list) or len(choices) != 4:
                raise ValueError(f"questions[{i}]['choices'] must be a list of exactly 4 strings")
            if not all(isinstance(c, str) for c in choices):
                raise ValueError(f"questions[{i}]['choices'] elements must be strings")
            idx = item["answer_index"]
            if not isinstance(idx, int) or not (0 <= idx <= 3):
                raise ValueError(f"questions[{i}]['answer_index'] must be an integer 0–3")
            if not isinstance(item["explanation"], str):
                raise ValueError(f"questions[{i}]['explanation'] must be a string")
            questions.append(
                Question(
                    stem=item["stem"],
                    choices=choices,
                    answer_index=idx,
                    explanation=item["explanation"],
                )
            )
        return questions
