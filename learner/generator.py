from __future__ import annotations

import datetime
import json
import re

from learner.llm import LLMBackend
from learner.models import Flashcard, Question

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

# Initial SM-2 values assigned to freshly generated cards.
_INITIAL_INTERVAL = 1.0  # days
_INITIAL_EASE = 2.5


def _extract_json(raw: str) -> str:
    stripped = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", stripped)
    if match:
        return match.group(1)
    return stripped


def _parse(raw: str) -> dict:
    json_str = _extract_json(raw)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}\nRaw output:\n{raw}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
    return data


def generate_questions(chunks: list[str], backend: LLMBackend) -> list[Question]:
    """Generate practice / mock-exam Questions from text chunks."""
    text = "\n\n".join(chunks)
    data = _parse(backend.complete(_PROMPT.format(text=text)))
    questions: list[Question] = []
    for item in data.get("questions", []):
        questions.append(Question(
            stem=item["stem"],
            choices=item["choices"],
            answer_index=item["answer_index"],
            explanation=item["explanation"],
        ))
    return questions


def generate_flashcards(chunks: list[str], backend: LLMBackend) -> list[Flashcard]:
    """Generate spaced-repetition Flashcards with SM-2-style initial scheduling."""
    text = "\n\n".join(chunks)
    data = _parse(backend.complete(_PROMPT.format(text=text)))
    due = datetime.datetime.utcnow() + datetime.timedelta(days=_INITIAL_INTERVAL)
    cards: list[Flashcard] = []
    for item in data.get("cards", []):
        cards.append(Flashcard(
            front=item["front"],
            back=item["back"],
            due=due,
            interval=_INITIAL_INTERVAL,
            ease=_INITIAL_EASE,
        ))
    return cards


def generate_summary(chunks: list[str], backend: LLMBackend) -> str:
    """Generate a concise summary from text chunks."""
    text = "\n\n".join(chunks)
    data = _parse(backend.complete(_PROMPT.format(text=text)))
    summary = data.get("summary", "")
    if not isinstance(summary, str):
        raise ValueError(f"'summary' must be a string, got {type(summary).__name__}")
    return summary
