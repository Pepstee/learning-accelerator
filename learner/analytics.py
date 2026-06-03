from __future__ import annotations

from learner.llm import LLMBackend
from learner.models import SessionRecord, WeakArea

_STUDY_PLAN_PROMPT = """\
You are a study coach. The learner's spaced-repetition sessions have identified the following \
weak areas (sorted by error rate, highest first):

{weak_area_lines}

Write a concise, actionable study plan. For each topic give specific improvement techniques, \
suggested resources, and a weekly schedule. Be encouraging and practical. Respond in plain text \
without markdown headers.
"""


def compute_weak_areas(history: list[SessionRecord]) -> list[WeakArea]:
    topic_ratings: dict[str, list[int]] = {}
    for record in history:
        for rating in record.ratings:
            topic_ratings.setdefault(rating.topic, []).append(rating.quality)

    weak: list[WeakArea] = []
    for topic, ratings in topic_ratings.items():
        mean_quality = sum(ratings) / len(ratings)
        error_rate = 1.0 - (mean_quality / 5.0)
        weak.append(WeakArea(topic=topic, error_rate=error_rate, question_count=len(ratings)))

    weak.sort(key=lambda w: w.error_rate, reverse=True)
    return weak


def generate_study_plan(weak_areas: list[WeakArea], backend: LLMBackend) -> str:
    if not weak_areas:
        return "No weak areas identified. Keep up the great work!"

    lines = [
        f"- {w.topic}: error rate {w.error_rate:.0%} over {w.question_count} card(s)"
        for w in weak_areas
    ]
    prompt = _STUDY_PLAN_PROMPT.format(weak_area_lines="\n".join(lines))
    return backend.complete(prompt)
