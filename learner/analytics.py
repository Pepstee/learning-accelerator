from __future__ import annotations

from learner.llm import LLMBackend
from learner.models import AnalyticsReport, SessionRecord, StudyPlan, WeakArea


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


def build_analytics_report(weak_areas: list[WeakArea], *, sessions: int = 0) -> AnalyticsReport:
    total_reviews = sum(w.question_count for w in weak_areas)
    return AnalyticsReport(
        sessions=sessions,
        total_reviews=total_reviews,
        weak_areas=list(weak_areas),
    )


def generate_study_plan(weak_areas: list[WeakArea], backend: LLMBackend | None = None) -> StudyPlan:
    sorted_areas = sorted(weak_areas, key=lambda w: w.error_rate, reverse=True)

    if not sorted_areas:
        advice = "No weak areas identified. Keep up the great work!"
    else:
        parts: list[str] = []
        for w in sorted_areas:
            parts.append(
                f"{w.topic} ({w.error_rate:.0%} error rate, {w.question_count} review(s)):"
                " review core concepts and practise additional exercises."
            )
        advice = " ".join(parts)
        if backend is not None:
            weak_area_lines = "\n".join(
                f"- {w.topic}: error rate {w.error_rate:.0%} over {w.question_count} review(s)"
                for w in sorted_areas
            )
            generated = backend.complete(
                _STUDY_PLAN_PROMPT.format(weak_area_lines=weak_area_lines)
            ).strip()
            if generated:
                advice = f"{advice}\n\n{generated}"

    return StudyPlan(weak_areas=sorted_areas, advice=advice)
