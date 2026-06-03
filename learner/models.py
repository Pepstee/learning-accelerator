from __future__ import annotations

import datetime
from dataclasses import dataclass, field


@dataclass
class Flashcard:
    front: str
    back: str
    due: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    interval: float = 1.0  # days
    ease: float = 2.5

    @property
    def due_date(self) -> datetime.datetime:
        return self.due


# Backwards-compat alias used by content.py / session.py
Card = Flashcard


@dataclass
class Question:
    stem: str
    choices: list[str]
    answer_index: int
    explanation: str


@dataclass
class StudySession:
    started_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    cards_reviewed: int = 0
    correct: int = 0
    duration_seconds: float = 0.0


@dataclass
class WeakArea:
    topic: str
    error_rate: float  # 0.0–1.0
    question_count: int


@dataclass
class StudyPlan:
    weak_areas: list[WeakArea]
    advice: str
    generated_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)


@dataclass
class AnalyticsReport:
    sessions: int
    total_reviews: int
    weak_areas: list[WeakArea]
    generated_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)


@dataclass
class CardRating:
    card_front: str
    topic: str
    quality: int  # 0-5


@dataclass
class SessionRecord:
    session_id: str
    started_at: datetime.datetime
    ratings: list[CardRating]
    duration_seconds: float
