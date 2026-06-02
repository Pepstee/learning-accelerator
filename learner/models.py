from __future__ import annotations

import datetime
from dataclasses import dataclass, field


@dataclass
class Card:
    front: str
    back: str
    due: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    interval: float = 1.0  # days
    ease: float = 2.5


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
