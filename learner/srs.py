from __future__ import annotations

import datetime

from learner.models import Card


def schedule(card: Card, quality: int) -> Card:
    """Apply SM-2 algorithm and return a new Card with updated scheduling fields."""
    if not 0 <= quality <= 5:
        raise ValueError(f"quality must be 0-5, got {quality}")

    if quality < 3:
        interval = 1.0
        ease = card.ease
    else:
        interval = 6.0 if card.interval <= 1.0 else card.interval * card.ease
        ease = card.ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        ease = max(1.3, ease)

    due = datetime.datetime.utcnow() + datetime.timedelta(days=interval)
    return Card(front=card.front, back=card.back, due=due, interval=interval, ease=ease)
