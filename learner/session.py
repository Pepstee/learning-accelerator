from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import uuid

from learner.models import Card, CardRating, SessionRecord
from learner.srs import schedule


class ReviewSession:
    def __init__(self, data_dir: pathlib.Path) -> None:
        self._data_dir = data_dir
        self._srs_dir = data_dir / "srs"
        self._sessions_dir = data_dir / "sessions"

    def _card_key(self, front: str) -> str:
        return hashlib.sha1(front.encode()).hexdigest()[:16]

    def _load_srs_state(self, card: Card) -> Card:
        path = self._srs_dir / f"{self._card_key(card.front)}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return Card(
                front=card.front,
                back=card.back,
                due=datetime.datetime.fromisoformat(data["due"]),
                interval=data["interval"],
                ease=data["ease"],
            )
        return card

    def _save_srs_state(self, card: Card) -> None:
        self._srs_dir.mkdir(parents=True, exist_ok=True)
        path = self._srs_dir / f"{self._card_key(card.front)}.json"
        path.write_text(
            json.dumps({
                "front": card.front,
                "back": card.back,
                "due": card.due.isoformat(),
                "interval": card.interval,
                "ease": card.ease,
            }),
            encoding="utf-8",
        )

    def _load_due_cards(self) -> list[tuple[Card, str]]:
        now = datetime.datetime.utcnow()
        due: list[tuple[Card, str]] = []
        for bundle_path in sorted(self._data_dir.glob("*.json")):
            try:
                data = json.loads(bundle_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            topic = bundle_path.stem
            for raw in data.get("cards", []):
                card = self._load_srs_state(Card(front=raw["front"], back=raw["back"]))
                if card.due <= now:
                    due.append((card, topic))
        return due

    def run(self) -> SessionRecord:
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        started_at = datetime.datetime.utcnow()
        due_cards = self._load_due_cards()
        ratings: list[CardRating] = []

        if not due_cards:
            print("No cards due for review.")
            return SessionRecord(
                session_id=str(uuid.uuid4()),
                started_at=started_at,
                ratings=[],
                duration_seconds=0.0,
            )

        print(f"Reviewing {len(due_cards)} due card(s). Rate each 0-5 (0=blackout, 5=perfect).\n")
        for card, topic in due_cards:
            print(f"Q: {card.front}")
            input("  [Press Enter to reveal answer]")
            print(f"A: {card.back}\n")
            while True:
                raw = input("Rating (0-5): ").strip()
                if raw.isdigit() and 0 <= int(raw) <= 5:
                    quality = int(raw)
                    break
                print("Please enter a number 0-5.")
            updated = schedule(card, quality)
            self._save_srs_state(updated)
            ratings.append(CardRating(card_front=card.front, topic=topic, quality=quality))
            print()

        duration = (datetime.datetime.utcnow() - started_at).total_seconds()
        record = SessionRecord(
            session_id=str(uuid.uuid4()),
            started_at=started_at,
            ratings=ratings,
            duration_seconds=duration,
        )
        ts = started_at.strftime("%Y%m%dT%H%M%S")
        session_path = self._sessions_dir / f"{ts}.json"
        session_path.write_text(
            json.dumps({
                "session_id": record.session_id,
                "started_at": record.started_at.isoformat(),
                "duration_seconds": record.duration_seconds,
                "ratings": [
                    {"card_front": r.card_front, "topic": r.topic, "quality": r.quality}
                    for r in record.ratings
                ],
            }, indent=2),
            encoding="utf-8",
        )
        print(f"Session saved. Reviewed {len(ratings)} card(s).")
        return record


def load_session_history(data_dir: pathlib.Path) -> list[SessionRecord]:
    sessions_dir = data_dir / "sessions"
    if not sessions_dir.exists():
        return []
    records: list[SessionRecord] = []
    for path in sorted(sessions_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        records.append(SessionRecord(
            session_id=data["session_id"],
            started_at=datetime.datetime.fromisoformat(data["started_at"]),
            duration_seconds=data["duration_seconds"],
            ratings=[
                CardRating(
                    card_front=r["card_front"],
                    topic=r["topic"],
                    quality=r["quality"],
                )
                for r in data.get("ratings", [])
            ],
        ))
    return records
