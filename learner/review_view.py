from __future__ import annotations

import datetime
import json
import sys
import uuid
from typing import TYPE_CHECKING

from learner.models import Card, CardRating, Question, SessionRecord
from learner.srs import schedule

if TYPE_CHECKING:
    from learner.session import ReviewSession


def _getch() -> str:
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch
    except Exception:
        return input()


def _wait_keypress() -> None:
    sys.stdout.write("  [press any key to reveal] ")
    sys.stdout.flush()
    _getch()
    print()


def _prompt_rating() -> int:
    while True:
        raw = input("Rating 1-5 (1=forgot, 5=perfect): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        print("  Enter a number from 1 to 5.")


def _display_question(idx: int, total: int, q: Question) -> None:
    print(f"[{idx}/{total}] {q.stem}")
    for i, choice in enumerate(q.choices):
        print(f"  {chr(65 + i)}) {choice}")


def _prompt_answer(q: Question) -> bool:
    valid = {chr(65 + i) for i in range(len(q.choices))}
    while True:
        raw = input(f"Answer ({'/'.join(sorted(valid))}): ").strip().upper()
        if raw in valid:
            return ord(raw) - 65 == q.answer_index
        print(f"  Enter one of: {', '.join(sorted(valid))}")


def _load_questions(session: ReviewSession) -> list[tuple[Question, str]]:
    result: list[tuple[Question, str]] = []
    for bundle_path in sorted(session._data_dir.glob("*.json")):
        try:
            data = json.loads(bundle_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        topic = bundle_path.stem
        for raw in data.get("questions", []):
            try:
                result.append((
                    Question(
                        stem=raw["stem"],
                        choices=raw["choices"],
                        answer_index=raw["answer_index"],
                        explanation=raw["explanation"],
                    ),
                    topic,
                ))
            except KeyError:
                continue
    return result


def _save_session_record(session: ReviewSession, record: SessionRecord) -> None:
    session._sessions_dir.mkdir(parents=True, exist_ok=True)
    ts = record.started_at.strftime("%Y%m%dT%H%M%S")
    path = session._sessions_dir / f"{ts}.json"
    path.write_text(
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


class ReviewView:
    def run(self, session: ReviewSession) -> SessionRecord:
        started_at = datetime.datetime.utcnow()
        due_cards = session._load_due_cards()
        questions = _load_questions(session)
        ratings: list[CardRating] = []
        updated_cards: list[Card] = []

        if not due_cards and not questions:
            print("Nothing due for review.")
            return SessionRecord(
                session_id=str(uuid.uuid4()),
                started_at=started_at,
                ratings=[],
                duration_seconds=0.0,
            )

        if due_cards:
            total = len(due_cards)
            print(f"\nFlashcards: {total} due. Rate each 1-5.\n")
            for idx, (card, topic) in enumerate(due_cards, 1):
                print(f"[{idx}/{total}] {card.front}")
                _wait_keypress()
                print(f"  {card.back}\n")
                quality = _prompt_rating()
                updated_cards.append(schedule(card, quality))
                ratings.append(CardRating(card_front=card.front, topic=topic, quality=quality))
                print()

        if questions:
            total = len(questions)
            print(f"\nPractice questions: {total}.\n")
            for idx, (q, topic) in enumerate(questions, 1):
                _display_question(idx, total, q)
                correct = _prompt_answer(q)
                label = "Correct!" if correct else f"Wrong — answer: {chr(65 + q.answer_index)}"
                print(f"  {label}")
                print(f"  {q.explanation}\n")
                ratings.append(CardRating(card_front=q.stem, topic=topic, quality=5 if correct else 2))
                print()

        for updated in updated_cards:
            session._save_srs_state(updated)

        duration = (datetime.datetime.utcnow() - started_at).total_seconds()
        record = SessionRecord(
            session_id=str(uuid.uuid4()),
            started_at=started_at,
            ratings=ratings,
            duration_seconds=duration,
        )
        _save_session_record(session, record)
        print(f"Session complete. {len(updated_cards)} card(s), {len(questions)} question(s) reviewed.")
        return record
