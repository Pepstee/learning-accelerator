from __future__ import annotations

import argparse
import json
import pathlib
import sys

from learner.analytics import compute_weak_areas, generate_study_plan
from learner.content import ContentProcessor
from learner.generator import generate_flashcards, generate_questions, generate_summary
from learner.ingest import ingest_text
from learner.llm import ClaudeCliBackend, MockBackend
from learner.session import ReviewSession, load_session_history

_DEFAULT_DATA_DIR = pathlib.Path.home() / ".learner"


def _backend(args: argparse.Namespace):
    import os
    if args.mock or os.environ.get("LEARNER_MOCK"):
        return MockBackend()
    return ClaudeCliBackend()


def _load_bundle(data_dir: pathlib.Path, topic: str) -> dict:
    path = data_dir / f"{topic}.json"
    if not path.exists():
        print(f"error: no bundle for '{topic}' — run 'learner ingest' first", file=sys.stderr)
        sys.exit(1)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read bundle: {exc}", file=sys.stderr)
        sys.exit(1)


def _generation_chunks(text: str, source: pathlib.Path) -> list[str]:
    chunks = ingest_text(text, source=str(source))
    if not chunks:
        return [text]
    return [chunk.metadata["generation_text"] for chunk in chunks]


def cmd_ingest(args: argparse.Namespace) -> None:
    source = pathlib.Path(args.source)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot read {source}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        bundle = ContentProcessor(_backend(args)).process(text)
    except Exception as exc:
        print(f"error: processing failed: {exc}", file=sys.stderr)
        sys.exit(1)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.data_dir / f"{source.stem}.json"
    out_path.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")

    print(f"Summary:   {bundle.summary}")
    print(f"Cards:     {len(bundle.cards)}")
    print(f"Questions: {len(bundle.questions)}")
    print(f"Saved:     {out_path}")


def cmd_summary(args: argparse.Namespace) -> None:
    data = _load_bundle(args.data_dir, args.topic)
    print(data.get("summary", "(no summary available)"))


def cmd_flashcards(args: argparse.Namespace) -> None:
    data = _load_bundle(args.data_dir, args.topic)
    cards = data.get("cards", [])
    if not cards:
        print("No flashcards found for this topic.")
        return
    for i, card in enumerate(cards, 1):
        print(f"[{i}] Q: {card['front']}")
        print(f"     A: {card['back']}")
        print()


def cmd_questions(args: argparse.Namespace) -> None:
    data = _load_bundle(args.data_dir, args.topic)
    questions = data.get("questions", [])
    if not questions:
        print("No questions found for this topic.")
        return
    for i, q in enumerate(questions, 1):
        print(f"[{i}] Q: {q['stem']}")
        for j, choice in enumerate(q["choices"]):
            print(f"     {j + 1}. {choice}")
        print(f"     Answer: {q['choices'][q['answer_index']]}")
        print()


def cmd_practice(args: argparse.Namespace) -> None:
    from learner.review_view import ReviewView
    session = ReviewSession(data_dir=args.data_dir)
    ReviewView().run(session)


def cmd_exam(args: argparse.Namespace) -> None:
    data = _load_bundle(args.data_dir, args.topic)
    questions = data.get("questions", [])
    if not questions:
        print("No questions found for this topic.")
        return

    score = 0
    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}: {q['stem']}")
        for j, choice in enumerate(q["choices"]):
            print(f"  {j + 1}. {choice}")
        while True:
            raw = input("Your answer (1-4): ").strip()
            if raw.isdigit() and 1 <= int(raw) <= 4:
                break
            print("Please enter a number 1-4.")
        chosen = int(raw) - 1
        correct = q["answer_index"]
        if chosen == correct:
            print("Correct!")
            score += 1
        else:
            print(f"Wrong. Correct answer: {q['choices'][correct]}")
        print(f"Explanation: {q['explanation']}")

    print(f"\nScore: {score}/{len(questions)}")


def cmd_generate(args: argparse.Namespace) -> None:
    source = pathlib.Path(args.source)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot read {source}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        backend = _backend(args)
        chunks = _generation_chunks(text, source)
        summary = generate_summary(chunks, backend)
        cards = generate_flashcards(chunks, backend)
        questions = generate_questions(chunks, backend)
    except Exception as exc:
        print(f"error: generation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.data_dir / f"{source.stem}.json"
    bundle = {
        "summary": summary,
        "cards": [{"front": c.front, "back": c.back} for c in cards],
        "questions": [
            {
                "stem": q.stem,
                "choices": q.choices,
                "answer_index": q.answer_index,
                "explanation": q.explanation,
            }
            for q in questions
        ],
    }
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    print(f"Summary:   {summary}")
    print(f"Cards:     {len(cards)}")
    print(f"Questions: {len(questions)}")
    print(f"Saved:     {out_path}")


def cmd_review(args: argparse.Namespace) -> None:
    from learner.review_view import ReviewView
    session = ReviewSession(data_dir=args.data_dir)
    ReviewView().run(session)


def cmd_study_plan(args: argparse.Namespace) -> None:
    history = load_session_history(args.data_dir)
    weak_areas = compute_weak_areas(history)
    if not weak_areas:
        print("No weak areas identified. Keep up the great work!")
        return
    plan = generate_study_plan(weak_areas)
    print(plan.advice)


def cmd_analytics(args: argparse.Namespace) -> None:
    history = load_session_history(args.data_dir)
    if not history:
        print("No session history found. Complete some practice sessions first.")
        return
    weak_areas = compute_weak_areas(history)
    total_reviews = sum(w.question_count for w in weak_areas)
    print(f"Sessions:     {len(history)}")
    print(f"Total reviews: {total_reviews}")
    if not weak_areas:
        print("No weak areas identified yet.")
        return
    print("\nWeak areas (by error rate):")
    for w in weak_areas:
        print(f"  {w.topic}: {w.error_rate:.0%} error rate ({w.question_count} review(s))")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="learner",
        description="AI-powered spaced-repetition learning accelerator.",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=_DEFAULT_DATA_DIR,
        metavar="DIR",
        help=f"Data directory (default: {_DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM backend for offline testing.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("ingest", help="Ingest source material and generate cards/questions.")
    p.add_argument("source", help="Path to the source file to ingest.")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("generate", help="Generate flashcards and questions from source material.")
    p.add_argument("source", help="Path to the source file.")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("questions", help="List all questions for a topic.")
    p.add_argument("topic", help="Topic name.")
    p.set_defaults(func=cmd_questions)

    p = sub.add_parser("summary", help="Print the summary for an ingested topic.")
    p.add_argument("topic", help="Topic name (file stem, e.g. 'notes' for notes.json).")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("flashcards", help="List all flashcards for a topic.")
    p.add_argument("topic", help="Topic name.")
    p.set_defaults(func=cmd_flashcards)

    p = sub.add_parser("practice", help="Interactive SRS review session.")
    p.set_defaults(func=cmd_practice)

    p = sub.add_parser("review", help="Interactive flashcard and question review session.")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("exam", help="Multiple-choice quiz for a topic.")
    p.add_argument("topic", help="Topic name.")
    p.set_defaults(func=cmd_exam)

    p = sub.add_parser("study-plan", help="Generate an AI study plan based on weak areas.")
    p.set_defaults(func=cmd_study_plan)

    p = sub.add_parser("analytics", help="Show study analytics and weak areas.")
    p.set_defaults(func=cmd_analytics)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
