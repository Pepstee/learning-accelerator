from __future__ import annotations

import argparse
import json
import pathlib
import sys

from learner.analytics import compute_weak_areas, generate_study_plan
from learner.content import ContentProcessor
from learner.llm import ClaudeCliBackend
from learner.session import ReviewSession, load_session_history

_DEFAULT_DATA_DIR = pathlib.Path.home() / ".learner"


def cmd_ingest(args: argparse.Namespace) -> None:
    source = pathlib.Path(args.source)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot read {source}: {exc}", file=sys.stderr)
        sys.exit(1)

    processor = ContentProcessor(ClaudeCliBackend())
    bundle = processor.process(text)

    print(f"Summary: {bundle.summary}")
    print(f"Cards:     {len(bundle.cards)}")
    print(f"Questions: {len(bundle.questions)}")

    data_dir: pathlib.Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / f"{source.stem}.json"
    out_path.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


def cmd_review(args: argparse.Namespace) -> None:
    session = ReviewSession(data_dir=args.data_dir)
    session.run()


def cmd_stats(args: argparse.Namespace) -> None:
    history = load_session_history(args.data_dir)
    if not history:
        print("No session history found. Run 'learner review' first.")
        return

    weak_areas = compute_weak_areas(history)
    if not weak_areas:
        print("No weak areas identified yet.")
        return

    print("Weak areas (sorted by error rate):")
    for w in weak_areas:
        print(f"  {w.topic}: {w.error_rate:.0%} error rate ({w.question_count} card(s))")

    if args.plan:
        print("\nGenerating study plan...\n")
        plan = generate_study_plan(weak_areas, ClaudeCliBackend())
        print(plan)


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
        help=f"Directory for cards, SRS state, and sessions (default: {_DEFAULT_DATA_DIR})",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    ingest = sub.add_parser("ingest", help="Ingest source material and generate cards/questions.")
    ingest.add_argument("source", help="Path to the material to ingest.")
    ingest.set_defaults(func=cmd_ingest)

    review = sub.add_parser("review", help="Start an interactive review session.")
    review.set_defaults(func=cmd_review)

    stats = sub.add_parser("stats", help="Show study statistics and weak areas.")
    stats.add_argument(
        "--plan",
        action="store_true",
        help="Generate an AI study plan for the identified weak areas.",
    )
    stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    args.func(args)
