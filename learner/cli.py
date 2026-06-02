from __future__ import annotations

import argparse
import json
import pathlib
import sys

from learner.content import ContentProcessor
from learner.llm import ClaudeCliBackend

_DATA_DIR = pathlib.Path("data")


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

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _DATA_DIR / f"{source.stem}.json"
    out_path.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


def cmd_review(args: argparse.Namespace) -> None:
    raise NotImplementedError("review not yet implemented")


def cmd_stats(args: argparse.Namespace) -> None:
    raise NotImplementedError("stats not yet implemented")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="learner",
        description="AI-powered spaced-repetition learning accelerator.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    ingest = sub.add_parser("ingest", help="Ingest source material and generate cards/questions.")
    ingest.add_argument("source", help="Path or URL of the material to ingest.")
    ingest.set_defaults(func=cmd_ingest)

    review = sub.add_parser("review", help="Start an interactive review session.")
    review.set_defaults(func=cmd_review)

    stats = sub.add_parser("stats", help="Show study statistics and weak areas.")
    stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    args.func(args)
