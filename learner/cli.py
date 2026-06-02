from __future__ import annotations

import argparse
import sys


def cmd_ingest(args: argparse.Namespace) -> None:
    raise NotImplementedError("ingest not yet implemented")


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
