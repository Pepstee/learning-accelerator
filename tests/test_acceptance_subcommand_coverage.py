"""
tests/test_acceptance_subcommand_coverage.py

Two independent test groups:

  1. acceptance.sh subprocess tests — verify the script exits 0 AND that the
     stdout contains the analytics section (a "Sessions:" line emitted by
     `learner analytics` after a seeded session record).

  2. Full subcommand coverage via --mock flag — every subcommand registered in
     build_parser(), including the previously-untested `questions`, must exit 0
     when invoked with the `--mock` CLI flag.  Tests are parametrised so each
     subcommand is an independent failure point.

The test-IDs are named so `pytest -k 'acceptance or subcommand'` collects all
of them.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_ACCEPTANCE_SCRIPT = _PROJECT_ROOT / "acceptance.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_acceptance(timeout: int = 90) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(_ACCEPTANCE_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=timeout,
    )


def _cli_mock(
    subcommand: str,
    extra_args: list[str],
    data_dir: pathlib.Path,
    *,
    timeout: int = 30,
    stdin: str | None = None,
) -> subprocess.CompletedProcess:
    """Invoke `python -m learner --mock <subcommand>` as a real subprocess."""
    cmd = [
        sys.executable, "-m", "learner",
        "--data-dir", str(data_dir),
        "--mock",
        subcommand,
    ] + extra_args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=timeout,
        input=stdin,
    )


# ---------------------------------------------------------------------------
# Fixture: populated data dir + source file
# ---------------------------------------------------------------------------

@pytest.fixture
def bundle_env(tmp_path: pathlib.Path) -> dict:
    """
    Provide:
      - topic.json  : full bundle (summary + 1 card + 1 question)
      - empty.json  : bundle with no questions (used by exam no-op path)
      - material.txt: source text for ingest / generate
    """
    source = tmp_path / "material.txt"
    source.write_text(
        "Osmosis is the diffusion of water across a semipermeable membrane "
        "from a region of low solute concentration to high solute concentration.",
        encoding="utf-8",
    )

    topic_bundle = {
        "summary": "Osmosis summary for testing.",
        "cards": [
            {
                "front": "What is osmosis?",
                "back": "Diffusion of water across a semipermeable membrane.",
            }
        ],
        "questions": [
            {
                "stem": "What drives osmosis?",
                "choices": ["Concentration gradient", "Temperature", "pH", "Light"],
                "answer_index": 0,
                "explanation": "Osmosis is driven by the water concentration gradient.",
            }
        ],
    }
    (tmp_path / "topic.json").write_text(json.dumps(topic_bundle, indent=2), encoding="utf-8")

    empty_bundle = {"summary": "Nothing here.", "cards": [], "questions": []}
    (tmp_path / "empty.json").write_text(json.dumps(empty_bundle, indent=2), encoding="utf-8")

    return {"data_dir": tmp_path, "source": source}


# ---------------------------------------------------------------------------
# 1. acceptance.sh tests
#    The class name contains "Acceptance" so every method is collected by
#    `pytest -k acceptance`.
# ---------------------------------------------------------------------------

class TestAcceptanceScript:

    def test_acceptance_script_exits_0(self):
        """acceptance.sh must complete with exit code 0."""
        result = _run_acceptance()
        assert result.returncode == 0, (
            f"acceptance.sh exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    def test_acceptance_script_analytics_section_present(self):
        """
        acceptance.sh must include a 'Sessions:' line in its stdout.

        The script seeds a session record before running `learner analytics`,
        so cmd_analytics() will find history and print the Sessions: summary
        line.  If this line is absent the analytics subcommand did not run or
        did not produce output.
        """
        result = _run_acceptance()
        assert "Sessions:" in result.stdout, (
            "No 'Sessions:' line found in acceptance.sh output. "
            "The analytics subcommand may have failed or not been called.\n"
            f"Full stdout:\n{result.stdout}"
        )

    def test_acceptance_script_ok_marker_present(self):
        """acceptance.sh must print 'acceptance: OK' to signal a complete run."""
        result = _run_acceptance()
        assert "acceptance: OK" in result.stdout, (
            f"'acceptance: OK' not found.\nstdout:\n{result.stdout}"
        )

    def test_acceptance_script_ingest_summary_line(self):
        """The ingest section inside acceptance.sh must emit a 'Summary:' line."""
        result = _run_acceptance()
        assert "Summary:" in result.stdout

    def test_acceptance_script_questions_q_line(self):
        """The questions section inside acceptance.sh must emit at least one 'Q:' line."""
        result = _run_acceptance()
        assert "Q:" in result.stdout, (
            "No 'Q:' found in acceptance.sh output; "
            "the questions subcommand may have failed.\n"
            f"stdout:\n{result.stdout}"
        )

    def test_acceptance_script_flashcards_run(self):
        """acceptance.sh must exercise the flashcards subcommand."""
        content = _ACCEPTANCE_SCRIPT.read_text(encoding="utf-8")
        assert "flashcards" in content

    def test_acceptance_script_analytics_run(self):
        """acceptance.sh must exercise the analytics subcommand."""
        content = _ACCEPTANCE_SCRIPT.read_text(encoding="utf-8")
        assert "analytics" in content

    def test_acceptance_script_uses_mock_env(self):
        """acceptance.sh must set LEARNER_MOCK so no real LLM calls are made."""
        content = _ACCEPTANCE_SCRIPT.read_text(encoding="utf-8")
        assert "LEARNER_MOCK" in content, (
            "acceptance.sh does not set LEARNER_MOCK=1; it would attempt real LLM calls"
        )

    def test_acceptance_script_stderr_empty_on_clean_run(self):
        """A clean run must not write anything to stderr."""
        result = _run_acceptance()
        if result.returncode == 0:
            assert not result.stderr.strip(), (
                f"acceptance.sh wrote to stderr on a clean run:\n{result.stderr}"
            )


# ---------------------------------------------------------------------------
# 2. Parametrised subcommand tests — every subcommand, --mock flag
#    Test IDs contain "subcommand" so `pytest -k subcommand` collects all.
# ---------------------------------------------------------------------------

_ALL_SUBCOMMANDS = [
    "ingest",
    "generate",
    "questions",    # was missing from prior coverage; requires topic arg
    "summary",
    "flashcards",
    "practice",     # empty dir → "No cards due" → immediate exit
    "review",       # empty dir → "Nothing due" → immediate exit
    "exam",         # no-questions bundle → "No questions found" → exit 0
    "study-plan",
    "analytics",
]

_INTERACTIVE_SUBCOMMANDS = {"practice", "review"}


def _extra_args(subcommand: str, env: dict) -> list[str]:
    """Map each subcommand to the positional arguments it requires."""
    if subcommand in ("ingest", "generate"):
        return [str(env["source"])]
    if subcommand in ("summary", "flashcards", "questions"):
        return ["topic"]
    if subcommand == "exam":
        # empty bundle → prints "No questions found" and exits 0 without blocking stdin
        return ["empty"]
    return []


def _data_dir(subcommand: str, env: dict) -> pathlib.Path:
    """Interactive subcommands get an empty dir so they return immediately."""
    if subcommand in _INTERACTIVE_SUBCOMMANDS:
        empty = env["data_dir"] / "_empty"
        empty.mkdir(exist_ok=True)
        return empty
    return env["data_dir"]


@pytest.mark.parametrize("subcommand", _ALL_SUBCOMMANDS)
def test_subcommand_exits_0_with_mock_flag(subcommand: str, bundle_env: dict) -> None:
    """Every named subcommand must exit 0 when invoked with --mock."""
    result = _cli_mock(
        subcommand,
        _extra_args(subcommand, bundle_env),
        _data_dir(subcommand, bundle_env),
    )
    assert result.returncode == 0, (
        f"'{subcommand}' --mock exited {result.returncode}\n"
        f"args: {_extra_args(subcommand, bundle_env)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("subcommand", _ALL_SUBCOMMANDS)
def test_subcommand_no_traceback_with_mock_flag(subcommand: str, bundle_env: dict) -> None:
    """A clean --mock run must not produce a Python traceback on stderr."""
    result = _cli_mock(
        subcommand,
        _extra_args(subcommand, bundle_env),
        _data_dir(subcommand, bundle_env),
    )
    assert "Traceback (most recent call last)" not in result.stderr, (
        f"'{subcommand}' produced a traceback:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# 3. Output contracts for ingest, questions, flashcards
#    (the three subcommands explicitly named in the acceptance criteria)
# ---------------------------------------------------------------------------

def test_ingest_subcommand_output_with_mock(bundle_env: dict) -> None:
    """ingest --mock must print Summary:, Cards:, and Questions: lines."""
    result = _cli_mock("ingest", [str(bundle_env["source"])], bundle_env["data_dir"])
    assert result.returncode == 0
    assert "Summary:" in result.stdout
    assert "Cards:" in result.stdout
    assert "Questions:" in result.stdout


def test_ingest_subcommand_writes_bundle_file(bundle_env: dict) -> None:
    """ingest --mock must create a .json bundle in the data dir."""
    source = bundle_env["source"]
    data_dir = bundle_env["data_dir"]
    result = _cli_mock("ingest", [str(source)], data_dir)
    assert result.returncode == 0
    bundle_path = data_dir / f"{source.stem}.json"
    assert bundle_path.exists(), f"Expected bundle at {bundle_path} but it was not created"
    parsed = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert "summary" in parsed
    assert "cards" in parsed
    assert "questions" in parsed


def test_ingest_subcommand_cards_count_at_least_1(bundle_env: dict) -> None:
    """The mock backend returns at least 1 card; ingest must reflect that."""
    result = _cli_mock("ingest", [str(bundle_env["source"])], bundle_env["data_dir"])
    assert result.returncode == 0
    # Line looks like: "Cards:     1"
    cards_line = next(
        (ln for ln in result.stdout.splitlines() if ln.startswith("Cards:")), None
    )
    assert cards_line is not None, "No 'Cards:' line in ingest output"
    count_str = cards_line.split()[-1]
    assert count_str.isdigit() and int(count_str) >= 1, (
        f"Expected Cards >= 1, got: {cards_line!r}"
    )


def test_ingest_subcommand_missing_source_exits_nonzero(tmp_path: pathlib.Path) -> None:
    """ingest with a nonexistent source file must exit non-zero and write to stderr."""
    result = _cli_mock("ingest", [str(tmp_path / "ghost.txt")], tmp_path)
    assert result.returncode != 0
    assert "error" in result.stderr.lower()


def test_questions_subcommand_prints_q_line(bundle_env: dict) -> None:
    """questions --mock must emit at least one '[N] Q:' line for the preloaded topic."""
    result = _cli_mock("questions", ["topic"], bundle_env["data_dir"])
    assert result.returncode == 0
    assert "Q:" in result.stdout, (
        f"'questions' subcommand produced no Q: line.\nstdout:\n{result.stdout}"
    )


def test_questions_subcommand_prints_answer_line(bundle_env: dict) -> None:
    """questions output must include an Answer: line so the user can check their answer."""
    result = _cli_mock("questions", ["topic"], bundle_env["data_dir"])
    assert result.returncode == 0
    assert "Answer:" in result.stdout, (
        f"'questions' subcommand produced no Answer: line.\nstdout:\n{result.stdout}"
    )


def test_questions_subcommand_empty_bundle_exits_0(tmp_path: pathlib.Path) -> None:
    """questions with a bundle that has no questions must exit 0 and say so."""
    empty = {"summary": "Nothing.", "cards": [], "questions": []}
    (tmp_path / "empty.json").write_text(json.dumps(empty), encoding="utf-8")
    result = _cli_mock("questions", ["empty"], tmp_path)
    assert result.returncode == 0
    assert "No questions" in result.stdout


def test_questions_subcommand_missing_topic_exits_nonzero(tmp_path: pathlib.Path) -> None:
    """questions with an unknown topic name must exit non-zero."""
    result = _cli_mock("questions", ["no_such_topic"], tmp_path)
    assert result.returncode != 0


def test_flashcards_subcommand_prints_q_and_a_lines(bundle_env: dict) -> None:
    """flashcards --mock must emit both 'Q:' and 'A:' lines for the preloaded topic."""
    result = _cli_mock("flashcards", ["topic"], bundle_env["data_dir"])
    assert result.returncode == 0
    assert "Q:" in result.stdout, (
        f"'flashcards' subcommand produced no Q: line.\nstdout:\n{result.stdout}"
    )
    assert "A:" in result.stdout, (
        f"'flashcards' subcommand produced no A: line.\nstdout:\n{result.stdout}"
    )


def test_flashcards_subcommand_empty_bundle_exits_0(tmp_path: pathlib.Path) -> None:
    """flashcards with a bundle that has no cards must exit 0 and say so."""
    empty = {"summary": "Empty.", "cards": [], "questions": []}
    (tmp_path / "empty.json").write_text(json.dumps(empty), encoding="utf-8")
    result = _cli_mock("flashcards", ["empty"], tmp_path)
    assert result.returncode == 0
    assert "No flashcards" in result.stdout


def test_flashcards_subcommand_missing_topic_exits_nonzero(tmp_path: pathlib.Path) -> None:
    """flashcards with an unknown topic must exit non-zero."""
    result = _cli_mock("flashcards", ["no_such_topic"], tmp_path)
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# 4. Analytics with real session history (tests the analytics code path that
#    produces the "Sessions:" line the acceptance script checks for)
# ---------------------------------------------------------------------------

def test_analytics_subcommand_with_seeded_history(tmp_path: pathlib.Path) -> None:
    """analytics --mock with a seeded session record must print a 'Sessions:' line."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    session_record = {
        "session_id": "unit-test-session",
        "started_at": "2026-06-12T00:00:00",
        "duration_seconds": 60.0,
        "ratings": [
            {"card_front": "What is osmosis?", "topic": "topic", "quality": 3},
        ],
    }
    (sessions_dir / "20260612T000000.json").write_text(
        json.dumps(session_record), encoding="utf-8"
    )
    result = _cli_mock("analytics", [], tmp_path)
    assert result.returncode == 0
    assert "Sessions:" in result.stdout, (
        f"Expected 'Sessions:' in analytics output.\nstdout:\n{result.stdout}"
    )


def test_analytics_subcommand_no_history_exits_0(tmp_path: pathlib.Path) -> None:
    """analytics with no session history must exit 0 and explain the situation."""
    result = _cli_mock("analytics", [], tmp_path)
    assert result.returncode == 0
    assert "No session history" in result.stdout
