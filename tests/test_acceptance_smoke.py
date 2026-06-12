"""
tests/test_acceptance_smoke.py

Smoke tests that verify three things independently:

  1. acceptance.sh completes with exit 0 and produces non-empty stdout.
  2. Every CLI subcommand (all 9 of them) exits cleanly when driven with
     LEARNER_MOCK=1 via subprocess — covering the full argparse dispatch,
     env-var handling, and sys.exit() paths that in-process main() calls
     cannot exercise.
  3. No assessment-automation symbols appear anywhere in the learner package
     source — ensuring review interactions remain genuinely user-driven.

These tests are independent of one another and can fail independently.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_ACCEPTANCE_SCRIPT = _PROJECT_ROOT / "acceptance.sh"
_LEARNER_PKG = _PROJECT_ROOT / "learner"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _run_acceptance(timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(_ACCEPTANCE_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=timeout,
    )


def _learner_sources() -> str:
    """Concatenated text of every .py file in the learner package."""
    return "\n".join(
        f.read_text(encoding="utf-8") for f in sorted(_LEARNER_PKG.glob("*.py"))
    )


@pytest.fixture
def test_env(tmp_path: pathlib.Path) -> dict:
    """
    A self-contained env dict with a data_dir and source file.

    data_dir contains:
      - topic.json   : full bundle (summary + cards + questions) for
                       summary / flashcards subcommands
      - empty.json   : bundle with no questions, for the exam subcommand
                       (exam exits 0 when there are no questions)
      - material.txt : source file for ingest / generate subcommands
    """
    source = tmp_path / "material.txt"
    source.write_text(
        "Photosynthesis converts sunlight into chemical energy stored as glucose.",
        encoding="utf-8",
    )

    topic_bundle = {
        "summary": "Photosynthesis topic summary.",
        "cards": [
            {"front": "What is ATP?", "back": "Adenosine triphosphate, an energy carrier."},
        ],
        "questions": [
            {
                "stem": "What does chlorophyll absorb?",
                "choices": ["Light", "Water", "Glucose", "Oxygen"],
                "answer_index": 0,
                "explanation": "Chlorophyll absorbs light energy.",
            }
        ],
    }
    (tmp_path / "topic.json").write_text(json.dumps(topic_bundle, indent=2), encoding="utf-8")

    empty_bundle = {"summary": "Empty topic.", "cards": [], "questions": []}
    (tmp_path / "empty.json").write_text(json.dumps(empty_bundle, indent=2), encoding="utf-8")

    return {"data_dir": tmp_path, "source": source}


def _cli_run(
    subcommand: str,
    extra_args: list[str],
    data_dir: pathlib.Path,
    *,
    timeout: int = 30,
    stdin: str | None = None,
) -> subprocess.CompletedProcess:
    """Run 'python -m learner' with LEARNER_MOCK=1 as a real subprocess."""
    env = {**os.environ, "LEARNER_MOCK": "1"}
    cmd = [sys.executable, "-m", "learner", "--data-dir", str(data_dir), subcommand] + extra_args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        env=env,
        timeout=timeout,
        input=stdin,
    )


# ---------------------------------------------------------------------------
# 1. acceptance.sh smoke tests
# ---------------------------------------------------------------------------

class TestAcceptanceScript:

    def test_script_exists(self):
        assert _ACCEPTANCE_SCRIPT.exists(), "acceptance.sh not found at project root"

    def test_script_is_executable(self):
        assert os.access(_ACCEPTANCE_SCRIPT, os.X_OK), (
            "acceptance.sh is not executable — run: chmod +x acceptance.sh"
        )

    def test_exit_0(self):
        result = _run_acceptance()
        assert result.returncode == 0, (
            f"acceptance.sh exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    def test_stdout_non_empty(self):
        result = _run_acceptance()
        assert result.stdout.strip(), (
            "acceptance.sh produced no stdout — it must print at least the summary"
        )

    def test_ok_marker_present(self):
        """The script must print 'acceptance: OK' to signal a complete run."""
        result = _run_acceptance()
        assert "acceptance: OK" in result.stdout, (
            f"'acceptance: OK' not found in stdout:\n{result.stdout}"
        )

    def test_summary_line_printed(self):
        """Ingest output must include a 'Summary:' line."""
        result = _run_acceptance()
        assert "Summary:" in result.stdout

    def test_cards_line_printed(self):
        """Ingest output must include a 'Cards:' line with a count."""
        result = _run_acceptance()
        assert "Cards:" in result.stdout

    def test_questions_line_printed(self):
        """Ingest output must include a 'Questions:' line with a count."""
        result = _run_acceptance()
        assert "Questions:" in result.stdout

    def test_flashcards_qa_pairs_in_output(self):
        """acceptance.sh runs 'flashcards' and checks for Q: pairs in stdout."""
        result = _run_acceptance()
        assert "Q:" in result.stdout, (
            "No Q: flashcard output found — did the flashcards subcommand run?"
        )

    def test_stderr_empty_on_success(self):
        """A clean run should produce nothing on stderr."""
        result = _run_acceptance()
        if result.returncode == 0:
            assert not result.stderr.strip(), (
                f"acceptance.sh wrote to stderr even though it succeeded:\n{result.stderr}"
            )

    def test_learner_mock_env_used(self):
        """Script must set LEARNER_MOCK so no real LLM calls are made."""
        content = _ACCEPTANCE_SCRIPT.read_text(encoding="utf-8")
        assert "LEARNER_MOCK" in content, (
            "acceptance.sh does not set LEARNER_MOCK=1; it would attempt real LLM calls"
        )

    def test_ingest_and_flashcards_both_run(self):
        """The script exercises both ingest and flashcards subcommands."""
        content = _ACCEPTANCE_SCRIPT.read_text(encoding="utf-8")
        assert "ingest" in content
        assert "flashcards" in content


# ---------------------------------------------------------------------------
# 2. Parametrised CLI subcommand tests
#
#    Every subcommand that build_parser() registers must exit 0 when given
#    valid (mock) inputs.  Interactive subcommands (practice, review) are
#    exercised with an empty data-dir so they return immediately; exam is
#    exercised with a no-questions bundle so it also returns immediately.
# ---------------------------------------------------------------------------

def _extra_args(subcommand: str, env: dict) -> list[str]:
    """Return the positional args a subcommand needs."""
    if subcommand in ("ingest", "generate"):
        return [str(env["source"])]
    if subcommand in ("summary", "flashcards"):
        return ["topic"]
    if subcommand == "exam":
        # bundle with no questions → prints "No questions found" and exits 0
        return ["empty"]
    return []


_ALL_SUBCOMMANDS = [
    "ingest",
    "generate",
    "summary",
    "flashcards",
    "practice",   # empty dir → "Nothing due" → immediate exit
    "review",     # empty dir → "Nothing due" → immediate exit
    "exam",       # no-questions bundle → "No questions found" → exit 0
    "study-plan",
    "analytics",
]


# Subcommands that are interactive require an EMPTY data dir so they exit
# immediately (no due cards, no questions) without blocking on stdin.
_INTERACTIVE_SUBCOMMANDS = {"practice", "review"}


def _data_dir_for(subcommand: str, test_env: dict) -> pathlib.Path:
    """Return the data_dir to use — empty subdir for interactive subcommands."""
    if subcommand in _INTERACTIVE_SUBCOMMANDS:
        empty = test_env["data_dir"] / "_empty"
        empty.mkdir(exist_ok=True)
        return empty
    return test_env["data_dir"]


@pytest.mark.parametrize("subcommand", _ALL_SUBCOMMANDS)
def test_subcommand_exits_cleanly(subcommand: str, test_env: dict) -> None:
    """Each subcommand must complete with exit code 0 when using MockLLM."""
    extra = _extra_args(subcommand, test_env)
    data_dir = _data_dir_for(subcommand, test_env)
    result = _cli_run(subcommand, extra, data_dir)
    assert result.returncode == 0, (
        f"'{subcommand}' exited {result.returncode}\n"
        f"args: {extra}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("subcommand", _ALL_SUBCOMMANDS)
def test_subcommand_produces_no_unhandled_traceback(subcommand: str, test_env: dict) -> None:
    """A clean run must not produce a Python traceback on stderr."""
    extra = _extra_args(subcommand, test_env)
    data_dir = _data_dir_for(subcommand, test_env)
    result = _cli_run(subcommand, extra, data_dir)
    assert "Traceback (most recent call last)" not in result.stderr, (
        f"'{subcommand}' produced a traceback:\n{result.stderr}"
    )


# Additional per-subcommand output contracts

def test_ingest_stdout_has_summary_cards_questions(test_env: dict) -> None:
    result = _cli_run("ingest", [str(test_env["source"])], test_env["data_dir"])
    assert result.returncode == 0
    assert "Summary:" in result.stdout
    assert "Cards:" in result.stdout
    assert "Questions:" in result.stdout


def test_ingest_creates_json_bundle(test_env: dict) -> None:
    source = test_env["source"]
    data_dir = test_env["data_dir"]
    result = _cli_run("ingest", [str(source)], data_dir)
    assert result.returncode == 0
    expected_bundle = data_dir / f"{source.stem}.json"
    assert expected_bundle.exists(), f"Expected {expected_bundle} to be created by ingest"
    parsed = json.loads(expected_bundle.read_text(encoding="utf-8"))
    assert "summary" in parsed
    assert "cards" in parsed
    assert "questions" in parsed


def test_generate_creates_json_bundle(test_env: dict) -> None:
    source = test_env["source"]
    data_dir = test_env["data_dir"]
    result = _cli_run("generate", [str(source)], data_dir)
    assert result.returncode == 0
    expected_bundle = data_dir / f"{source.stem}.json"
    assert expected_bundle.exists()
    parsed = json.loads(expected_bundle.read_text(encoding="utf-8"))
    assert "summary" in parsed


def test_summary_prints_bundle_text(test_env: dict) -> None:
    result = _cli_run("summary", ["topic"], test_env["data_dir"])
    assert result.returncode == 0
    assert "Photosynthesis topic summary." in result.stdout


def test_flashcards_prints_q_and_a_lines(test_env: dict) -> None:
    result = _cli_run("flashcards", ["topic"], test_env["data_dir"])
    assert result.returncode == 0
    # cmd_flashcards prints "[N] Q: ..." and "     A: ..."
    assert "Q:" in result.stdout
    assert "A:" in result.stdout


def test_practice_with_empty_dir_says_nothing_due(tmp_path: pathlib.Path) -> None:
    # Fresh empty dir: no due cards, no questions → immediate exit
    result = _cli_run("practice", [], tmp_path)
    assert result.returncode == 0
    assert "Nothing due" in result.stdout


def test_review_with_empty_dir_says_nothing_due(tmp_path: pathlib.Path) -> None:
    result = _cli_run("review", [], tmp_path)
    assert result.returncode == 0
    assert "Nothing due" in result.stdout


def test_exam_no_questions_says_no_questions_found(test_env: dict) -> None:
    result = _cli_run("exam", ["empty"], test_env["data_dir"])
    assert result.returncode == 0
    assert "No questions" in result.stdout


def test_analytics_no_history_says_no_session_history(tmp_path: pathlib.Path) -> None:
    result = _cli_run("analytics", [], tmp_path)
    assert result.returncode == 0
    assert "No session history" in result.stdout


def test_study_plan_no_history_says_no_weak_areas(tmp_path: pathlib.Path) -> None:
    result = _cli_run("study-plan", [], tmp_path)
    assert result.returncode == 0
    assert "No weak areas" in result.stdout


# Error-path subcommand tests

def test_ingest_missing_file_exits_1(tmp_path: pathlib.Path) -> None:
    result = _cli_run("ingest", [str(tmp_path / "ghost.txt")], tmp_path)
    assert result.returncode == 1
    assert "error" in result.stderr.lower()


def test_generate_missing_file_exits_1(tmp_path: pathlib.Path) -> None:
    result = _cli_run("generate", [str(tmp_path / "ghost.txt")], tmp_path)
    assert result.returncode == 1
    assert "error" in result.stderr.lower()


def test_summary_missing_topic_exits_1(tmp_path: pathlib.Path) -> None:
    result = _cli_run("summary", ["nonexistent"], tmp_path)
    assert result.returncode == 1
    assert "nonexistent" in result.stderr or "error" in result.stderr.lower()


def test_flashcards_missing_topic_exits_1(tmp_path: pathlib.Path) -> None:
    result = _cli_run("flashcards", ["nonexistent"], tmp_path)
    assert result.returncode == 1


def test_exam_missing_topic_exits_1(tmp_path: pathlib.Path) -> None:
    result = _cli_run("exam", ["nonexistent"], tmp_path)
    assert result.returncode == 1


def test_no_subcommand_exits_0(tmp_path: pathlib.Path) -> None:
    """Invoking the CLI with no subcommand must print help and exit 0."""
    env = {**os.environ, "LEARNER_MOCK": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "learner", "--data-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        env=env,
        timeout=10,
    )
    assert result.returncode == 0
    assert result.stdout.strip(), "No-subcommand invocation should print help"


# ---------------------------------------------------------------------------
# 3. No assessment-automation symbols
#
#    The learner package must contain no symbols that would bypass genuine
#    user interaction during review sessions.
# ---------------------------------------------------------------------------

class TestNoAssessmentAutomationSymbols:

    def test_no_autograde(self):
        assert "autograde" not in _learner_sources().lower(), (
            "'autograde' symbol found in learner source — assessment must be user-driven"
        )

    def test_no_auto_correct(self):
        assert "auto_correct" not in _learner_sources(), (
            "'auto_correct' symbol found in learner source"
        )

    def test_no_bypass_review(self):
        assert "bypass_review" not in _learner_sources(), (
            "'bypass_review' symbol found in learner source"
        )

    def test_no_skip_review(self):
        assert "skip_review" not in _learner_sources(), (
            "'skip_review' symbol found in learner source"
        )

    def test_no_auto_rate(self):
        assert "auto_rate" not in _learner_sources(), (
            "'auto_rate' symbol found in learner source"
        )

    def test_no_assessment_automation(self):
        assert "assessment_automation" not in _learner_sources(), (
            "'assessment_automation' symbol found in learner source"
        )

    def test_review_view_has_prompt_rating(self):
        """review_view.py must prompt the user for a rating — not auto-assign it."""
        rv_src = (_LEARNER_PKG / "review_view.py").read_text(encoding="utf-8")
        assert "_prompt_rating" in rv_src, (
            "_prompt_rating missing from review_view.py — user must rate cards"
        )

    def test_review_view_has_prompt_answer(self):
        """review_view.py must prompt the user for question answers."""
        rv_src = (_LEARNER_PKG / "review_view.py").read_text(encoding="utf-8")
        assert "_prompt_answer" in rv_src, (
            "_prompt_answer missing from review_view.py — user must answer questions"
        )

    def test_review_view_calls_builtin_input(self):
        """Ratings and answers must come from real user input(), not a mock."""
        rv_src = (_LEARNER_PKG / "review_view.py").read_text(encoding="utf-8")
        assert "input(" in rv_src, (
            "builtin input() not found in review_view.py"
        )

    def test_review_view_does_not_import_llm_backend(self):
        """Ratings must come from the user, NOT an LLM completion."""
        rv_src = (_LEARNER_PKG / "review_view.py").read_text(encoding="utf-8")
        assert "MockLLM" not in rv_src, "review_view.py imports MockLLM"
        assert "ClaudeCliBackend" not in rv_src, "review_view.py imports ClaudeCliBackend"
        assert "LLMBackend" not in rv_src, "review_view.py imports LLMBackend"

    def test_grep_assessment_automation_returns_empty(self):
        """Direct grep equivalent: no file in learner/ contains 'assessment_automation'."""
        matches = []
        for f in sorted(_LEARNER_PKG.glob("*.py")):
            text = f.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), 1):
                if "assessment_automation" in line.lower():
                    matches.append(f"{f.name}:{lineno}: {line.rstrip()}")
        assert not matches, (
            "assessment_automation symbol found:\n" + "\n".join(matches)
        )
