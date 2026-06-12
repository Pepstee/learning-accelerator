"""
tests/test_acceptance_repair_verification.py

Adversarial verification of the repaired acceptance.sh pipeline.

Acceptance criteria exercised:
  1. subprocess.run(['bash', 'acceptance.sh']) exits with returncode 0.
  2. Combined stdout contains output tokens from every subcommand section:
       ingest, questions, flashcards, summary, analytics, study-plan.
  3. No assessment-automation or anti-cheat symbols in learner/ source.
  4. Every CLI subcommand individually with --mock flag also exits 0.

The acceptance.sh script is run once per pytest session (session-scoped
fixture) to avoid redundant shell invocations.  Individual subcommand tests
are isolated with their own temp dirs.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_ACCEPTANCE_SCRIPT = _PROJECT_ROOT / "acceptance.sh"
_LEARNER_PKG = _PROJECT_ROOT / "learner"


# ---------------------------------------------------------------------------
# Session-scoped fixture: run acceptance.sh exactly once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def acceptance_result() -> subprocess.CompletedProcess:
    """Run acceptance.sh as the acceptance criteria specifies; cache for the session."""
    return subprocess.run(
        ["bash", str(_ACCEPTANCE_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=120,
    )


# ---------------------------------------------------------------------------
# Module-scoped fixture: populated data dir for subcommand tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_data_dir(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Module-scoped env with a populated data dir and source file for --mock tests."""
    base = tmp_path_factory.mktemp("mock_data")
    source = base / "material.txt"
    source.write_text(
        "Mitosis is cell division producing two genetically identical daughter cells.",
        encoding="utf-8",
    )
    topic_bundle = {
        "summary": "Mitosis produces two identical daughter cells.",
        "cards": [
            {"front": "What is mitosis?", "back": "Cell division producing two identical cells."}
        ],
        "questions": [
            {
                "stem": "How many cells does mitosis produce?",
                "choices": ["One", "Two", "Four", "Eight"],
                "answer_index": 1,
                "explanation": "Mitosis produces two genetically identical daughter cells.",
            }
        ],
    }
    (base / "material.json").write_text(json.dumps(topic_bundle, indent=2), encoding="utf-8")
    empty_bundle = {"summary": "Empty.", "cards": [], "questions": []}
    (base / "empty.json").write_text(json.dumps(empty_bundle, indent=2), encoding="utf-8")
    return {"data_dir": base, "source": source}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INTERACTIVE_SUBCOMMANDS = {"practice", "review"}


def _run_mock_subcommand(
    subcommand: str,
    extra_args: list[str],
    data_dir: pathlib.Path,
    *,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    """Invoke python -m learner --mock <subcommand> as a real subprocess."""
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
    )


def _extra_args_for(subcommand: str, env: dict) -> list[str]:
    if subcommand in ("ingest", "generate"):
        return [str(env["source"])]
    if subcommand in ("summary", "flashcards", "questions"):
        return ["material"]
    if subcommand == "exam":
        return ["empty"]
    return []


def _data_dir_for(subcommand: str, env: dict) -> pathlib.Path:
    if subcommand in _INTERACTIVE_SUBCOMMANDS:
        empty = env["data_dir"] / "_empty"
        empty.mkdir(exist_ok=True)
        return empty
    return env["data_dir"]


def _learner_sources() -> str:
    return "\n".join(f.read_text(encoding="utf-8") for f in sorted(_LEARNER_PKG.glob("*.py")))


# ---------------------------------------------------------------------------
# 1. acceptance.sh exit code — the primary acceptance criterion
# ---------------------------------------------------------------------------

class TestAcceptanceExitCode:

    def test_exits_zero(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """subprocess.run(['bash', 'acceptance.sh']) must exit with returncode 0."""
        assert acceptance_result.returncode == 0, (
            f"acceptance.sh exited {acceptance_result.returncode}\n"
            f"stdout:\n{acceptance_result.stdout}\n"
            f"stderr:\n{acceptance_result.stderr}"
        )

    def test_ok_terminal_marker_present(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """acceptance.sh must print 'acceptance: OK' to signal a complete run."""
        assert "acceptance: OK" in acceptance_result.stdout, (
            f"Terminal marker 'acceptance: OK' absent.\nstdout:\n{acceptance_result.stdout}"
        )

    def test_no_stderr_on_clean_run(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """A successful run must produce no output on stderr."""
        if acceptance_result.returncode == 0:
            assert not acceptance_result.stderr.strip(), (
                f"acceptance.sh produced stderr on a clean run:\n{acceptance_result.stderr}"
            )

    def test_stdout_is_non_empty(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """acceptance.sh must produce at least some stdout."""
        assert acceptance_result.stdout.strip(), "acceptance.sh produced no stdout at all"

    def test_exact_invocation_relative_path(self) -> None:
        """['bash', 'acceptance.sh'] with cwd=project root must also exit 0."""
        result = subprocess.run(
            ["bash", "acceptance.sh"],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Relative-path invocation exited {result.returncode}\n"
            f"stderr:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# 2. Output tokens from every subcommand section
#    Each method verifies a specific token from a specific section.
# ---------------------------------------------------------------------------

class TestSectionOutputTokens:

    # --- ingest section ---

    def test_ingest_section_header(self, acceptance_result: subprocess.CompletedProcess) -> None:
        assert "=== ingest ===" in acceptance_result.stdout, (
            "Ingest section header '=== ingest ===' not found — section may have been skipped."
        )

    def test_ingest_summary_token(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """ingest must print a 'Summary:' line."""
        assert "Summary:" in acceptance_result.stdout, (
            "No 'Summary:' in acceptance.sh stdout — ingest section may have failed.\n"
            f"stdout:\n{acceptance_result.stdout}"
        )

    def test_ingest_cards_token(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """ingest must print a 'Cards:' line."""
        assert "Cards:" in acceptance_result.stdout

    def test_ingest_questions_token(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """ingest must print a 'Questions:' line."""
        assert "Questions:" in acceptance_result.stdout

    def test_ingest_saved_token(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """ingest must print a 'Saved:' line confirming the bundle was written."""
        assert "Saved:" in acceptance_result.stdout, (
            "No 'Saved:' line — ingest may not have written the bundle file."
        )

    def test_ingest_cards_count_at_least_1(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """The numeric count on the Cards: line must be >= 1."""
        lines = acceptance_result.stdout.splitlines()
        cards_line = next((ln for ln in lines if ln.startswith("Cards:")), None)
        assert cards_line is not None, "No 'Cards:' line found in stdout"
        count_str = cards_line.split()[-1]
        assert count_str.isdigit(), f"Cards: value is not numeric: {cards_line!r}"
        assert int(count_str) >= 1, f"Expected Cards >= 1, got: {cards_line!r}"

    def test_ingest_questions_count_at_least_1(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """The numeric count on the Questions: line must be >= 1."""
        lines = acceptance_result.stdout.splitlines()
        q_line = next((ln for ln in lines if ln.startswith("Questions:")), None)
        assert q_line is not None, "No 'Questions:' line found in stdout"
        count_str = q_line.split()[-1]
        assert count_str.isdigit(), f"Questions: value is not numeric: {q_line!r}"
        assert int(count_str) >= 1, f"Expected Questions >= 1, got: {q_line!r}"

    # --- questions section ---

    def test_questions_section_header(self, acceptance_result: subprocess.CompletedProcess) -> None:
        assert "=== questions ===" in acceptance_result.stdout, (
            "'=== questions ===' header not found — questions section may have been skipped."
        )

    def test_questions_q_token_after_header(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """questions subcommand must emit at least one 'Q:' line after its header."""
        lines = acceptance_result.stdout.splitlines()
        after_header = False
        for line in lines:
            if "=== questions ===" in line:
                after_header = True
                continue
            if after_header and line.startswith("==="):
                break  # entered next section without finding Q:
            if after_header and "Q:" in line:
                return
        pytest.fail(
            "No 'Q:' line found after '=== questions ===' header.\n"
            f"stdout:\n{acceptance_result.stdout}"
        )

    # --- flashcards section ---

    def test_flashcards_section_header(self, acceptance_result: subprocess.CompletedProcess) -> None:
        assert "=== flashcards ===" in acceptance_result.stdout, (
            "'=== flashcards ===' header not found — flashcards section may have been skipped."
        )

    def test_flashcards_q_token_after_header(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """flashcards subcommand must emit a 'Q:' line after its header."""
        lines = acceptance_result.stdout.splitlines()
        after_header = False
        for line in lines:
            if "=== flashcards ===" in line:
                after_header = True
                continue
            if after_header and line.startswith("==="):
                break
            if after_header and "Q:" in line:
                return
        pytest.fail(
            "No 'Q:' line found after '=== flashcards ===' header.\n"
            f"stdout:\n{acceptance_result.stdout}"
        )

    def test_flashcards_a_token(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """flashcards subcommand must emit at least one 'A:' line."""
        assert "A:" in acceptance_result.stdout, (
            "No 'A:' in acceptance.sh stdout — flashcards subcommand may not have run.\n"
            f"stdout:\n{acceptance_result.stdout}"
        )

    # --- summary section ---

    def test_summary_section_header(self, acceptance_result: subprocess.CompletedProcess) -> None:
        assert "=== summary ===" in acceptance_result.stdout, (
            "'=== summary ===' header not found — summary section may have been skipped."
        )

    def test_summary_section_non_empty_output(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """The summary section must produce at least one non-blank output line after its header."""
        lines = acceptance_result.stdout.splitlines()
        found_header = False
        for line in lines:
            if "=== summary ===" in line:
                found_header = True
                continue
            if found_header and line.startswith("==="):
                break
            if found_header and line.strip():
                return
        pytest.fail(
            "No non-empty output found after '=== summary ===' header.\n"
            f"stdout:\n{acceptance_result.stdout}"
        )

    # --- analytics section ---

    def test_analytics_section_header(self, acceptance_result: subprocess.CompletedProcess) -> None:
        assert "=== analytics ===" in acceptance_result.stdout, (
            "'=== analytics ===' header not found — analytics section may have been skipped."
        )

    def test_analytics_sessions_token(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """analytics must emit 'Sessions:' (from the seeded session record)."""
        assert "Sessions:" in acceptance_result.stdout, (
            "No 'Sessions:' in acceptance.sh stdout — analytics seeding or output broken.\n"
            f"stdout:\n{acceptance_result.stdout}"
        )

    def test_analytics_total_reviews_token(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """analytics must emit a 'Total reviews:' line."""
        assert "Total reviews:" in acceptance_result.stdout, (
            "No 'Total reviews:' line — analytics may not have printed full output.\n"
            f"stdout:\n{acceptance_result.stdout}"
        )

    def test_analytics_sessions_count_numeric_and_positive(
        self, acceptance_result: subprocess.CompletedProcess
    ) -> None:
        """The Sessions: count must be a positive integer (at least 1 session was seeded)."""
        lines = acceptance_result.stdout.splitlines()
        # The 'Sessions:' line emitted by cmd_analytics looks like: "Sessions:     1"
        sessions_line = next(
            (ln for ln in lines if ln.strip().startswith("Sessions:") and "===" not in ln),
            None,
        )
        assert sessions_line is not None, "No 'Sessions:' line from analytics found"
        parts = sessions_line.split()
        count_str = parts[-1]
        assert count_str.isdigit(), f"Sessions: value not numeric: {sessions_line!r}"
        assert int(count_str) >= 1, f"Expected Sessions >= 1, got: {sessions_line!r}"

    # --- study-plan section ---

    def test_study_plan_section_header(self, acceptance_result: subprocess.CompletedProcess) -> None:
        assert "=== study-plan ===" in acceptance_result.stdout, (
            "'=== study-plan ===' header not found — study-plan section may have been skipped."
        )

    def test_study_plan_non_empty_output(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """study-plan must produce at least one non-blank line after its header."""
        lines = acceptance_result.stdout.splitlines()
        found_header = False
        for line in lines:
            if "=== study-plan ===" in line:
                found_header = True
                continue
            if found_header and line.startswith("==="):
                break
            if found_header and line.strip() and line.strip() != "acceptance: OK":
                return
        pytest.fail(
            "study-plan section produced no output after its header.\n"
            f"stdout:\n{acceptance_result.stdout}"
        )

    def test_study_plan_references_seeded_topic(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """study-plan must reference 'sample_material' — the topic from the seeded session."""
        assert "sample_material" in acceptance_result.stdout, (
            "'sample_material' not found in study-plan output. "
            "The seeded session (quality=2) should produce a weak area for this topic.\n"
            f"stdout:\n{acceptance_result.stdout}"
        )

    def test_study_plan_contains_error_rate(self, acceptance_result: subprocess.CompletedProcess) -> None:
        """study-plan advice for a quality=2 session must include 'error rate'."""
        assert "error rate" in acceptance_result.stdout, (
            "'error rate' not found — study-plan may not have generated weak-area advice.\n"
            f"stdout:\n{acceptance_result.stdout}"
        )


# ---------------------------------------------------------------------------
# 3. No assessment-automation or anti-cheat symbols in learner/ source
# ---------------------------------------------------------------------------

_FORBIDDEN_SYMBOLS = [
    "autograde",
    "auto_correct",
    "auto_rate",
    "bypass_review",
    "skip_review",
    "assessment_automation",
    "anti_cheat",
    "cheat_detection",
    "force_correct",
    "auto_answer",
    "auto_submit",
    "cheat_flag",
]


@pytest.mark.parametrize("symbol", _FORBIDDEN_SYMBOLS)
def test_no_forbidden_symbol_in_learner_source(symbol: str) -> None:
    """No assessment-automation or anti-cheat symbol may appear in learner/ source."""
    matches = []
    for f in sorted(_LEARNER_PKG.glob("*.py")):
        text = f.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if symbol.lower() in line.lower():
                matches.append(f"{f.name}:{lineno}: {line.rstrip()}")
    assert not matches, (
        f"Forbidden symbol '{symbol}' found in learner/ source:\n" + "\n".join(matches)
    )


def test_review_view_has_prompt_rating() -> None:
    """review_view.py must call _prompt_rating() to block on real user input for cards."""
    src = (_LEARNER_PKG / "review_view.py").read_text(encoding="utf-8")
    assert "_prompt_rating" in src, (
        "_prompt_rating() absent from review_view.py — card ratings must require user interaction"
    )


def test_review_view_has_prompt_answer() -> None:
    """review_view.py must call _prompt_answer() for question responses."""
    src = (_LEARNER_PKG / "review_view.py").read_text(encoding="utf-8")
    assert "_prompt_answer" in src, (
        "_prompt_answer() absent from review_view.py — question answers must require user interaction"
    )


def test_review_view_calls_builtin_input() -> None:
    """Ratings and answers must originate from real user input(), not a mock."""
    src = (_LEARNER_PKG / "review_view.py").read_text(encoding="utf-8")
    assert "input(" in src, "builtin input() not found in review_view.py"


def test_review_view_does_not_use_llm_backend() -> None:
    """Ratings must come from the user, not an LLM completion."""
    src = (_LEARNER_PKG / "review_view.py").read_text(encoding="utf-8")
    assert "MockLLM" not in src, "review_view.py references MockLLM"
    assert "ClaudeCliBackend" not in src, "review_view.py references ClaudeCliBackend"
    assert "LLMBackend" not in src, "review_view.py references LLMBackend"


def test_review_view_prompt_rating_is_function_not_constant() -> None:
    """_prompt_rating must be a function definition, not a constant that bypasses input."""
    src = (_LEARNER_PKG / "review_view.py").read_text(encoding="utf-8")
    assert "def _prompt_rating" in src, (
        "_prompt_rating must be defined as a function so it actually calls input()"
    )


def test_review_view_prompt_answer_is_function_not_constant() -> None:
    """_prompt_answer must be a function definition."""
    src = (_LEARNER_PKG / "review_view.py").read_text(encoding="utf-8")
    assert "def _prompt_answer" in src, (
        "_prompt_answer must be defined as a function so it actually calls input()"
    )


# ---------------------------------------------------------------------------
# 4. Every CLI subcommand with --mock flag exits 0
# ---------------------------------------------------------------------------

_ALL_SUBCOMMANDS = [
    "ingest",
    "generate",
    "questions",
    "summary",
    "flashcards",
    "practice",
    "review",
    "exam",
    "study-plan",
    "analytics",
]


@pytest.mark.parametrize("subcommand", _ALL_SUBCOMMANDS)
def test_subcommand_mock_flag_exits_0(subcommand: str, mock_data_dir: dict) -> None:
    """Every registered subcommand must exit 0 when invoked with the --mock flag."""
    result = _run_mock_subcommand(
        subcommand,
        _extra_args_for(subcommand, mock_data_dir),
        _data_dir_for(subcommand, mock_data_dir),
    )
    assert result.returncode == 0, (
        f"'{subcommand} --mock' exited {result.returncode}\n"
        f"args: {_extra_args_for(subcommand, mock_data_dir)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("subcommand", _ALL_SUBCOMMANDS)
def test_subcommand_mock_flag_no_traceback(subcommand: str, mock_data_dir: dict) -> None:
    """No subcommand with --mock should produce a Python traceback on stderr."""
    result = _run_mock_subcommand(
        subcommand,
        _extra_args_for(subcommand, mock_data_dir),
        _data_dir_for(subcommand, mock_data_dir),
    )
    assert "Traceback (most recent call last)" not in result.stderr, (
        f"'{subcommand} --mock' produced a traceback:\n{result.stderr}"
    )


@pytest.mark.parametrize("subcommand", _ALL_SUBCOMMANDS)
def test_subcommand_mock_flag_produces_output(subcommand: str, mock_data_dir: dict) -> None:
    """Every subcommand with --mock must produce at least one character of stdout."""
    result = _run_mock_subcommand(
        subcommand,
        _extra_args_for(subcommand, mock_data_dir),
        _data_dir_for(subcommand, mock_data_dir),
    )
    # Only assert output when the invocation succeeded — failure is reported by the exit-0 test.
    if result.returncode == 0:
        assert result.stdout.strip(), (
            f"'{subcommand} --mock' exited 0 but produced no stdout.\n"
            f"stderr:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# 5. Per-subcommand output contracts (supplementary, targetted)
# ---------------------------------------------------------------------------

def test_ingest_mock_creates_bundle_with_cards_and_questions(mock_data_dir: dict) -> None:
    """ingest --mock must write a .json bundle containing non-empty cards and questions."""
    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        source = td_path / "osmosis.txt"
        source.write_text("Osmosis moves water across semipermeable membranes.", encoding="utf-8")
        result = _run_mock_subcommand("ingest", [str(source)], td_path)
        assert result.returncode == 0, f"ingest failed:\n{result.stderr}"
        bundle_path = td_path / "osmosis.json"
        assert bundle_path.exists(), "ingest --mock did not create osmosis.json"
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert bundle.get("cards"), "Bundle has no cards — MockLLM may have broken"
        assert bundle.get("questions"), "Bundle has no questions — MockLLM may have broken"
        assert bundle.get("summary"), "Bundle has no summary"


def test_questions_mock_prints_answer_line(mock_data_dir: dict) -> None:
    """questions --mock must print an 'Answer:' line for each question in the bundle."""
    result = _run_mock_subcommand("questions", ["material"], mock_data_dir["data_dir"])
    assert result.returncode == 0
    assert "Q:" in result.stdout, f"No Q: line in questions output:\n{result.stdout}"
    assert "Answer:" in result.stdout, (
        f"No 'Answer:' line in questions output.\nstdout:\n{result.stdout}"
    )


def test_flashcards_mock_prints_q_and_a_lines(mock_data_dir: dict) -> None:
    """flashcards --mock must emit both Q: and A: lines."""
    result = _run_mock_subcommand("flashcards", ["material"], mock_data_dir["data_dir"])
    assert result.returncode == 0
    assert "Q:" in result.stdout, f"No Q: in flashcards output:\n{result.stdout}"
    assert "A:" in result.stdout, f"No A: in flashcards output:\n{result.stdout}"


def test_summary_mock_prints_bundle_text(tmp_path: pathlib.Path) -> None:
    """summary --mock must print the bundle's summary field verbatim from disk."""
    known_summary = "Unique canary: osmosis drives water across membranes."
    bundle = {"summary": known_summary, "cards": [], "questions": []}
    (tmp_path / "osmotopic.json").write_text(json.dumps(bundle), encoding="utf-8")
    result = _run_mock_subcommand("summary", ["osmotopic"], tmp_path)
    assert result.returncode == 0
    assert known_summary in result.stdout, (
        f"summary --mock did not print the bundle summary verbatim.\nstdout:\n{result.stdout}"
    )


def test_analytics_mock_with_seeded_session_prints_sessions(tmp_path: pathlib.Path) -> None:
    """analytics --mock with a seeded session record must print a 'Sessions:' line."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "20260612T000000.json").write_text(
        json.dumps({
            "session_id": "repair-test-id",
            "started_at": "2026-06-12T00:00:00",
            "duration_seconds": 45.0,
            "ratings": [{"card_front": "Q?", "topic": "mytopic", "quality": 3}],
        }),
        encoding="utf-8",
    )
    result = _run_mock_subcommand("analytics", [], tmp_path)
    assert result.returncode == 0
    assert "Sessions:" in result.stdout, (
        f"'Sessions:' absent from analytics output:\n{result.stdout}"
    )


def test_study_plan_mock_with_weak_session_references_topic(tmp_path: pathlib.Path) -> None:
    """study-plan --mock with a low-quality session must name the weak topic in output."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "20260612T000000.json").write_text(
        json.dumps({
            "session_id": "weak-id",
            "started_at": "2026-06-12T00:00:00",
            "duration_seconds": 30.0,
            "ratings": [{"card_front": "Q?", "topic": "hard_topic", "quality": 1}],
        }),
        encoding="utf-8",
    )
    result = _run_mock_subcommand("study-plan", [], tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip(), "study-plan produced no output"
    assert "hard_topic" in result.stdout, (
        f"study-plan did not reference 'hard_topic'.\nstdout:\n{result.stdout}"
    )
    assert "error rate" in result.stdout, (
        f"study-plan did not include 'error rate' for the weak area.\nstdout:\n{result.stdout}"
    )


def test_ingest_mock_missing_source_exits_nonzero(tmp_path: pathlib.Path) -> None:
    """ingest --mock with a nonexistent file must exit non-zero and report to stderr."""
    result = _run_mock_subcommand("ingest", [str(tmp_path / "ghost.txt")], tmp_path)
    assert result.returncode != 0
    assert "error" in result.stderr.lower(), (
        f"Expected 'error' in stderr for missing source:\n{result.stderr}"
    )


def test_questions_mock_missing_topic_exits_nonzero(tmp_path: pathlib.Path) -> None:
    """questions --mock with an unknown topic must exit non-zero."""
    result = _run_mock_subcommand("questions", ["no_such_topic"], tmp_path)
    assert result.returncode != 0


def test_flashcards_mock_missing_topic_exits_nonzero(tmp_path: pathlib.Path) -> None:
    """flashcards --mock with an unknown topic must exit non-zero."""
    result = _run_mock_subcommand("flashcards", ["no_such_topic"], tmp_path)
    assert result.returncode != 0


def test_summary_mock_missing_topic_exits_nonzero(tmp_path: pathlib.Path) -> None:
    """summary --mock with an unknown topic must exit non-zero."""
    result = _run_mock_subcommand("summary", ["no_such_topic"], tmp_path)
    assert result.returncode != 0


def test_exam_mock_empty_bundle_exits_0_and_says_no_questions(mock_data_dir: dict) -> None:
    """exam --mock with an empty bundle must exit 0 and print 'No questions'."""
    result = _run_mock_subcommand("exam", ["empty"], mock_data_dir["data_dir"])
    assert result.returncode == 0
    assert "No questions" in result.stdout, (
        f"Expected 'No questions' from exam on empty bundle:\n{result.stdout}"
    )


def test_practice_mock_empty_dir_exits_0(tmp_path: pathlib.Path) -> None:
    """practice --mock with no bundles must exit 0 immediately (nothing due)."""
    result = _run_mock_subcommand("practice", [], tmp_path)
    assert result.returncode == 0, f"practice failed:\n{result.stderr}"
    assert result.stdout.strip(), "practice produced no output at all"


def test_review_mock_empty_dir_exits_0(tmp_path: pathlib.Path) -> None:
    """review --mock with no bundles must exit 0 immediately (nothing due)."""
    result = _run_mock_subcommand("review", [], tmp_path)
    assert result.returncode == 0, f"review failed:\n{result.stderr}"
    assert result.stdout.strip(), "review produced no output at all"


def test_analytics_mock_no_history_exits_0(tmp_path: pathlib.Path) -> None:
    """analytics --mock with no session history must exit 0 and explain the situation."""
    result = _run_mock_subcommand("analytics", [], tmp_path)
    assert result.returncode == 0
    assert "No session history" in result.stdout


def test_study_plan_mock_no_history_exits_0(tmp_path: pathlib.Path) -> None:
    """study-plan --mock with no history must exit 0 and say no weak areas."""
    result = _run_mock_subcommand("study-plan", [], tmp_path)
    assert result.returncode == 0
    assert "No weak areas" in result.stdout


# ---------------------------------------------------------------------------
# 6. acceptance.sh script completeness — all required subcommands are present
# ---------------------------------------------------------------------------

class TestAcceptanceScriptCompleteness:
    """The script source must invoke every required subcommand."""

    def _content(self) -> str:
        return _ACCEPTANCE_SCRIPT.read_text(encoding="utf-8")

    def test_script_exists(self) -> None:
        assert _ACCEPTANCE_SCRIPT.exists(), f"acceptance.sh not found at {_ACCEPTANCE_SCRIPT}"

    def test_script_is_executable(self) -> None:
        import os
        assert os.access(_ACCEPTANCE_SCRIPT, os.X_OK), (
            "acceptance.sh is not executable — run: chmod +x acceptance.sh"
        )

    def test_script_has_set_euo_pipefail(self) -> None:
        assert "set -euo pipefail" in self._content(), (
            "acceptance.sh missing 'set -euo pipefail' — errors will be silently swallowed"
        )

    def test_script_sets_learner_mock(self) -> None:
        assert "LEARNER_MOCK" in self._content(), (
            "acceptance.sh does not set LEARNER_MOCK — real LLM calls would be attempted"
        )

    def test_script_invokes_ingest(self) -> None:
        assert " ingest " in self._content() or "\ningest " in self._content() or "ingest\n" in self._content() or "ingest " in self._content()

    def test_script_invokes_questions(self) -> None:
        assert "questions" in self._content(), (
            "'questions' not found in acceptance.sh — questions subcommand not exercised"
        )

    def test_script_invokes_flashcards(self) -> None:
        assert "flashcards" in self._content()

    def test_script_invokes_summary(self) -> None:
        assert "summary" in self._content()

    def test_script_invokes_analytics(self) -> None:
        assert "analytics" in self._content()

    def test_script_invokes_study_plan(self) -> None:
        assert "study-plan" in self._content(), (
            "'study-plan' not found in acceptance.sh — study-plan subcommand not exercised"
        )

    def test_script_seeds_session_for_analytics(self) -> None:
        """acceptance.sh must seed a sessions/ dir so analytics produces real data."""
        content = self._content()
        assert "sessions" in content, (
            "acceptance.sh does not seed a sessions/ directory — "
            "analytics would print 'No session history' instead of 'Sessions:'"
        )

    def test_script_validates_analytics_output(self) -> None:
        """acceptance.sh must grep/check for 'Sessions:' in analytics output."""
        assert "Sessions:" in self._content(), (
            "acceptance.sh does not validate 'Sessions:' in analytics output"
        )

    def test_script_validates_summary_not_empty(self) -> None:
        """acceptance.sh must validate that the ingest Summary line is non-empty."""
        content = self._content()
        assert "Summary" in content and ("FAIL" in content or "exit 1" in content), (
            "acceptance.sh does not validate the Summary line from ingest"
        )
