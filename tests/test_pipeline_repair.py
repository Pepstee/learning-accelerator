"""
tests/test_pipeline_repair.py

Independent verification of three acceptance criteria for the repaired pipeline:

  1. acceptance.sh subprocess exits 0 with non-empty stdout.
  2. Each CLI subcommand (ingest, generate, summary, flashcards, study-plan,
     analytics) exits 0 under ``python -m learner --mock`` with valid inputs.
  3. The learner package contains none of the forbidden assessment-automation
     symbols: 'cheat', 'bypass_exam', 'submit_answer', 'automate.*quiz'.

Tests are written as independent checks so they can fail individually.
Implementation files are never modified or mocked.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_ACCEPTANCE_SCRIPT = _PROJECT_ROOT / "acceptance.sh"
_LEARNER_PKG = _PROJECT_ROOT / "learner"

# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------

def _py_files() -> list[pathlib.Path]:
    return sorted(_LEARNER_PKG.glob("*.py"))


def _learner_sources() -> str:
    return "\n".join(f.read_text(encoding="utf-8") for f in _py_files())


def _run_acceptance(timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(_ACCEPTANCE_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=timeout,
    )


def _cli(
    subcommand: str,
    extra_args: list[str],
    data_dir: pathlib.Path,
    *,
    timeout: int = 30,
    stdin: str | None = None,
) -> subprocess.CompletedProcess:
    """Invoke ``python -m learner --mock`` as a real subprocess."""
    env = {**os.environ, "LEARNER_MOCK": "1"}
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
        env=env,
        timeout=timeout,
        input=stdin,
    )


@pytest.fixture
def populated_data_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    A self-contained data directory used by subcommands that need a bundle
    (summary, flashcards) and a source file for ingest / generate.
    """
    # Source material
    source = tmp_path / "material.txt"
    source.write_text(
        "Photosynthesis converts sunlight into chemical energy stored as glucose.",
        encoding="utf-8",
    )

    # Pre-built topic bundle consumed by summary + flashcards
    bundle = {
        "summary": "Mock photosynthesis summary.",
        "cards": [
            {"front": "What is ATP?", "back": "Adenosine triphosphate, an energy carrier."},
            {"front": "What is glucose?", "back": "A simple sugar produced by photosynthesis."},
        ],
        "questions": [
            {
                "stem": "What does photosynthesis produce?",
                "choices": ["Glucose", "Starch", "Protein", "Lipid"],
                "answer_index": 0,
                "explanation": "Photosynthesis produces glucose.",
            }
        ],
    }
    (tmp_path / "material.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    # Empty bundle used for exam (must exit 0 when there are no questions)
    (tmp_path / "empty.json").write_text(
        json.dumps({"summary": "empty", "cards": [], "questions": []}), encoding="utf-8"
    )

    return tmp_path


# ===========================================================================
# 1. acceptance.sh must exit 0 with non-empty stdout
# ===========================================================================

class TestAcceptanceShellScript:
    """Criterion: subprocess.run(['bash', 'acceptance.sh'], ...) returns
    returncode==0 with non-empty stdout."""

    def test_script_exists_at_project_root(self):
        assert _ACCEPTANCE_SCRIPT.exists(), (
            f"acceptance.sh not found at {_ACCEPTANCE_SCRIPT}"
        )

    def test_script_is_bash_executable(self):
        assert os.access(_ACCEPTANCE_SCRIPT, os.X_OK), (
            "acceptance.sh is not executable — chmod +x acceptance.sh"
        )

    def test_exit_code_is_zero(self):
        result = _run_acceptance()
        assert result.returncode == 0, (
            f"acceptance.sh exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    def test_stdout_is_non_empty(self):
        result = _run_acceptance()
        assert result.stdout.strip(), (
            "acceptance.sh produced no stdout — it must print at minimum the Summary line"
        )

    def test_stdout_contains_ok_marker(self):
        result = _run_acceptance()
        assert "acceptance: OK" in result.stdout, (
            f"'acceptance: OK' marker absent from stdout:\n{result.stdout}"
        )

    def test_stdout_contains_summary_line(self):
        result = _run_acceptance()
        assert re.search(r"^Summary:", result.stdout, re.MULTILINE), (
            f"No 'Summary:' line in acceptance.sh stdout:\n{result.stdout}"
        )

    def test_summary_line_is_non_empty(self):
        result = _run_acceptance()
        match = re.search(r"^Summary:\s*(.+)$", result.stdout, re.MULTILINE)
        assert match and match.group(1).strip(), (
            f"Summary: line exists but is blank:\n{result.stdout}"
        )

    def test_stdout_contains_cards_line_with_positive_count(self):
        result = _run_acceptance()
        match = re.search(r"^Cards:\s*(\d+)", result.stdout, re.MULTILINE)
        assert match, f"No 'Cards: N' line in stdout:\n{result.stdout}"
        assert int(match.group(1)) >= 1, (
            f"Cards count is 0, must be ≥1:\n{result.stdout}"
        )

    def test_stdout_contains_questions_line_with_positive_count(self):
        result = _run_acceptance()
        match = re.search(r"^Questions:\s*(\d+)", result.stdout, re.MULTILINE)
        assert match, f"No 'Questions: N' line in stdout:\n{result.stdout}"
        assert int(match.group(1)) >= 1, (
            f"Questions count is 0, must be ≥1:\n{result.stdout}"
        )

    def test_stdout_contains_flashcard_q_pair(self):
        result = _run_acceptance()
        assert "Q:" in result.stdout, (
            f"No 'Q:' flashcard pair in acceptance.sh stdout:\n{result.stdout}"
        )

    def test_stderr_empty_on_clean_run(self):
        result = _run_acceptance()
        if result.returncode == 0:
            assert not result.stderr.strip(), (
                f"acceptance.sh wrote to stderr on a clean run:\n{result.stderr}"
            )

    def test_script_sets_learner_mock_env(self):
        content = _ACCEPTANCE_SCRIPT.read_text(encoding="utf-8")
        assert "LEARNER_MOCK" in content, (
            "acceptance.sh does not set LEARNER_MOCK — it would attempt real LLM calls"
        )

    def test_script_exercises_both_ingest_and_flashcards(self):
        content = _ACCEPTANCE_SCRIPT.read_text(encoding="utf-8")
        assert "ingest" in content, "acceptance.sh does not call the ingest subcommand"
        assert "flashcards" in content, "acceptance.sh does not call the flashcards subcommand"

    def test_saved_bundle_path_printed(self):
        result = _run_acceptance()
        assert "Saved:" in result.stdout, (
            f"'Saved:' path line missing from stdout:\n{result.stdout}"
        )

    def test_rerun_exits_zero_again(self):
        """A second run in the same process still exits 0 (idempotent)."""
        r1 = _run_acceptance()
        r2 = _run_acceptance()
        assert r1.returncode == 0 and r2.returncode == 0


# ===========================================================================
# 2. CLI subcommands must exit 0 under --mock with valid inputs
# ===========================================================================

class TestIngestSubcommand:
    """ingest: reads a source file, calls the LLM, writes a JSON bundle."""

    def test_exits_zero_with_valid_source(self, populated_data_dir):
        source = populated_data_dir / "material.txt"
        result = _cli("ingest", [str(source)], populated_data_dir)
        assert result.returncode == 0, (
            f"ingest exited {result.returncode}:\n{result.stderr}"
        )

    def test_stdout_has_summary_label(self, populated_data_dir):
        source = populated_data_dir / "material.txt"
        result = _cli("ingest", [str(source)], populated_data_dir)
        assert result.returncode == 0
        assert "Summary:" in result.stdout

    def test_stdout_has_cards_label(self, populated_data_dir):
        source = populated_data_dir / "material.txt"
        result = _cli("ingest", [str(source)], populated_data_dir)
        assert "Cards:" in result.stdout

    def test_stdout_has_questions_label(self, populated_data_dir):
        source = populated_data_dir / "material.txt"
        result = _cli("ingest", [str(source)], populated_data_dir)
        assert "Questions:" in result.stdout

    def test_stdout_has_saved_label(self, populated_data_dir):
        source = populated_data_dir / "material.txt"
        result = _cli("ingest", [str(source)], populated_data_dir)
        assert "Saved:" in result.stdout

    def test_creates_json_bundle_with_correct_stem(self, tmp_path):
        source = tmp_path / "mycontent.txt"
        source.write_text("Neurons transmit signals.", encoding="utf-8")
        result = _cli("ingest", [str(source)], tmp_path)
        assert result.returncode == 0
        assert (tmp_path / "mycontent.json").exists()

    def test_bundle_contains_all_keys(self, tmp_path):
        source = tmp_path / "topic.txt"
        source.write_text("Gravity pulls objects toward Earth.", encoding="utf-8")
        _cli("ingest", [str(source)], tmp_path)
        bundle = json.loads((tmp_path / "topic.json").read_text(encoding="utf-8"))
        assert "summary" in bundle
        assert "cards" in bundle
        assert "questions" in bundle

    def test_bundle_cards_is_a_list(self, tmp_path):
        source = tmp_path / "data.txt"
        source.write_text("DNA carries genetic information.", encoding="utf-8")
        _cli("ingest", [str(source)], tmp_path)
        bundle = json.loads((tmp_path / "data.json").read_text(encoding="utf-8"))
        assert isinstance(bundle["cards"], list)

    def test_bundle_questions_is_a_list(self, tmp_path):
        source = tmp_path / "data2.txt"
        source.write_text("Mitosis produces two daughter cells.", encoding="utf-8")
        _cli("ingest", [str(source)], tmp_path)
        bundle = json.loads((tmp_path / "data2.json").read_text(encoding="utf-8"))
        assert isinstance(bundle["questions"], list)

    def test_no_traceback_on_success(self, populated_data_dir):
        source = populated_data_dir / "material.txt"
        result = _cli("ingest", [str(source)], populated_data_dir)
        assert "Traceback" not in result.stderr

    def test_missing_source_file_exits_1(self, tmp_path):
        result = _cli("ingest", [str(tmp_path / "ghost.txt")], tmp_path)
        assert result.returncode == 1

    def test_missing_source_file_prints_error_to_stderr(self, tmp_path):
        result = _cli("ingest", [str(tmp_path / "ghost.txt")], tmp_path)
        assert "error" in result.stderr.lower()

    def test_creates_data_dir_if_missing(self, tmp_path):
        new_dir = tmp_path / "deep" / "nested"
        source = tmp_path / "src.txt"
        source.write_text("content", encoding="utf-8")
        result = _cli("ingest", [str(source)], new_dir)
        assert result.returncode == 0
        assert new_dir.exists()


class TestGenerateSubcommand:
    """generate: same behaviour as ingest — writes a bundle from source."""

    def test_exits_zero_with_valid_source(self, populated_data_dir):
        source = populated_data_dir / "material.txt"
        result = _cli("generate", [str(source)], populated_data_dir)
        assert result.returncode == 0, (
            f"generate exited {result.returncode}:\n{result.stderr}"
        )

    def test_stdout_has_summary_label(self, populated_data_dir):
        source = populated_data_dir / "material.txt"
        result = _cli("generate", [str(source)], populated_data_dir)
        assert "Summary:" in result.stdout

    def test_stdout_has_cards_label(self, populated_data_dir):
        source = populated_data_dir / "material.txt"
        result = _cli("generate", [str(source)], populated_data_dir)
        assert "Cards:" in result.stdout

    def test_stdout_has_questions_label(self, populated_data_dir):
        source = populated_data_dir / "material.txt"
        result = _cli("generate", [str(source)], populated_data_dir)
        assert "Questions:" in result.stdout

    def test_creates_json_bundle(self, tmp_path):
        source = tmp_path / "gen_topic.txt"
        source.write_text("RNA is a messenger molecule.", encoding="utf-8")
        result = _cli("generate", [str(source)], tmp_path)
        assert result.returncode == 0
        assert (tmp_path / "gen_topic.json").exists()

    def test_bundle_structure_matches_ingest(self, tmp_path):
        source = tmp_path / "shared.txt"
        source.write_text("ATP is the energy currency of the cell.", encoding="utf-8")
        _cli("generate", [str(source)], tmp_path)
        bundle = json.loads((tmp_path / "shared.json").read_text(encoding="utf-8"))
        assert all(k in bundle for k in ("summary", "cards", "questions"))

    def test_no_traceback_on_success(self, populated_data_dir):
        source = populated_data_dir / "material.txt"
        result = _cli("generate", [str(source)], populated_data_dir)
        assert "Traceback" not in result.stderr

    def test_missing_source_exits_1(self, tmp_path):
        result = _cli("generate", [str(tmp_path / "missing.txt")], tmp_path)
        assert result.returncode == 1


class TestSummarySubcommand:
    """summary: prints the 'summary' field from an existing bundle."""

    def test_exits_zero_with_valid_topic(self, populated_data_dir):
        result = _cli("summary", ["material"], populated_data_dir)
        assert result.returncode == 0, (
            f"summary exited {result.returncode}:\n{result.stderr}"
        )

    def test_stdout_contains_bundle_summary_text(self, populated_data_dir):
        result = _cli("summary", ["material"], populated_data_dir)
        assert "Mock photosynthesis summary." in result.stdout

    def test_no_traceback(self, populated_data_dir):
        result = _cli("summary", ["material"], populated_data_dir)
        assert "Traceback" not in result.stderr

    def test_missing_topic_exits_1(self, tmp_path):
        result = _cli("summary", ["no_such_topic"], tmp_path)
        assert result.returncode == 1

    def test_missing_topic_writes_error(self, tmp_path):
        result = _cli("summary", ["no_such_topic"], tmp_path)
        assert "error" in result.stderr.lower() or "no_such_topic" in result.stderr

    def test_corrupt_bundle_exits_1(self, tmp_path):
        (tmp_path / "broken.json").write_text("{invalid", encoding="utf-8")
        result = _cli("summary", ["broken"], tmp_path)
        assert result.returncode == 1

    def test_bundle_without_summary_key_falls_back(self, tmp_path):
        (tmp_path / "partial.json").write_text(
            json.dumps({"cards": [], "questions": []}), encoding="utf-8"
        )
        result = _cli("summary", ["partial"], tmp_path)
        assert result.returncode == 0
        assert "(no summary available)" in result.stdout

    def test_ingest_then_summary_round_trip(self, tmp_path):
        source = tmp_path / "roundtrip.txt"
        source.write_text("The Calvin cycle fixes CO2 into organic molecules.", encoding="utf-8")
        ingest_result = _cli("ingest", [str(source)], tmp_path)
        assert ingest_result.returncode == 0
        summary_result = _cli("summary", ["roundtrip"], tmp_path)
        assert summary_result.returncode == 0
        assert summary_result.stdout.strip()


class TestFlashcardsSubcommand:
    """flashcards: lists all Q/A pairs from an existing bundle."""

    def test_exits_zero_with_valid_topic(self, populated_data_dir):
        result = _cli("flashcards", ["material"], populated_data_dir)
        assert result.returncode == 0, (
            f"flashcards exited {result.returncode}:\n{result.stderr}"
        )

    def test_stdout_contains_q_prefix(self, populated_data_dir):
        result = _cli("flashcards", ["material"], populated_data_dir)
        assert "Q:" in result.stdout

    def test_stdout_contains_a_prefix(self, populated_data_dir):
        result = _cli("flashcards", ["material"], populated_data_dir)
        assert "A:" in result.stdout

    def test_all_cards_listed(self, populated_data_dir):
        result = _cli("flashcards", ["material"], populated_data_dir)
        assert "What is ATP?" in result.stdout
        assert "What is glucose?" in result.stdout

    def test_cards_numbered_from_one(self, populated_data_dir):
        result = _cli("flashcards", ["material"], populated_data_dir)
        assert "[1]" in result.stdout
        assert "[2]" in result.stdout

    def test_empty_cards_list_exits_0(self, tmp_path):
        (tmp_path / "empty.json").write_text(
            json.dumps({"summary": "s", "cards": [], "questions": []}), encoding="utf-8"
        )
        result = _cli("flashcards", ["empty"], tmp_path)
        assert result.returncode == 0
        assert "No flashcards" in result.stdout

    def test_missing_topic_exits_1(self, tmp_path):
        result = _cli("flashcards", ["ghost"], tmp_path)
        assert result.returncode == 1

    def test_no_traceback(self, populated_data_dir):
        result = _cli("flashcards", ["material"], populated_data_dir)
        assert "Traceback" not in result.stderr

    def test_ingest_then_flashcards_round_trip(self, tmp_path):
        source = tmp_path / "rt2.txt"
        source.write_text("NADPH is a reducing agent used in biosynthesis.", encoding="utf-8")
        assert _cli("ingest", [str(source)], tmp_path).returncode == 0
        result = _cli("flashcards", ["rt2"], tmp_path)
        assert result.returncode == 0
        assert "Q:" in result.stdout


class TestStudyPlanSubcommand:
    """study-plan: prints advice based on session history."""

    def test_exits_zero_with_empty_history(self, tmp_path):
        result = _cli("study-plan", [], tmp_path)
        assert result.returncode == 0, (
            f"study-plan exited {result.returncode}:\n{result.stderr}"
        )

    def test_no_history_prints_no_weak_areas_message(self, tmp_path):
        result = _cli("study-plan", [], tmp_path)
        assert "No weak areas" in result.stdout

    def test_exits_zero_with_session_history(self, tmp_path):
        _write_session(tmp_path, "biology", quality=0)
        result = _cli("study-plan", [], tmp_path)
        assert result.returncode == 0, (
            f"study-plan exited {result.returncode}:\n{result.stderr}"
        )

    def test_with_weak_areas_mentions_topic(self, tmp_path):
        _write_session(tmp_path, "calculus", quality=0)
        result = _cli("study-plan", [], tmp_path)
        assert "calculus" in result.stdout

    def test_no_traceback(self, tmp_path):
        result = _cli("study-plan", [], tmp_path)
        assert "Traceback" not in result.stderr

    def test_multiple_weak_areas_all_mentioned(self, tmp_path):
        for topic, q in [("math", 0), ("physics", 1)]:
            _write_session(tmp_path, topic, quality=q, suffix=topic)
        result = _cli("study-plan", [], tmp_path)
        assert result.returncode == 0
        assert "math" in result.stdout
        assert "physics" in result.stdout

    def test_perfect_history_no_crash(self, tmp_path):
        _write_session(tmp_path, "history", quality=5)
        result = _cli("study-plan", [], tmp_path)
        assert result.returncode == 0


class TestAnalyticsSubcommand:
    """analytics: prints session stats and weak areas."""

    def test_exits_zero_with_no_history(self, tmp_path):
        result = _cli("analytics", [], tmp_path)
        assert result.returncode == 0, (
            f"analytics exited {result.returncode}:\n{result.stderr}"
        )

    def test_no_history_prints_expected_message(self, tmp_path):
        result = _cli("analytics", [], tmp_path)
        assert "No session history" in result.stdout

    def test_exits_zero_with_session_history(self, tmp_path):
        _write_session(tmp_path, "chemistry", quality=3)
        result = _cli("analytics", [], tmp_path)
        assert result.returncode == 0

    def test_with_history_shows_sessions_count(self, tmp_path):
        _write_session(tmp_path, "chemistry", quality=3)
        result = _cli("analytics", [], tmp_path)
        assert "Sessions:" in result.stdout

    def test_with_history_shows_total_reviews(self, tmp_path):
        _write_session(tmp_path, "chemistry", quality=3)
        result = _cli("analytics", [], tmp_path)
        assert "Total reviews" in result.stdout

    def test_high_error_rate_shows_weak_area_topic(self, tmp_path):
        _write_session(tmp_path, "topology", quality=0)
        result = _cli("analytics", [], tmp_path)
        assert "topology" in result.stdout

    def test_no_traceback(self, tmp_path):
        result = _cli("analytics", [], tmp_path)
        assert "Traceback" not in result.stderr

    def test_multiple_sessions_aggregated(self, tmp_path):
        for suffix in ("a", "b"):
            _write_session(tmp_path, "algebra", quality=2, suffix=suffix)
        result = _cli("analytics", [], tmp_path)
        assert result.returncode == 0
        assert "Sessions:" in result.stdout

    def test_analytics_error_rate_format_percentage(self, tmp_path):
        _write_session(tmp_path, "stats", quality=0)
        result = _cli("analytics", [], tmp_path)
        assert "%" in result.stdout


# ---------------------------------------------------------------------------
# Helpers used by the CLI subcommand tests above
# ---------------------------------------------------------------------------

def _write_session(
    data_dir: pathlib.Path,
    topic: str,
    quality: int,
    suffix: str = "0",
) -> None:
    sessions_dir = data_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    # Use a stable hash of the suffix to produce a unique-but-valid timestamp
    sec = abs(hash(suffix)) % 60
    record = {
        "session_id": f"test-session-{suffix}",
        "started_at": f"2024-01-01T10:00:{sec:02d}",
        "duration_seconds": 60.0,
        "ratings": [{"card_front": "Q", "topic": topic, "quality": quality}],
    }
    filename = f"20240101T1000{sec:02d}_{suffix}.json"
    (sessions_dir / filename).write_text(
        json.dumps(record), encoding="utf-8"
    )


# ===========================================================================
# Parametric smoke test: each of the six required subcommands exits 0
# ===========================================================================

_REQUIRED_SUBCOMMANDS = [
    "ingest",
    "generate",
    "summary",
    "flashcards",
    "study-plan",
    "analytics",
]


@pytest.fixture
def smoke_env(tmp_path: pathlib.Path):
    source = tmp_path / "smoke.txt"
    source.write_text("Enzymes catalyse biochemical reactions.", encoding="utf-8")

    bundle = {
        "summary": "Smoke test summary.",
        "cards": [{"front": "Q1", "back": "A1"}],
        "questions": [{
            "stem": "What do enzymes do?",
            "choices": ["Catalyse", "Inhibit", "Digest", "Bind"],
            "answer_index": 0,
            "explanation": "Enzymes catalyse reactions.",
        }],
    }
    (tmp_path / "smoke.json").write_text(json.dumps(bundle), encoding="utf-8")
    return {"data_dir": tmp_path, "source": source}


def _smoke_args(subcommand: str, env: dict) -> list[str]:
    if subcommand in ("ingest", "generate"):
        return [str(env["source"])]
    if subcommand in ("summary", "flashcards"):
        return ["smoke"]
    return []


@pytest.mark.parametrize("subcommand", _REQUIRED_SUBCOMMANDS)
def test_required_subcommand_exits_zero(subcommand: str, smoke_env: dict) -> None:
    """Each required subcommand exits 0 under --mock with valid inputs."""
    args = _smoke_args(subcommand, smoke_env)
    result = _cli(subcommand, args, smoke_env["data_dir"])
    assert result.returncode == 0, (
        f"'{subcommand}' exited {result.returncode}\n"
        f"args: {args}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("subcommand", _REQUIRED_SUBCOMMANDS)
def test_required_subcommand_no_traceback(subcommand: str, smoke_env: dict) -> None:
    """No required subcommand should dump a Python traceback on stderr."""
    args = _smoke_args(subcommand, smoke_env)
    result = _cli(subcommand, args, smoke_env["data_dir"])
    assert "Traceback (most recent call last)" not in result.stderr, (
        f"'{subcommand}' dumped a traceback:\n{result.stderr}"
    )


@pytest.mark.parametrize("subcommand", _REQUIRED_SUBCOMMANDS)
def test_required_subcommand_produces_some_stdout(subcommand: str, smoke_env: dict) -> None:
    """Every required subcommand should write at least one line of output."""
    args = _smoke_args(subcommand, smoke_env)
    result = _cli(subcommand, args, smoke_env["data_dir"])
    assert result.stdout.strip(), (
        f"'{subcommand}' produced empty stdout — expected at least one output line"
    )


# ===========================================================================
# 3. Forbidden assessment-automation symbols must not appear in learner/
# ===========================================================================

def _grep_in_sources(pattern: str, *, flags: int = 0) -> list[str]:
    """Return 'file:line: content' strings for every line that matches pattern."""
    compiled = re.compile(pattern, flags)
    hits: list[str] = []
    for py_file in _py_files():
        text = py_file.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if compiled.search(line):
                hits.append(f"{py_file.name}:{lineno}: {line.rstrip()}")
    return hits


class TestForbiddenSymbols:
    """
    Criterion: grep across learner/ for the following patterns finds zero matches:
      - 'cheat'
      - 'bypass_exam'
      - 'submit_answer'
      - 'automate.*quiz'
    """

    def test_no_cheat_symbol(self):
        hits = _grep_in_sources(r"\bcheat\b", flags=re.IGNORECASE)
        assert not hits, (
            "'cheat' found in learner source — assessment shortcuts are not permitted:\n"
            + "\n".join(hits)
        )

    def test_no_bypass_exam_symbol(self):
        hits = _grep_in_sources(r"\bbypass_exam\b", flags=re.IGNORECASE)
        assert not hits, (
            "'bypass_exam' found in learner source:\n" + "\n".join(hits)
        )

    def test_no_submit_answer_symbol(self):
        hits = _grep_in_sources(r"\bsubmit_answer\b", flags=re.IGNORECASE)
        assert not hits, (
            "'submit_answer' found in learner source:\n" + "\n".join(hits)
        )

    def test_no_automate_quiz_pattern(self):
        hits = _grep_in_sources(r"automate.*quiz", flags=re.IGNORECASE)
        assert not hits, (
            "'automate.*quiz' pattern found in learner source:\n" + "\n".join(hits)
        )

    # Broader variants of the same patterns (partial matches, CamelCase, etc.)

    def test_no_cheat_as_substring(self):
        hits = _grep_in_sources(r"cheat", flags=re.IGNORECASE)
        assert not hits, (
            "'cheat' (as substring) found in learner source:\n" + "\n".join(hits)
        )

    def test_no_bypass_exam_as_substring(self):
        hits = _grep_in_sources(r"bypass.exam", flags=re.IGNORECASE)
        assert not hits, (
            "'bypass_exam' (as substring) found in learner source:\n" + "\n".join(hits)
        )

    def test_no_submit_answer_as_substring(self):
        hits = _grep_in_sources(r"submit.answer", flags=re.IGNORECASE)
        assert not hits, (
            "'submit_answer' (as substring) found in learner source:\n" + "\n".join(hits)
        )

    def test_all_py_files_scanned(self):
        """Sanity: at least the core modules are present so the grep isn't vacuously true."""
        names = {f.name for f in _py_files()}
        for required in ("cli.py", "llm.py", "generator.py", "session.py", "review_view.py"):
            assert required in names, (
                f"Expected learner module '{required}' not found — symbol scan may be incomplete"
            )

    def test_no_additional_assessment_automation_symbols(self):
        """Belt-and-suspenders: also check the symbols from the broader spec."""
        for symbol in ("autograde", "auto_correct", "bypass_review", "skip_review", "auto_rate"):
            hits = _grep_in_sources(re.escape(symbol), flags=re.IGNORECASE)
            assert not hits, (
                f"'{symbol}' found in learner source:\n" + "\n".join(hits)
            )

    def test_review_view_prompt_rating_present(self):
        """review_view.py must solicit a rating from the user, not auto-assign it."""
        rv_src = (_LEARNER_PKG / "review_view.py").read_text(encoding="utf-8")
        assert "_prompt_rating" in rv_src, (
            "_prompt_rating missing from review_view.py — user must explicitly rate cards"
        )

    def test_review_view_prompt_answer_present(self):
        """review_view.py must solicit answers from the user, not auto-answer."""
        rv_src = (_LEARNER_PKG / "review_view.py").read_text(encoding="utf-8")
        assert "_prompt_answer" in rv_src, (
            "_prompt_answer missing from review_view.py — user must explicitly answer questions"
        )

    def test_review_view_uses_builtin_input(self):
        """Ratings and answers must come from real user input(), not a pre-computed value."""
        rv_src = (_LEARNER_PKG / "review_view.py").read_text(encoding="utf-8")
        assert "input(" in rv_src, "builtin input() not found in review_view.py"

    def test_review_view_does_not_import_any_llm_backend(self):
        """Answer quality must come from the user, never from an LLM completion."""
        rv_src = (_LEARNER_PKG / "review_view.py").read_text(encoding="utf-8")
        for sym in ("MockLLM", "ClaudeCliBackend", "LLMBackend", "MockBackend"):
            assert sym not in rv_src, (
                f"review_view.py references '{sym}' — answers must not be LLM-generated"
            )
