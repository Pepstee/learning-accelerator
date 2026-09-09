# learner

## ArtVault installation

On this ArtVault host the project workspace is
`/srv/artvault/projects/learning-accelerator/workspace`. From that directory,
run the installed tool with:

```sh
../.venv/bin/learner --help
```

The reconciled build passed 1019 tests on both Mac and Linux. Offline
acceptance testing uses a labelled mock backend (`--mock`); production AI
generation still requires Claude CLI installation and authentication. Always
pass an explicit `--data-dir` for learning data, since personal learning data
was excluded from this workspace.

An AI-powered spaced-repetition learning accelerator.  Feed it source material
and it generates flashcards and multiple-choice questions, tracks your
performance, and surfaces your weak areas so you study smarter.

## Install

```sh
pip install -e .
learner --help
```

## Global flags

| Flag | Description |
|------|-------------|
| `--data-dir DIR` | Data directory (default: `~/.learner`) |
| `--mock` | Use mock LLM backend for offline testing |

## Subcommands

| Command | Description |
|---------|-------------|
| `ingest <source>` | Ingest source material and generate cards/questions |
| `generate <source>` | Generate flashcards and questions from source material |
| `questions <topic>` | List questions and their answers for a topic |
| `summary <topic>` | Print the summary for an ingested topic |
| `flashcards <topic>` | List all flashcards for a topic |
| `practice` | Interactive SRS review session |
| `review` | Interactive flashcard and question review session |
| `exam <topic>` | Multiple-choice quiz for a topic |
| `study-plan` | Generate a study plan based on weak areas |
| `analytics` | Show study analytics and weak areas |

## Example invocations

```sh
# Ingest a local text file (uses mock backend for offline testing)
learner --mock ingest notes.txt

# Generate flashcards and questions from source material
learner --mock generate notes.txt

# List questions and their answers for a topic
learner questions notes

# Print the summary for an ingested topic
learner summary notes

# List flashcards for a topic
learner flashcards notes

# Start an interactive SRS review session
learner practice

# Start an interactive review session (flashcards + questions)
learner review

# Run a multiple-choice exam for a topic
learner exam notes

# Show analytics and weak areas
learner analytics

# Generate a study plan based on weak areas without calling the production backend
learner --mock study-plan
```

`ingest` and `generate` currently accept local UTF-8 text files. Without
`--mock` (or `LEARNER_MOCK=1`), generation and non-empty study plans use the
configured production Claude CLI backend.

## Architecture

- `learner/models.py` — core dataclasses (`Flashcard`/`Card`, `Question`, `WeakArea`, `StudyPlan`, `SessionRecord`)
- `learner/llm.py` — `LLMBackend` ABC; `ClaudeCliBackend` (production); `MockLLM`/`MockBackend` (testing)
- `learner/generator.py` — `generate_summary`, `generate_flashcards`, `generate_questions` (LLM-backed)
- `learner/analytics.py` — `compute_weak_areas`, `build_analytics_report`, `generate_study_plan`
- `learner/review_view.py` — `ReviewView`: interactive flashcard + question review session
- `learner/session.py` — `ReviewSession` (SRS state + session history); `load_session_history`
- `learner/srs.py` — SM-2 spaced-repetition scheduling
- `learner/cli.py` — `argparse`-based CLI wired to all subcommand handlers

## Migration scope

The canonical baseline `bc2ca0ff` already contains the capability union. Both
registered donor worktrees share `699a19b`; their named Python capabilities are
retained, including the `Card` and `MockBackend` compatibility aliases. Structured
ingestion, generated study plans, analytics and due-card review evidence are
connected to the current workflow. The study-plan library now returns a structured
`StudyPlan`; callers read `.advice` instead of expecting the historical plain string.
Old compiled bytecode is intentionally omitted.

The deterministic `graphify-out` index covers tracked Python code; document
semantics are not indexed. User learning data under `~/.learner` is excluded from
source migration. Use an explicit `--data-dir` for test or migrated datasets.
The offline acceptance workflow uses a labelled mock backend. Production
content generation still requires the configured Claude CLI and authentication;
that provider is not installed or verified on ArtVault by this source migration.

Untracked donor test variants were compared with the canonical suites. Their
production behaviour is already covered; generic dataclass assertions and
serialisation implemented only inside donor tests are intentionally not copied.
