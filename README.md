# learner

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
| `summary <topic>` | Print the summary for an ingested topic |
| `flashcards <topic>` | List all flashcards for a topic |
| `practice` | Interactive SRS review session |
| `exam <topic>` | Multiple-choice quiz for a topic |
| `study-plan` | Generate a study plan based on weak areas |
| `analytics` | Show study analytics and weak areas |

## Example invocations

```sh
# Ingest a local text file (uses mock backend for offline testing)
learner --mock ingest notes.txt

# Print the summary for an ingested topic
learner summary notes

# List flashcards for a topic
learner flashcards notes

# Start an interactive SRS review session
learner practice

# Run a multiple-choice exam for a topic
learner exam notes

# Show analytics and weak areas
learner analytics

# Generate a study plan based on weak areas
learner study-plan
```

## Architecture

- `learner/models.py` — core dataclasses (`Flashcard`/`Card`, `Question`, `WeakArea`, `StudyPlan`, `SessionRecord`)
- `learner/llm.py` — `LLMBackend` ABC; `ClaudeCliBackend` (production); `MockLLM`/`MockBackend` (testing)
- `learner/generator.py` — `generate_summary`, `generate_flashcards`, `generate_questions` (LLM-backed)
- `learner/analytics.py` — `compute_weak_areas`, `build_analytics_report`, `generate_study_plan`
- `learner/review_view.py` — `ReviewView`: interactive flashcard + question review session
- `learner/session.py` — `ReviewSession` (SRS state + session history); `load_session_history`
- `learner/srs.py` — SM-2 spaced-repetition scheduling
- `learner/cli.py` — `argparse`-based CLI wired to all subcommand handlers
