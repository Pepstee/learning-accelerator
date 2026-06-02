# learner

An AI-powered spaced-repetition learning accelerator.  Feed it source material
(text, PDFs, URLs) and it generates flashcards and multiple-choice questions,
tracks your performance, and surfaces your weak areas so you study smarter.

## Install

```sh
pip install -e .
learner --help
```

## Subcommands

| Command | Description |
|---------|-------------|
| `ingest <source>` | Parse material and generate cards/questions |
| `review` | Interactive review session (spaced-repetition) |
| `stats` | Show progress and weak areas |

## Example invocations

```sh
# Ingest a local PDF
learner ingest notes/chapter1.pdf

# Ingest from a URL
learner ingest https://example.com/article

# Start a review session
learner review

# Print statistics
learner stats
```

## Architecture

- `learner/models.py` — core dataclasses (`Card`, `Question`, `StudySession`, `WeakArea`)
- `learner/llm.py` — `LLMBackend` Protocol; `ClaudeCliBackend` (production); `MockBackend` (testing)
- `learner/cli.py` — `argparse`-based CLI wired to subcommand handlers
