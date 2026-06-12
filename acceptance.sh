#!/usr/bin/env bash
# End-to-end acceptance smoke test. Uses LEARNER_MOCK=1 to avoid real LLM calls.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(mktemp -d)"
trap 'rm -rf "$DATA_DIR"' EXIT

export LEARNER_MOCK=1
LEARNER="python -m learner"

# --- ingest ---
INGEST_OUT=$($LEARNER --data-dir "$DATA_DIR" ingest "$SCRIPT_DIR/sample_material.txt")
echo "$INGEST_OUT"

SUMMARY=$(echo "$INGEST_OUT" | grep '^Summary:' | sed 's/^Summary:[[:space:]]*//')
if [[ -z "$SUMMARY" ]]; then
    echo "FAIL: Summary line is empty" >&2; exit 1
fi

CARDS=$(echo "$INGEST_OUT" | grep '^Cards:' | awk '{print $2}')
if [[ "$CARDS" -lt 1 ]]; then
    echo "FAIL: Cards count ($CARDS) < 1" >&2; exit 1
fi

QUESTIONS=$(echo "$INGEST_OUT" | grep '^Questions:' | awk '{print $2}')
if [[ "$QUESTIONS" -lt 1 ]]; then
    echo "FAIL: Questions count ($QUESTIONS) < 1" >&2; exit 1
fi

# --- flashcards ---
echo ""
echo "flashcards:"
FLASHCARDS_OUT=$($LEARNER --data-dir "$DATA_DIR" flashcards sample_material)
echo "$FLASHCARDS_OUT"

if ! echo "$FLASHCARDS_OUT" | grep -q 'Q:'; then
    echo "FAIL: No Q/A pairs in flashcards output" >&2; exit 1
fi

echo ""
echo "acceptance: OK"
