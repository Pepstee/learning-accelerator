#!/usr/bin/env bash
# End-to-end acceptance smoke test. Uses LEARNER_MOCK=1 to avoid real LLM calls.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(mktemp -d)"
trap 'rm -rf "$DATA_DIR"' EXIT

export LEARNER_MOCK=1
LEARNER="python -m learner"

# --- ingest ---
echo "=== ingest ==="
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

# Seed a session record so analytics / study-plan produce real data.
mkdir -p "$DATA_DIR/sessions"
cat > "$DATA_DIR/sessions/20260612T000000.json" <<'SESSIONEOF'
{
  "session_id": "acceptance-test-session",
  "started_at": "2026-06-12T00:00:00",
  "duration_seconds": 60.0,
  "ratings": [
    {"card_front": "What is spaced repetition?", "topic": "sample_material", "quality": 2}
  ]
}
SESSIONEOF

# --- questions ---
echo ""
echo "=== questions ==="
QUESTIONS_OUT=$($LEARNER --data-dir "$DATA_DIR" questions sample_material)
echo "$QUESTIONS_OUT"
if ! echo "$QUESTIONS_OUT" | grep -q 'Q:'; then
    echo "FAIL: No Q: line in questions output" >&2; exit 1
fi

# --- flashcards ---
echo ""
echo "=== flashcards ==="
FLASHCARDS_OUT=$($LEARNER --data-dir "$DATA_DIR" flashcards sample_material)
echo "$FLASHCARDS_OUT"
if ! echo "$FLASHCARDS_OUT" | grep -q 'Q:'; then
    echo "FAIL: No Q/A pairs in flashcards output" >&2; exit 1
fi

# --- summary ---
echo ""
echo "=== summary ==="
SUMMARY_OUT=$($LEARNER --data-dir "$DATA_DIR" summary sample_material)
echo "$SUMMARY_OUT"
if [[ -z "$SUMMARY_OUT" ]]; then
    echo "FAIL: summary output is empty" >&2; exit 1
fi

# --- analytics ---
echo ""
echo "=== analytics ==="
ANALYTICS_OUT=$($LEARNER --data-dir "$DATA_DIR" analytics)
echo "$ANALYTICS_OUT"
if ! echo "$ANALYTICS_OUT" | grep -q 'Sessions:'; then
    echo "FAIL: No 'Sessions:' line in analytics output" >&2; exit 1
fi

# --- study-plan ---
echo ""
echo "=== study-plan ==="
STUDYPLAN_OUT=$($LEARNER --data-dir "$DATA_DIR" study-plan)
echo "$STUDYPLAN_OUT"
if [[ -z "$STUDYPLAN_OUT" ]]; then
    echo "FAIL: study-plan output is empty" >&2; exit 1
fi

echo ""
echo "acceptance: OK"
