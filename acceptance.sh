#!/usr/bin/env bash
# End-to-end acceptance smoke test.
# Fill in concrete assertions once ingest/review/stats are implemented.
set -euo pipefail

learner --help
# learner ingest sample_material.txt
# learner review --non-interactive
# learner stats
echo "acceptance: OK"
