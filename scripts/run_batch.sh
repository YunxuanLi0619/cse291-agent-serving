#!/usr/bin/env bash
# Batch supervisor: restart if python exits abnormally (completed Runs are skipped), at most 10 times; if any Run failed, do one more pass to retry, 2 passes in total.
# Usage: caffeinate -is scripts/run_batch.sh [arguments of run_batch.py]
cd "$(dirname "$0")/.." || exit 2
passes=0; crashes=0
while true; do
  .venv/bin/python scripts/run_batch.py "$@"; rc=$?
  case $rc in
    0) exit 0 ;;
    2|3|4) exit $rc ;;  # bad arguments / not on AC / pre-check failed: needs a human, restarting would not help
    1) passes=$((passes + 1)); [ $passes -ge 2 ] && exit 1 ;;
    *) crashes=$((crashes + 1)); [ $crashes -ge 10 ] && exit $rc; sleep 60 ;;
  esac
done
