#!/usr/bin/env bash
# Poll /health until the server is ready (it returns 503 while loading the model, curl -f fails, keep waiting).
# Usage: scripts/wait_health.sh [timeout_s=180]
# Timing starts when this script starts, so run it immediately after start_server.sh for the cold-start seconds to be accurate (whole-second precision).
# On success: prints the cold-start wait in seconds to stdout, exit code 0; on timeout: error to stderr, exit code 1.
# Can be overridden via environment variables: LLAMA_URL (default http://127.0.0.1:8080)
T=${1:-180}; URL=${LLAMA_URL:-http://127.0.0.1:8080}; S=$(date +%s)
until curl -sf --max-time 5 "$URL/health" >/dev/null; do
  [ $(( $(date +%s) - S )) -ge "$T" ] && { echo "health timeout after ${T}s" >&2; exit 1; }
  sleep 0.5
done
echo $(( $(date +%s) - S ))
