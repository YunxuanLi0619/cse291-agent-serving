#!/usr/bin/env bash
# Start llama-server; all fixed parameters are kept here so that every start is identical.
# Usage: scripts/start_server.sh <np> <ctx> [extra flags...]
# The thread count is read from configs/server.json (the value frozen in #7; 8 if the file is missing).
# Can be overridden via environment variables: LLAMA_SERVER_BIN, MODEL, SERVER_CONFIG
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: $0 <np> <ctx> [extra flags...]" >&2; exit 2; }
NP=$1; CTX=$2; shift 2
BIN=${LLAMA_SERVER_BIN:-$HOME/src/llama.cpp/build/bin/llama-server}
MODEL=${MODEL:-$HOME/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf}
CONFIG=${SERVER_CONFIG:-$(dirname "$0")/../configs/server.json}
THREADS=8
[ -f "$CONFIG" ] && THREADS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["threads"])' "$CONFIG")
exec "$BIN" -m "$MODEL" --host 127.0.0.1 --port 8080 \
  -np "$NP" -c "$CTX" -t "$THREADS" --cache-ram 0 --metrics "$@"
