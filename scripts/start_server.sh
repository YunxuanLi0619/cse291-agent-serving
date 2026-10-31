#!/usr/bin/env bash
# 启动 llama-server;所有固定参数集中在这里,保证每次一致。
# 用法: scripts/start_server.sh <np> <ctx> [extra flags...]
# 线程数读 configs/server.json(#7 冻结的值;缺文件则 8)。
# 可用环境变量覆盖: LLAMA_SERVER_BIN, MODEL, SERVER_CONFIG
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
