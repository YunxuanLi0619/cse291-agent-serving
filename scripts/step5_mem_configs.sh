#!/usr/bin/env bash
# Step 5.3/5.4: 对每种 server 配置记录 KV buffer、/slots 的 n_ctx、空闲 RSS 与 footprint。
# 用法: scripts/step5_mem_configs.sh   (输出到 results/raw/step5/)
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/raw/step5; mkdir -p "$OUT"
BIN=~/src/llama.cpp/build/bin/llama-server; MODEL=~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
snap() { # label pid
  local rss fp; rss=$(( $(ps -o rss= -p "$2") / 1024 )); fp=$(footprint "$2" 2>/dev/null | awk '/Footprint:/{print $5}')
  echo "$1,rss_mb=$rss,footprint_mb=$fp" | tee -a "$OUT/summary.csv"
}
run() { # name, args...
  local name=$1; shift
  $BIN -m $MODEL --host 127.0.0.1 --port 8080 -t 8 --cache-ram 0 --metrics -lv 4 "$@" > "$OUT/$name.server.log" 2>&1 &
  local pid=$!; local s=$(date +%s)
  until curl -sf http://127.0.0.1:8080/health >/dev/null; do sleep 0.5; kill -0 $pid 2>/dev/null || { echo "$name: server died" | tee -a "$OUT/summary.csv"; return; }; done
  echo "$name,args=$*,cold_start_s=$(( $(date +%s) - s ))" | tee -a "$OUT/summary.csv"
  curl -s http://127.0.0.1:8080/slots | jq -c '{n_slots:length, n_ctx:[.[].n_ctx]}' | tee -a "$OUT/summary.csv"
  grep -E "llama_kv_cache: size|KV buffer" "$OUT/$name.server.log" | cut -c1-200 | tee -a "$OUT/summary.csv"
  snap "$name,idle_t0" $pid
  curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Say hi."}],"max_tokens":16,"temperature":0}' -o /dev/null
  sleep 20; snap "$name,idle_after_1req_20s" $pid
  kill $pid; wait $pid 2>/dev/null; sleep 3
}
echo "== $(date) ==" >> "$OUT/summary.csv"
run A_mmap      -np 4 -c 16384 --no-kv-unified
run A_nommap    -np 4 -c 16384 --no-kv-unified -lm none
run B_np8_c32k  -np 8 -c 32768 --no-kv-unified
run C_kvu_np8   -np 8 -c 16384 -kvu
