#!/usr/bin/env bash
# usage: sampler.sh PID OUT.csv  -- every ~1s record t, rss_kb, footprint_mb (a row is also written on error)
PID=$1; OUT=$2; echo "t,rss_kb,footprint_mb,err" > "$OUT"
while kill -0 $PID 2>/dev/null; do
  T=$(python3 -c 'import time;print(f"{time.time():.3f}")')
  R=$(ps -o rss= -p $PID | tr -d ' ')
  F=$(footprint $PID 2>/dev/null | awk '/Footprint:/{print $5}')
  if [ -z "$R" ] || [ -z "$F" ]; then echo "$T,${R},${F},sample_failed" >> "$OUT"; else echo "$T,$R,$F," >> "$OUT"; fi
  sleep 0.5
done
