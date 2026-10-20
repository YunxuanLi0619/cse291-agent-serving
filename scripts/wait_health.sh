#!/usr/bin/env bash
# 轮询 /health 直到 server 就绪(加载模型期间返回 503,curl -f 会失败,继续等)。
# 用法: scripts/wait_health.sh [timeout_s=180]
# 计时从本脚本启动算起,所以要在 start_server.sh 之后立即运行,冷启动秒数才准(整秒精度)。
# 成功: stdout 打印冷启动等待秒数,退出码 0;超时: stderr 报错,退出码 1。
# 可用环境变量覆盖: LLAMA_URL (默认 http://127.0.0.1:8080)
T=${1:-180}; URL=${LLAMA_URL:-http://127.0.0.1:8080}; S=$(date +%s)
until curl -sf --max-time 5 "$URL/health" >/dev/null; do
  [ $(( $(date +%s) - S )) -ge "$T" ] && { echo "health timeout after ${T}s" >&2; exit 1; }
  sleep 0.5
done
echo $(( $(date +%s) - S ))
