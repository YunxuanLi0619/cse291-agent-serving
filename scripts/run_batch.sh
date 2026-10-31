#!/usr/bin/env bash
# 批跑守护:python 异常退出就重启(已完成的 Run 会被跳过),最多 10 次;有 Run 失败时多跑一遍重试,共 2 遍。
# 用法: caffeinate -is scripts/run_batch.sh [run_batch.py 的参数]
cd "$(dirname "$0")/.." || exit 2
passes=0; crashes=0
while true; do
  .venv/bin/python scripts/run_batch.py "$@"; rc=$?
  case $rc in
    0) exit 0 ;;
    2|3|4) exit $rc ;;  # 参数错 / 未插电 / 预检不过:要人处理,重启没用
    1) passes=$((passes + 1)); [ $passes -ge 2 ] && exit 1 ;;
    *) crashes=$((crashes + 1)); [ $crashes -ge 10 ] && exit $rc; sleep 60 ;;
  esac
done
