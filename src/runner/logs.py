"""JSONL 日志写入:一行一条,每行写完立即落盘。"""
import json
from pathlib import Path


class JsonlWriter:
    def __init__(self, path: Path):
        self._f = open(path, "w", encoding="utf-8")

    def write(self, row: dict) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()
