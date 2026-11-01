"""JSONL log writing: one record per line, flushed to disk as soon as each line is written."""
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
