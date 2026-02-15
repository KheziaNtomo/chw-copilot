"""Minimal I/O helpers used by the notebook and CLI.

Keep functions tiny so notebook can import and reuse them.
"""
import json
import csv
from typing import Iterable, Mapping


def save_jsonl(path: str, records: Iterable[Mapping]) -> None:
	with open(path, "w", encoding="utf8") as f:
		for r in records:
			f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_csv(path: str, rows, fieldnames=None):
	with open(path, "w", newline="", encoding="utf8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames or (rows[0].keys() if rows else []))
		writer.writeheader()
		writer.writerows(rows)
