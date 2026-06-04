from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Any

from .schema import Case, SchemaError


def load_cases_jsonl(path: str | Path) -> list[Case]:
    p = Path(path)
    cases: list[Case] = []
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            cases.append(Case.from_dict(obj))
        except (json.JSONDecodeError, SchemaError) as exc:
            raise SchemaError(f"{p}:{line_no}: {exc}") from exc
    return cases


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
