from __future__ import annotations

import json
from pathlib import Path


def save_json(data: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))
