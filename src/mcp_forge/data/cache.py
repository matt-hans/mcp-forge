"""Simple JSON-backed cache for API responses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ResponseCache:
    """File-backed cache for JSON-serializable responses."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, Any] | None = None

    def _load(self) -> None:
        if self._data is not None:
            return
        if not self.path.exists():
            self._data = {}
            return
        try:
            with open(self.path) as f:
                self._data = json.load(f)
        except json.JSONDecodeError:
            self._data = {}

    def get(self, key: str) -> Any | None:
        """Return cached value for key, if present."""
        self._load()
        return self._data.get(key) if self._data is not None else None

    def set(self, key: str, value: Any) -> None:
        """Set cache value and persist to disk."""
        self._load()
        if self._data is None:
            self._data = {}
        self._data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(self._data, f, indent=2)
        temp_path.replace(self.path)
