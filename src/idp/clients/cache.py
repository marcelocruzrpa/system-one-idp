"""Disk cache for model responses, so repeated eval runs and UI replays don't pay twice."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from idp.config import CACHE_DIR


def _default(value: Any) -> Any:
    if isinstance(value, bytes | bytearray | memoryview):
        return "sha256:" + hashlib.sha256(bytes(value)).hexdigest()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Unserializable cache key part: {type(value)!r}")


class ResponseCache:
    def __init__(self, namespace: str, enabled: bool = True, directory: Path | None = None) -> None:
        self.dir = (directory or CACHE_DIR) / namespace
        self.enabled = enabled

    def key(self, request: Any) -> str:
        blob = json.dumps(request, sort_keys=True, default=_default, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Any | None:
        """The cached value, or None when there is none or the entry is unreadable (it is then deleted)."""
        path = self._path(key)
        if not self.enabled or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A truncated or corrupt entry (an interrupted run, a disk hiccup) is dropped so the next call goes live.
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def put(self, key: str, value: Any) -> None:
        """Write atomically: a reader (or a parallel eval worker) never sees a half-written entry."""
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        # A private temp file per writer: two workers caching the same key must not truncate each other's file.
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=f"{key[:12]}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(value, ensure_ascii=False))
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
