"""Inverse-visit-count level sampling with cross-process shared state."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha1
from pathlib import Path
from typing import IO

if os.name == "nt":
    import msvcrt
else:
    import fcntl

LEVEL_LOADER_NAMESPACE_ENV = "LEVEL_LOADER_NAMESPACE"
# Import-time side effect (load-bearing): the first process to import this
# module names the shared-counts namespace. Spawned vector-env workers inherit
# it through the environment, so all workers of one training run share one
# counts file while separate runs stay isolated.
os.environ.setdefault(LEVEL_LOADER_NAMESPACE_ENV, f"{os.getpid()}-{time.time_ns()}")

_SHARED_FILE_PREFIX = "pcgrl_level_loader"
_DIGEST_CHARS = 12
_LOCK_POLL_INTERVAL_S = 0.01
_LOCK_TIMEOUT_S = 30.0


def _lock_file(handle: IO[str]) -> None:
    """Acquire an exclusive lock on ``handle`` (blocking, cross-platform).
    Windows polls a 1-byte ``msvcrt`` lock, raising ``TimeoutError`` after
    ``_LOCK_TIMEOUT_S``; POSIX uses a blocking ``flock``."""
    if os.name == "nt":
        handle.seek(0)
        handle.write(" ")
        handle.flush()
        handle.seek(0)
        deadline = time.monotonic() + _LOCK_TIMEOUT_S
        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for level-loader lock: {handle.name}"
                    ) from exc
                time.sleep(_LOCK_POLL_INTERVAL_S)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: IO[str]) -> None:
    """Release the lock taken by :func:`_lock_file`."""
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive cross-process lock on ``lock_path`` for the block."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as handle:
        _lock_file(handle)
        try:
            yield
        finally:
            _unlock_file(handle)


class LevelSampler:
    """Samples levels with weight ``1 / (visits + 1)``, shared across processes.
    Visit counts live in a lock-guarded JSON file in the OS temp dir (keyed by
    level-dir digest + run namespace); sampling uses the global ``random`` module."""

    def __init__(self, level_dir: str, namespace: str | None = None):
        level_path = Path(level_dir)
        lvl_files = list(level_path.glob("*.lvl"))
        txt_files = list(level_path.glob("*.txt"))
        self.level_files = [str(f.resolve()) for f in sorted(lvl_files + txt_files)]
        if not self.level_files:
            raise ValueError(f"No level files found in {level_path.resolve()}")
        self._shared_namespace = self._resolve_namespace(namespace)
        digest = sha1(str(level_path.resolve()).encode("utf-8")).hexdigest()[:_DIGEST_CHARS]
        shared_name = f"{_SHARED_FILE_PREFIX}_{digest}_{self._shared_namespace}"
        shared_dir = Path(tempfile.gettempdir())
        self._counts_path = shared_dir / f"{shared_name}.json"
        self._lock_path = shared_dir / f"{shared_name}.lock"
        self._sync_shared_counts()

    def select_level(self) -> str:
        """Sample one level path with inverse-visit weights and record the visit."""
        with _file_lock(self._lock_path):
            counts = self._load_counts_unlocked()
            weights = [1.0 / (counts.get(level, 0) + 1.0) for level in self.level_files]
            level = random.choices(self.level_files, weights=weights, k=1)[0]
            counts[level] = int(counts.get(level, 0)) + 1
            self._save_counts_unlocked(counts)
        return level

    def read_level(self, level_file: str) -> str:
        """Return the UTF-8 text of ``level_file`` (path resolved first)."""
        with open(Path(level_file).resolve(), "r", encoding="utf-8") as handle:
            return handle.read()

    def set_level_files(self, level_files: list[str]) -> None:
        """Replace the eligible pool with the existing files among ``level_files``.
        New files register with zero visits; old counts are kept, never pruned.
        Raises ``ValueError`` if none of the given paths exist on disk."""
        self.level_files = [
            str(Path(f).resolve()) for f in level_files if Path(f).resolve().exists()
        ]
        if not self.level_files:
            raise ValueError("set_level_files received no existing level files")
        self._sync_shared_counts()

    def _sync_shared_counts(self) -> None:
        with _file_lock(self._lock_path):
            counts = self._load_counts_unlocked()
            for level in self.level_files:
                counts.setdefault(level, 0)
            self._save_counts_unlocked(counts)

    def _load_counts_unlocked(self) -> dict[str, int]:
        if not self._counts_path.exists():
            return {level: 0 for level in self.level_files}
        try:
            with open(self._counts_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {level: 0 for level in self.level_files}
        counts = {str(level): int(count) for level, count in data.items()}
        for level in self.level_files:
            counts.setdefault(level, 0)
        return counts

    def _save_counts_unlocked(self, counts: dict[str, int]) -> None:
        self._counts_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._counts_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(counts, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, self._counts_path)

    @staticmethod
    def _resolve_namespace(namespace: str | None) -> str:
        if namespace:
            return namespace
        env_namespace = os.environ.get(LEVEL_LOADER_NAMESPACE_ENV)
        if env_namespace:
            return env_namespace
        process = mp.current_process()
        return str(os.getppid() if process.name != "MainProcess" else os.getpid())
