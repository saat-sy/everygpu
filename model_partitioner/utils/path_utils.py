"""Stdlib-only path and disk helpers shared before dependencies are installed."""

import os
import shutil
from pathlib import Path


def existing_ancestor(path: Path) -> Path:
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def free_bytes(path: Path) -> int:
    """Free space on the filesystem that holds (or would hold) `path`."""
    return shutil.disk_usage(existing_ancestor(path)).free


def device_of(path: Path) -> int:
    return os.stat(existing_ancestor(path)).st_dev


def overlaps(first: Path, second: Path) -> bool:
    """True if the resolved paths are equal or one contains the other."""
    first, second = Path(first).resolve(), Path(second).resolve()
    return first == second or first in second.parents or second in first.parents


def is_writable(path: Path) -> bool:
    return os.access(existing_ancestor(path), os.W_OK | os.X_OK)


def require_empty_dir(path: Path) -> None:
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"{path} exists and is not an empty directory")
