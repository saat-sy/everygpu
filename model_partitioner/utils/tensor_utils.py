"""Stateless hashing and directory helpers shared across the partitioner."""

import hashlib
import shutil
from pathlib import Path

import torch

from model_partitioner.utils import constants


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(constants.HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def numel(shape) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def free_bytes(path: Path) -> int:
    """Free space on the filesystem that holds (or would hold) `path`."""
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def require_empty_dir(path: Path) -> None:
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"{path} exists and is not an empty directory")
