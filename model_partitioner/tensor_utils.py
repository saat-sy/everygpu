"""Stateless hashing helpers shared by partitioning, loading and verification."""
import hashlib
from pathlib import Path

import torch

from model_partitioner import constants


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(constants.HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def numel(shape) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count
