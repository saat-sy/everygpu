"""Manifest retrieval for a pinned artifact revision."""

from pathlib import Path

import huggingface_hub

from model_partitioner.artifact import loader
from model_partitioner.utils import constants as partitioner_constants
from model_partitioner.utils import models


def fetch_manifest(repo: str, revision: str) -> models.Manifest:
    path = huggingface_hub.hf_hub_download(
        repo, partitioner_constants.MANIFEST_NAME, revision=revision
    )
    return loader.read_manifest(Path(path).parent)
