"""Manifest retrieval for a pinned artifact revision."""

import json
from pathlib import Path

import huggingface_hub

from model_partitioner.utils import constants as partitioner_constants
from model_partitioner.utils import models


def fetch_manifest(repo: str, revision: str) -> models.Manifest:
    path = huggingface_hub.hf_hub_download(
        repo, partitioner_constants.MANIFEST_NAME, revision=revision
    )
    return models.Manifest.from_dict(json.loads(Path(path).read_text()))
