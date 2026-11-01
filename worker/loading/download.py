"""Space-checked, manifest-verified download of artifact files."""

import logging
import time
from pathlib import Path

import huggingface_hub
import huggingface_hub.constants

from model_partitioner.utils import constants as partitioner_constants
from monitoring.logging import logger as log_utils
from model_partitioner.utils import models, tensor_utils

logger = logging.getLogger(__name__)


def download_verified(
    repo: str, revision: str, manifest: models.Manifest, files: list[str]
) -> Path:
    """Download whole files, refuse if the cache cannot hold them, check each sha256."""
    started = time.monotonic()
    _check_space(repo, revision, manifest, files)
    root = Path(
        huggingface_hub.snapshot_download(
            repo,
            revision=revision,
            allow_patterns=[partitioner_constants.MANIFEST_NAME, *files],
        )
    )
    for name in files:
        if tensor_utils.file_sha256(root / name) != manifest.files[name].sha256:
            raise models.ChecksumMismatch(f"{name} differs from the manifest")
    logger.info(
        "download verified files=%d elapsed_ms=%d",
        len(files),
        log_utils.elapsed_ms(started),
    )
    return root


def _check_space(
    repo: str, revision: str, manifest: models.Manifest, files: list[str]
) -> None:
    """Report the download size; refuse if the Hugging Face cache cannot hold it."""
    missing = [
        f
        for f in files
        if not isinstance(
            huggingface_hub.try_to_load_from_cache(repo, f, revision=revision), str
        )
    ]
    size = sum(manifest.files[f].size for f in missing)
    free = tensor_utils.free_bytes(Path(huggingface_hub.constants.HF_HUB_CACHE))
    logger.info(
        "download starting files=%d of %d bytes=%d free_bytes=%d",
        len(missing),
        len(files),
        size,
        free,
    )
    if free < size:
        raise ValueError(f"download needs {size:,} bytes but {free:,} are free")
