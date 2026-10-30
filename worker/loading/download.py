"""Space-checked, manifest-verified download of artifact files."""

from pathlib import Path

import huggingface_hub
import huggingface_hub.constants

from model_partitioner.utils import constants as partitioner_constants
from model_partitioner.utils import models, tensor_utils


def download_verified(
    repo: str, revision: str, manifest: models.Manifest, files: list[str]
) -> Path:
    """Download whole files, refuse if the cache cannot hold them, check each sha256."""
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
    print(f"verified {len(files)} files")
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
    print(
        f"download: {len(missing)} of {len(files)} files, {size:,} bytes "
        f"({free:,} free in the Hugging Face cache)"
    )
    if free < size:
        raise ValueError(f"download needs {size:,} bytes but {free:,} are free")
