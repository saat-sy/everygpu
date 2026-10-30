"""Hugging Face Hub access: pinned source download, upload and upload verification."""

import contextlib
from pathlib import Path

import huggingface_hub
import huggingface_hub.errors

from model_partitioner.utils import constants, models, tensor_utils
from model_partitioner.artifact import loader

_SOURCE_PATTERNS = ("*.safetensors", constants.INDEX_NAME, *constants.ASSET_PATTERNS)


@contextlib.contextmanager
def _hub_errors(what: str):
    try:
        yield
    except (huggingface_hub.errors.HfHubHTTPError, OSError) as error:
        raise ValueError(f"{what}: {error}") from error


def resolve_revision(repo_id: str) -> str:
    """The repo's current immutable commit SHA."""
    with _hub_errors(f"cannot resolve {repo_id}"):
        return huggingface_hub.HfApi().model_info(repo_id).sha


def fetch_config(repo_id: str, revision: str) -> Path:
    with _hub_errors(f"cannot download config of {repo_id}@{revision}"):
        return Path(
            huggingface_hub.hf_hub_download(repo_id, "config.json", revision=revision)
        )


def fetch_source(repo_id: str, revision: str) -> Path:
    """Weights and assets at exactly `revision`, via the Hugging Face cache."""
    with _hub_errors(f"cannot download {repo_id}@{revision}"):
        return Path(
            huggingface_hub.snapshot_download(
                repo_id, revision=revision, allow_patterns=list(_SOURCE_PATTERNS)
            )
        )


def remote_files(repo_id: str, revision: str) -> dict[str, models.FileInfo]:
    """Size and sha256 the Hub reports for each LFS file at `revision`."""
    with _hub_errors(f"cannot read file hashes of {repo_id}@{revision}"):
        info = huggingface_hub.HfApi().model_info(
            repo_id, revision=revision, files_metadata=True
        )
    return {
        s.rfilename: models.FileInfo(s.lfs.size, s.lfs.sha256)
        for s in info.siblings
        if getattr(s, "lfs", None)
    }


def ensure_repo(repo_id: str) -> bool:
    """Create a private model repo if missing; True if created, else untouched."""
    with _hub_errors(f"cannot access {repo_id}"):
        api = huggingface_hub.HfApi()
        if api.repo_exists(repo_id):
            return False
        api.create_repo(repo_id, private=True)
        return True


def upload_artifact(artifact_dir: Path, repo_id: str) -> str:
    """Upload only the files the manifest declares; return the resulting revision."""
    manifest = loader.read_manifest(artifact_dir)
    patterns = [constants.MANIFEST_NAME, *loader.declared_files(manifest)]
    with _hub_errors(f"cannot upload to {repo_id}"):
        api = huggingface_hub.HfApi()
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(artifact_dir),
            allow_patterns=patterns,
        )
        return api.model_info(repo_id).sha


def check_upload(artifact_dir: Path, repo_id: str, revision: str) -> list[str]:
    """Problems found comparing every declared file at `revision` with the manifest."""
    manifest = loader.read_manifest(artifact_dir)
    expected = {n: i.sha256 for n, i in loader.declared_files(manifest).items()}
    expected[constants.MANIFEST_NAME] = tensor_utils.file_sha256(
        Path(artifact_dir) / constants.MANIFEST_NAME
    )
    with _hub_errors(f"cannot list {repo_id}@{revision}"):
        present = set(huggingface_hub.list_repo_files(repo_id, revision=revision))
    lfs = {name: info.sha256 for name, info in remote_files(repo_id, revision).items()}
    problems = []
    for name, digest in expected.items():
        if name not in present:
            problems.append(f"{name}: not on Hub at {revision}")
            continue
        with _hub_errors(f"cannot download {name}"):
            remote = lfs.get(name) or tensor_utils.file_sha256(
                Path(huggingface_hub.hf_hub_download(repo_id, name, revision=revision))
            )
        if remote != digest:
            problems.append(f"{name}: sha256 differs")
    return problems
