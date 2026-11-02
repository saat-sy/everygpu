"""Hugging Face Hub access: pinned source download, upload and upload verification."""

import contextlib
from pathlib import Path

import huggingface_hub
import huggingface_hub.constants
import huggingface_hub.errors

from model_partitioner.utils import constants, models, tensor_utils
from model_partitioner.artifact import loader

_ORG_WRITERS = frozenset({"write", "contributor", "admin"})
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
    manifest_path = Path(artifact_dir) / constants.MANIFEST_NAME
    expected = {
        n: (i.size, i.sha256) for n, i in loader.declared_files(manifest).items()
    }
    expected[constants.MANIFEST_NAME] = (
        manifest_path.stat().st_size,
        tensor_utils.file_sha256(manifest_path),
    )
    with _hub_errors(f"cannot list {repo_id}@{revision}"):
        info = huggingface_hub.HfApi().model_info(
            repo_id, revision=revision, files_metadata=True
        )
    sizes = {s.rfilename: s.size for s in info.siblings}
    lfs = {s.rfilename: s.lfs.sha256 for s in info.siblings if getattr(s, "lfs", None)}
    problems = []
    for name, (size, digest) in expected.items():
        if name not in sizes:
            problems.append(f"{name}: not on Hub at {revision}")
            continue
        if sizes[name] != size:
            problems.append(f"{name}: size {sizes[name]} != manifest {size}")
            continue
        with _hub_errors(f"cannot download {name}"):
            remote = lfs.get(name) or tensor_utils.file_sha256(
                Path(huggingface_hub.hf_hub_download(repo_id, name, revision=revision))
            )
        if remote != digest:
            problems.append(f"{name}: sha256 differs")
    return problems


def cache_dir() -> str:
    return huggingface_hub.constants.HF_HUB_CACHE


def missing_download_bytes(
    repo_id: str, revision: str, files: dict[str, models.FileInfo]
) -> int:
    """Bytes of declared weight files not already in the Hugging Face cache."""
    return sum(
        info.size
        for name, info in files.items()
        if name.endswith(".safetensors")
        and not isinstance(
            huggingface_hub.try_to_load_from_cache(repo_id, name, revision=revision),
            str,
        )
    )


def check_upload_access(repo_id: str) -> str:
    """Raise ValueError unless the token can write (existing) or plausibly create `repo_id`."""
    token = huggingface_hub.get_token()
    if not token:
        raise ValueError(
            "no Hugging Face token: run `hf auth login` or set HF_TOKEN (write access)"
        )
    api = huggingface_hub.HfApi()
    with _hub_errors(
        "Hugging Face token rejected; run `hf auth login` or set HF_TOKEN"
    ):
        who = api.whoami(token=token, cache=False)
    with _hub_errors(f"cannot check {repo_id}"):
        exists = api.repo_exists(repo_id, repo_type="model")
    if exists:
        with _hub_errors(f"token cannot write to {repo_id}"):
            api.auth_check(repo_id, repo_type="model", write=True)
        return f"{who['name']} may write to existing {repo_id} (visibility unchanged)"
    namespace = repo_id.split("/")[0]
    orgs = {org["name"]: org.get("roleInOrg") for org in who.get("orgs", [])}
    role = who.get("auth", {}).get("accessToken", {}).get("role")
    if role == "read":
        raise ValueError("token is read-only; a write token is required")
    if namespace != who["name"] and orgs.get(namespace) not in _ORG_WRITERS:
        raise ValueError(f"{who['name']} cannot create repos under {namespace!r}")
    return (
        f"{who['name']} may create private {repo_id} (token role {role}); "
        "create permission is confirmed only at upload"
    )
