"""Hugging Face Hub access: remote headers, upload, retrieval and checksum checks."""
import json
from pathlib import Path

import huggingface_hub

from model_partitioner import constants, loader, models, tensor_utils


def remote_specs(repo_id: str, revision: str) -> list[models.TensorSpec]:
    """Tensor specs read from remote shard headers by range requests."""
    metadata = huggingface_hub.get_safetensors_metadata(repo_id, revision=revision)
    specs = []
    for file, file_meta in metadata.files_metadata.items():
        for key, info in file_meta.tensors.items():
            start, end = info.data_offsets
            specs.append(models.TensorSpec(key, file, tuple(info.shape), info.dtype, end - start))
    return sorted(specs, key=lambda spec: spec.key)


def remote_config(repo_id: str, revision: str) -> dict:
    path = huggingface_hub.hf_hub_download(repo_id, 'config.json', revision=revision)
    return json.loads(Path(path).read_text())


def lfs_hashes(repo_id: str, revision: str | None = None) -> dict[str, str]:
    info = huggingface_hub.HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    return {s.rfilename: s.lfs.sha256 for s in info.siblings if getattr(s, 'lfs', None)}


def upload_artifact(artifact_dir: Path, repo_id: str, private: bool = True) -> str:
    """Upload and return the resulting commit SHA, which pins later verification."""
    api = huggingface_hub.HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_large_folder(repo_id=repo_id, repo_type='model', folder_path=str(artifact_dir))
    return api.model_info(repo_id).sha


def check_upload(artifact_dir: Path, repo_id: str, revision: str) -> list[str]:
    """Problems comparing every declared file at `revision` with the manifest.

    LFS files compare the Hub sha256; other files are downloaded and hashed.
    """
    manifest = loader.read_manifest(artifact_dir)
    expected = {name: info.sha256 for name, info in manifest.files.items()}
    expected.update({f'{constants.ASSETS_DIR}/{name}': info.sha256
                     for name, info in manifest.assets.items()})
    expected[constants.MANIFEST_NAME] = tensor_utils.file_sha256(
        Path(artifact_dir) / constants.MANIFEST_NAME)
    present = set(huggingface_hub.list_repo_files(repo_id, revision=revision))
    lfs = lfs_hashes(repo_id, revision)
    problems = []
    for name, digest in expected.items():
        if name not in present:
            problems.append(f'{name}: not on Hub at {revision}')
            continue
        remote = lfs.get(name) or tensor_utils.file_sha256(Path(
            huggingface_hub.hf_hub_download(repo_id, name, revision=revision)))
        if remote != digest:
            problems.append(f'{name}: sha256 differs')
    return problems


def check_load(repo_id: str, revision: str, unit_id: str, dest: Path) -> int:
    """Download one unit and its dependencies at `revision`; load them with checksums."""
    huggingface_hub.hf_hub_download(repo_id, constants.MANIFEST_NAME, revision=revision,
                                    local_dir=str(dest))
    units = {unit.id: unit for unit in loader.read_manifest(dest).units}
    wanted = [unit_id, *units[unit_id].depends_on]
    huggingface_hub.snapshot_download(repo_id, revision=revision, local_dir=str(dest),
                                      allow_patterns=[units[wanted_id].file for wanted_id in wanted])
    with loader.Artifact(dest) as artifact:
        return sum(len(artifact.load_unit(wanted_id)) for wanted_id in wanted)
