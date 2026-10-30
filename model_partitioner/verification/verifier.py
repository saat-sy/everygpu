"""Verifies a partitioned artifact against its manifest and the source checkpoint."""

from pathlib import Path

from safetensors import safe_open

from model_partitioner.utils import constants, models, tensor_utils
from model_partitioner.artifact import hub, loader
from model_partitioner.partition import source

RECONSTRUCTION = "exact tensor reconstruction"


def _result(name: str, problems: list[str]) -> models.CheckResult:
    if problems:
        detail = f"{len(problems)} problems, first: {problems[:3]}"
        return models.CheckResult(name, constants.FAILED, detail)
    return models.CheckResult(name, constants.PASSED)


def _skipped(name: str, reason: str) -> models.CheckResult:
    return models.CheckResult(name, constants.UNVERIFIED, f"skipped: {reason}")


def check_files(artifact: loader.Artifact) -> models.CheckResult:
    problems = []
    for name, info in loader.declared_files(artifact.manifest).items():
        path = artifact.root / name
        if not path.is_file():
            problems.append(f"{name}: missing")
        elif (
            path.stat().st_size != info.size
            or tensor_utils.file_sha256(path) != info.sha256
        ):
            problems.append(f"{name}: size or sha256 mismatch")
    return _result("file checksums", problems)


def _manifest_relations(manifest: models.Manifest) -> list[str]:
    """Units, pieces, files and source keys must reference one another consistently."""
    problems = []
    unit_ids = [unit.id for unit in manifest.units]
    names = [piece.name for piece in manifest.tensors]
    if len(unit_ids) != len(set(unit_ids)):
        problems.append("duplicate unit ids")
    if len(names) != len(set(names)):
        problems.append("duplicate tensor names")
    pieces = {piece.name: piece for piece in manifest.tensors}
    owned = [name for unit in manifest.units for name in unit.tensors]
    if len(owned) != len(set(owned)) or set(owned) != set(pieces):
        problems.append("unit tensor lists do not partition the tensors exactly once")
    for unit in manifest.units:
        if unit.file not in manifest.files:
            problems.append(f"{unit.id}: file {unit.file} not in manifest files")
        problems += [
            f"{unit.id}: dependency {dep} missing"
            for dep in unit.depends_on
            if dep not in unit_ids
        ]
        for name in unit.tensors:
            piece = pieces.get(name)
            if piece and (piece.unit, piece.file) != (unit.id, unit.file):
                problems.append(f"{name}: unit or file differs from {unit.id}")
    for piece in manifest.tensors:
        if piece.source_key not in manifest.source_tensors:
            problems.append(f"{piece.name}: unknown source key {piece.source_key}")
        if piece.file not in manifest.files:
            problems.append(f"{piece.name}: file {piece.file} not in manifest files")
    return problems


def _shard_headers(artifact: loader.Artifact) -> list[str]:
    problems = []
    for file in sorted({piece.file for piece in artifact.manifest.tensors}):
        expected = {p.name: p for p in artifact.manifest.tensors if p.file == file}
        with safe_open(artifact.root / file, framework="pt") as handle:
            on_disk = {key: handle.get_slice(key) for key in handle.keys()}
        if set(on_disk) != set(expected):
            problems.append(f"{file}: tensors differ from manifest")
        for name in set(expected) & set(on_disk):
            header = (on_disk[name].get_dtype(), list(on_disk[name].get_shape()))
            if header != (expected[name].dtype, expected[name].shape):
                problems.append(f"{name}: header differs from manifest")
    return problems


def check_coverage(
    artifact: loader.Artifact, specs: list[models.TensorSpec]
) -> models.CheckResult:
    manifest = artifact.manifest
    problems = _manifest_relations(manifest)
    by_key = {spec.key: spec for spec in specs}
    if set(by_key) != set(manifest.source_tensors):
        diff = sorted(set(by_key) ^ set(manifest.source_tensors))[:5]
        problems.append(f"source keys differ: {diff}")
    for key, spec in by_key.items():
        info = manifest.source_tensors.get(key)
        if info and (list(spec.shape), spec.dtype, spec.nbytes) != (
            info.shape,
            info.dtype,
            info.nbytes,
        ):
            problems.append(f"{key}: shape/dtype/size differs from source")
        covered = sum(piece.nbytes for piece in artifact.pieces_by_key.get(key, []))
        if covered != spec.nbytes:
            problems.append(f"{key}: pieces cover {covered} of {spec.nbytes} bytes")
    if not problems:
        problems = _shard_headers(artifact)
    return _result("coverage", problems)


def check_reconstruction(
    artifact: loader.Artifact, src: source.LocalSource, specs: list[models.TensorSpec]
) -> models.CheckResult:
    problems = []
    for spec in specs:
        try:
            rebuilt = artifact.tensor(spec.key)
        except (models.ChecksumMismatch, ValueError, KeyError) as error:
            problems.append(f"{spec.key}: {error}")
            continue
        original = src.read(spec)
        if (
            rebuilt.dtype != original.dtype
            or rebuilt.shape != original.shape
            or tensor_utils.tensor_sha256(rebuilt)
            != tensor_utils.tensor_sha256(original)
        ):
            problems.append(f"{spec.key}: reconstruction differs from source")
    return _result(RECONSTRUCTION, problems)


def verify_structure(artifact_dir: Path) -> list[models.CheckResult]:
    """Run the checks in order; a failed prerequisite skips the checks that need it."""
    try:
        artifact = loader.Artifact(artifact_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [models.CheckResult("manifest", constants.FAILED, str(error))]
    with artifact:
        results = [check_files(artifact)]
        if results[0].status != constants.PASSED:
            return results + _skip_rest("file checksums failed")
        manifest = artifact.manifest
        try:
            source_dir = hub.fetch_source(manifest.repo_id, manifest.revision)
        except ValueError as error:
            return results + _skip_rest(f"source unavailable: {error}")
        with source.LocalSource(source_dir) as src:
            try:
                specs = src.specs()
            except (OSError, ValueError, KeyError) as error:
                failed = models.CheckResult(
                    "source checkpoint", constants.FAILED, str(error)
                )
                return results + [failed]
            results.append(check_coverage(artifact, specs))
            if results[-1].status != constants.PASSED:
                return results + [_skipped(RECONSTRUCTION, "coverage not passed")]
            return results + [check_reconstruction(artifact, src, specs)]


def _skip_rest(reason: str) -> list[models.CheckResult]:
    return [_skipped("coverage", reason), _skipped(RECONSTRUCTION, reason)]
