"""Verifies a partitioned artifact against its manifest and the source checkpoint."""
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open

from model_partitioner import constants, loader, source, tensor_utils

PASSED, FAILED, UNVERIFIED = 'PASSED', 'FAILED', 'UNVERIFIED'


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ''


def _check(name: str, problems: list[str]) -> CheckResult:
    if problems:
        return CheckResult(name, FAILED, f'{len(problems)} problems, first: {problems[:3]}')
    return CheckResult(name, PASSED)


def check_files(artifact: loader.Artifact) -> CheckResult:
    problems = []
    infos = {**artifact.manifest.files,
             **{f'{constants.ASSETS_DIR}/{n}': i for n, i in artifact.manifest.assets.items()}}
    for name, info in infos.items():
        path = artifact.root / name
        if not path.is_file():
            problems.append(f'{name}: missing')
        elif path.stat().st_size != info.size or tensor_utils.file_sha256(path) != info.sha256:
            problems.append(f'{name}: size or sha256 mismatch')
    return _check('file checksums', problems)


def check_coverage(artifact: loader.Artifact, src: source.LocalSource) -> CheckResult:
    manifest = artifact.manifest
    problems = []
    specs = {spec.key: spec for spec in src.specs()}
    if set(specs) != set(manifest.source_tensors):
        problems.append(f'source keys differ: {sorted(set(specs) ^ set(manifest.source_tensors))[:5]}')
    for key, spec in specs.items():
        info = manifest.source_tensors.get(key)
        if info and (list(spec.shape), spec.dtype, spec.nbytes) != (
                info.shape, info.dtype, info.nbytes):
            problems.append(f'{key}: shape/dtype/size differs from source')
        covered = sum(piece.nbytes for piece in artifact.pieces_by_key.get(key, []))
        if covered != spec.nbytes:
            problems.append(f'{key}: pieces cover {covered} of {spec.nbytes} bytes')
    problems += _check_unit_membership(artifact)
    return _check('coverage', problems)


def _check_unit_membership(artifact: loader.Artifact) -> list[str]:
    owned = [name for unit in artifact.manifest.units for name in unit.tensors]
    problems = []
    if len(owned) != len(set(owned)) or set(owned) != set(artifact.pieces):
        problems.append('unit tensor lists do not partition the manifest tensors exactly once')
    for file in {piece.file for piece in artifact.manifest.tensors}:
        with safe_open(artifact.root / file, framework='pt') as handle:
            on_disk = {key: handle.get_slice(key) for key in handle.keys()}
            expected = {p.name for p in artifact.manifest.tensors if p.file == file}
            if set(on_disk) != expected:
                problems.append(f'{file}: tensors differ from manifest')
            for name in expected & set(on_disk):
                piece = artifact.pieces[name]
                if (on_disk[name].get_dtype(), list(on_disk[name].get_shape())) != (
                        piece.dtype, piece.shape):
                    problems.append(f'{name}: header differs from manifest')
    units = {unit.id for unit in artifact.manifest.units}
    problems += [f'{u.id}: dependency {d} missing' for u in artifact.manifest.units
                 for d in u.depends_on if d not in units]
    return problems


def check_reconstruction(artifact: loader.Artifact, src: source.LocalSource) -> CheckResult:
    problems = []
    specs = {spec.key: spec for spec in src.specs()}
    for key, spec in specs.items():
        try:
            rebuilt = artifact.tensor(key)
        except (loader.ChecksumMismatch, ValueError, KeyError) as error:
            problems.append(f'{key}: {error}')
            continue
        original = src.read(spec)
        if rebuilt.dtype != original.dtype or tensor_utils.tensor_sha256(
                rebuilt) != tensor_utils.tensor_sha256(original):
            problems.append(f'{key}: reconstruction differs from source bytes')
    return _check('exact tensor reconstruction', problems)


def verify_structure(artifact_dir: Path, source_dir: Path) -> list[CheckResult]:
    with loader.Artifact(artifact_dir) as artifact, source.LocalSource(source_dir) as src:
        return [check_files(artifact), check_coverage(artifact, src),
                check_reconstruction(artifact, src)]
