"""Plans units from tensor specs and writes Safetensors partitions plus manifest."""

import json
import re
from collections import defaultdict
from pathlib import Path

from safetensors.torch import save_file

from model_partitioner.utils import constants, models, tensor_utils
from model_partitioner.partition import adapters, source


def _unit_for(placement: models.Placement, experts_per_file: int) -> models.PlannedUnit:
    role, layer, expert = placement.role, placement.layer, placement.expert
    if role == constants.ROLE_LAYER:
        return models.PlannedUnit(
            f"layer:{layer}", role, f"layers/layer-{layer:03d}.safetensors", layer=layer
        )
    if role == constants.ROLE_EXPERT:
        chunk = expert // experts_per_file
        return models.PlannedUnit(
            f"expert:{layer}:{expert}",
            role,
            f"experts/layer-{layer:03d}-experts-{chunk:03d}.safetensors",
            layer=layer,
            expert=expert,
            depends_on=[f"layer:{layer}"],
        )
    if role == constants.ROLE_PERCEPTION:
        return models.PlannedUnit(
            f"perception:{placement.component}",
            role,
            f"perception/{placement.component}.safetensors",
            component=placement.component,
        )
    return models.PlannedUnit(role, role, f"{role}.safetensors")


def _add_fused(units, spec, placement, num_experts, experts_per_file):
    if not spec.shape or spec.shape[0] != num_experts:
        raise models.UnsupportedLayout(
            f"{spec.key}: expected leading expert dim {num_experts}, got shape {spec.shape}"
        )
    for expert in range(num_experts):
        unit = units.setdefault(
            f"expert:{placement.layer}:{expert}",
            _unit_for(
                models.Placement(constants.ROLE_EXPERT, placement.layer, expert),
                experts_per_file,
            ),
        )
        name = f"{spec.key}{constants.SLICE_SEPARATOR}{expert}"
        unit.pieces.append(models.PlannedPiece(spec, name, expert, expert + 1))


def plan(
    specs: list[models.TensorSpec],
    adapter: adapters.Adapter,
    config: dict,
    experts_per_file: int,
) -> list[models.PlannedUnit]:
    if experts_per_file < 1:
        raise ValueError("experts_per_file must be positive")
    adapter.check_config(config)
    num_experts = adapter.num_experts(config)
    units: dict[str, models.PlannedUnit] = {}
    for spec in specs:
        placement = adapter.classify(spec.key)
        if placement.fused_experts:
            _add_fused(units, spec, placement, num_experts, experts_per_file)
            continue
        if placement.expert is not None and placement.expert >= num_experts:
            raise models.UnsupportedLayout(
                f"{spec.key}: expert id outside 0..{num_experts - 1}"
            )
        unit = _unit_for(placement, experts_per_file)
        unit = units.setdefault(unit.id, unit)
        unit.pieces.append(models.PlannedPiece(spec, spec.key))
    _check_complete(units, adapter, config)
    return sorted(units.values(), key=lambda unit: (unit.file, unit.id))


def _check_complete(units, adapter: adapters.Adapter, config: dict) -> None:
    num_experts = adapter.num_experts(config)
    required = {constants.ROLE_EMBEDDING, constants.ROLE_FINAL}
    if missing := required - set(units):
        raise models.IncompleteSource(f"missing units: {sorted(missing)}")
    layers = {u.layer for u in units.values() if u.role == constants.ROLE_LAYER}
    if layers != set(range(adapter.num_layers(config))):
        raise models.IncompleteSource(
            f"layers {sorted(set(range(adapter.num_layers(config))) - layers)} missing "
            f"or unexpected layers {sorted(layers - set(range(adapter.num_layers(config))))}"
        )
    by_layer = defaultdict(set)
    for unit in units.values():
        if unit.role == constants.ROLE_EXPERT:
            by_layer[unit.layer].add(unit.expert)
    if set(by_layer) != adapter.sparse_layers(config):
        raise models.IncompleteSource(
            f"expert layers {sorted(by_layer)} differ from config {sorted(adapter.sparse_layers(config))}"
        )
    for layer, experts in by_layer.items():
        if experts != set(range(num_experts)):
            raise models.IncompleteSource(
                f"layer {layer}: experts {sorted(set(range(num_experts)) - experts)} missing"
            )
    _check_expert_shapes(units, adapter)


_EXPERT_ID = re.compile(r"experts\.\d+\.")


def _check_expert_shapes(units, adapter: adapters.Adapter) -> None:
    """Within each layer, every expert must hold the same tensors (names, dtypes, shapes)."""
    signatures = {}
    for unit in units.values():
        if unit.role != constants.ROLE_EXPERT:
            continue
        signatures[unit.id] = (
            unit.layer,
            frozenset(
                (
                    _EXPERT_ID.sub(
                        "experts.*.",
                        piece.spec.key[len(adapter.layer_prefix) :].partition(".")[2],
                    ),
                    piece.spec.dtype,
                    piece.spec.shape if piece.start is None else piece.spec.shape[1:],
                )
                for piece in unit.pieces
            ),
        )
    reference = {}
    for uid, (layer, signature) in sorted(signatures.items()):
        if reference.setdefault(layer, signature) != signature:
            raise models.IncompleteSource(
                f"layer {layer}: expert {uid} differs in tensors, dtypes or shapes"
            )


def plan_source(
    src: source.LocalSource,
    adapter: adapters.Adapter,
    experts_per_file: int = constants.DEFAULT_EXPERTS_PER_FILE,
) -> list[models.PlannedUnit]:
    return plan(src.specs(), adapter, src.config(), experts_per_file)


def _piece_bytes(piece: models.PlannedPiece) -> int:
    if piece.start is None:
        return piece.spec.nbytes
    return piece.spec.nbytes // piece.spec.shape[0]


def format_summary(units: list[models.PlannedUnit]) -> str:
    by_role = defaultdict(lambda: [0, 0, 0])
    for unit in units:
        row = by_role[unit.role]
        row[0] += 1
        row[1] += len(unit.pieces)
        row[2] += sum(_piece_bytes(piece) for piece in unit.pieces)
    lines = [
        f"{role:11} {units_:6} units {tensors:7} tensors {size:>16,} bytes"
        for role, (units_, tensors, size) in sorted(by_role.items())
    ]
    total = sum(row[2] for row in by_role.values())
    lines.append(
        f"plan: {len({unit.file for unit in units})} files, {total:,} tensor bytes"
    )
    return "\n".join(lines)


def _write_file(
    src: source.LocalSource, out_dir: Path, file: str, units: list[models.PlannedUnit]
) -> list[models.Piece]:
    tensors, pieces = {}, []
    for unit in units:
        for planned in unit.pieces:
            tensor = src.read(planned.spec, planned.start, planned.stop)
            tensors[planned.name] = tensor
            pieces.append(
                models.Piece(
                    name=planned.name,
                    unit=unit.id,
                    file=file,
                    source_key=planned.spec.key,
                    shape=list(tensor.shape),
                    dtype=planned.spec.dtype,
                    nbytes=tensor.numel() * tensor.element_size(),
                    sha256=tensor_utils.tensor_sha256(tensor),
                    slice_start=planned.start,
                    slice_stop=planned.stop,
                )
            )
    path = out_dir / file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path, metadata=dict(constants.SAFETENSORS_METADATA))
    return pieces


def check_revision(revision: str) -> str:
    if not constants.REVISION_PATTERN.fullmatch(revision):
        raise ValueError(
            "revision must be a full 40-character lowercase hex commit SHA"
        )
    return revision


def check_destination(source_root: Path, out_dir: Path) -> None:
    source_root, out_dir = source_root.resolve(), out_dir.resolve()
    if (
        out_dir == source_root
        or source_root in out_dir.parents
        or out_dir in source_root.parents
    ):
        raise ValueError(f"output {out_dir} overlaps source {source_root}")
    tensor_utils.require_empty_dir(out_dir)


def _source_files(src: source.LocalSource, expected: dict[str, str]):
    if unchecked := set(src.files) - set(expected):
        raise ValueError(f"no Hub sha256 to check shards {sorted(unchecked)}")
    files = []
    for name in src.files:
        path = src.root / name
        digest = tensor_utils.file_sha256(path)
        if expected[name] != digest:
            raise ValueError(f"{name}: sha256 {digest} != Hub {expected[name]}")
        files.append(
            models.SourceFile(
                name, path.stat().st_size, digest, src.shard_metadata(name)
            )
        )
    return files


def partition(
    src: source.LocalSource,
    adapter: adapters.Adapter,
    out_dir: Path,
    repo_id: str,
    revision: str,
    expected_hashes: dict[str, str],
    experts_per_file: int = constants.DEFAULT_EXPERTS_PER_FILE,
) -> models.Manifest:
    check_revision(revision)
    out_dir = Path(out_dir)
    check_destination(src.root, out_dir)
    config = src.config()
    specs = src.specs()
    units = plan(specs, adapter, config, experts_per_file)
    source_files = _source_files(src, expected_hashes)
    by_file = defaultdict(list)
    for unit in units:
        by_file[unit.file].append(unit)
    pieces = []
    for file, file_units in by_file.items():
        pieces.extend(_write_file(src, out_dir, file, file_units))
    files = {
        file: models.FileInfo(
            (out_dir / file).stat().st_size, tensor_utils.file_sha256(out_dir / file)
        )
        for file in by_file
    }
    manifest = models.Manifest(
        family=adapter.family,
        repo_id=repo_id,
        revision=revision,
        config=config,
        experts_per_file=experts_per_file,
        source_files=source_files,
        source_tensors={
            s.key: models.SourceTensor(list(s.shape), s.dtype, s.nbytes, s.file)
            for s in specs
        },
        assets=source.copy_assets(src, out_dir / constants.ASSETS_DIR),
        files=files,
        units=[
            models.Unit(
                u.id,
                u.role,
                u.file,
                u.layer,
                u.expert,
                u.component,
                u.depends_on,
                [p.name for p in u.pieces],
            )
            for u in units
        ],
        tensors=pieces,
    )
    (out_dir / constants.MANIFEST_NAME).write_text(
        json.dumps(manifest.to_dict(), indent=1)
    )
    return manifest
