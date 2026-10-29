"""Plans units from tensor specs and writes Safetensors partitions plus manifest."""
import json
import re
from collections import defaultdict
from pathlib import Path

from safetensors.torch import save_file

from model_partitioner import adapters, constants, models, source, tensor_utils

_FORMAT_METADATA = {'format': 'pt'}


def _unit_for(placement: models.Placement, experts_per_file: int) -> models.PlannedUnit:
    role, layer, expert = placement.role, placement.layer, placement.expert
    if role == constants.ROLE_LAYER:
        return models.PlannedUnit(f'layer:{layer}', role, f'layers/layer-{layer:03d}.safetensors',
                                  layer=layer)
    if role == constants.ROLE_EXPERT:
        chunk = expert // experts_per_file
        return models.PlannedUnit(
            f'expert:{layer}:{expert}', role,
            f'experts/layer-{layer:03d}-experts-{chunk:03d}.safetensors',
            layer=layer, expert=expert, depends_on=[f'layer:{layer}'])
    if role == constants.ROLE_PERCEPTION:
        return models.PlannedUnit(f'perception:{placement.component}', role,
                                  f'perception/{placement.component}.safetensors',
                                  component=placement.component)
    return models.PlannedUnit(role, role, f'{role}.safetensors')


def _add_fused(units, spec, placement, num_experts, experts_per_file):
    if not spec.shape or spec.shape[0] != num_experts:
        raise models.UnsupportedLayout(
            f'{spec.key}: expected leading expert dim {num_experts}, got shape {spec.shape}')
    for expert in range(num_experts):
        unit = units.setdefault(
            f'expert:{placement.layer}:{expert}',
            _unit_for(models.Placement(constants.ROLE_EXPERT, placement.layer, expert),
                      experts_per_file))
        name = f'{spec.key}{constants.SLICE_SEPARATOR}{expert}'
        unit.pieces.append(models.PlannedPiece(spec, name, expert, expert + 1))


def plan(specs: list[models.TensorSpec], adapter: adapters.Adapter, config: dict,
         experts_per_file: int) -> list[models.PlannedUnit]:
    num_experts = adapter.num_experts(config)
    units: dict[str, models.PlannedUnit] = {}
    for spec in specs:
        placement = adapter.classify(spec.key)
        if placement.fused_experts:
            _add_fused(units, spec, placement, num_experts, experts_per_file)
            continue
        if placement.expert is not None and placement.expert >= num_experts:
            raise models.UnsupportedLayout(f'{spec.key}: expert id outside 0..{num_experts - 1}')
        unit = units.setdefault(_unit_for(placement, experts_per_file).id,
                                _unit_for(placement, experts_per_file))
        unit.pieces.append(models.PlannedPiece(spec, spec.key))
    _check_complete(units, adapter, config)
    return sorted(units.values(), key=lambda unit: (unit.file, unit.id))


def _check_complete(units, adapter: adapters.Adapter, config: dict) -> None:
    num_experts = adapter.num_experts(config)
    required = {constants.ROLE_EMBEDDING, constants.ROLE_FINAL}
    if missing := required - set(units):
        raise models.IncompleteSource(f'missing units: {sorted(missing)}')
    layers = {u.layer for u in units.values() if u.role == constants.ROLE_LAYER}
    if layers != set(range(adapter.num_layers(config))):
        raise models.IncompleteSource(
            f'layers {sorted(set(range(adapter.num_layers(config))) - layers)} missing '
            f'or unexpected layers {sorted(layers - set(range(adapter.num_layers(config))))}')
    by_layer = defaultdict(set)
    for unit in units.values():
        if unit.role == constants.ROLE_EXPERT:
            by_layer[unit.layer].add(unit.expert)
    if set(by_layer) != adapter.sparse_layers(config):
        raise models.IncompleteSource(
            f'expert layers {sorted(by_layer)} differ from config {sorted(adapter.sparse_layers(config))}')
    for layer, experts in by_layer.items():
        if experts != set(range(num_experts)):
            raise models.IncompleteSource(
                f'layer {layer}: experts {sorted(set(range(num_experts)) - experts)} missing')
    _check_expert_shapes(units, adapter)


_EXPERT_ID = re.compile(r'experts\.\d+\.')


def _check_expert_shapes(units, adapter: adapters.Adapter) -> None:
    """Within each layer, every expert must hold the same tensors (names, dtypes, shapes)."""
    signatures = {}
    for unit in units.values():
        if unit.role != constants.ROLE_EXPERT:
            continue
        signatures[unit.id] = (unit.layer, frozenset(
            (_EXPERT_ID.sub('experts.*.', piece.spec.key[len(adapter.layer_prefix):].partition('.')[2]),
             piece.spec.dtype,
             piece.spec.shape if piece.start is None else piece.spec.shape[1:])
            for piece in unit.pieces))
    reference = {}
    for uid, (layer, signature) in sorted(signatures.items()):
        if reference.setdefault(layer, signature) != signature:
            raise models.IncompleteSource(
                f'layer {layer}: expert {uid} differs in tensors, dtypes or shapes')


def summarize(units: list[models.PlannedUnit]) -> dict:
    by_role = defaultdict(lambda: {'units': 0, 'tensors': 0, 'bytes': 0})
    for unit in units:
        row = by_role[unit.role]
        row['units'] += 1
        row['tensors'] += len(unit.pieces)
        for piece in unit.pieces:
            full = piece.spec.nbytes
            rows = piece.spec.shape[0] if piece.start is not None else 1
            row['bytes'] += full // rows if piece.start is not None else full
    return dict(by_role)


def _write_file(src: source.LocalSource, out_dir: Path, file: str,
                units: list[models.PlannedUnit]) -> list[models.Piece]:
    tensors, pieces = {}, []
    for unit in units:
        for planned in unit.pieces:
            tensor = src.read(planned.spec, planned.start, planned.stop)
            tensors[planned.name] = tensor
            pieces.append(models.Piece(
                name=planned.name, unit=unit.id, file=file, source_key=planned.spec.key,
                shape=list(tensor.shape),
                dtype=planned.spec.dtype, nbytes=tensor.numel() * tensor.element_size(),
                sha256=tensor_utils.tensor_sha256(tensor),
                slice_start=planned.start, slice_stop=planned.stop))
    path = out_dir / file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path, metadata=_FORMAT_METADATA)
    return pieces


def _source_files(src: source.LocalSource, expected: dict[str, str] | None):
    files = []
    for name in src.files:
        path = src.root / name
        digest = tensor_utils.file_sha256(path)
        if expected and name in expected and expected[name] != digest:
            raise ValueError(f'{name}: sha256 {digest} != Hub {expected[name]}')
        files.append(models.SourceFile(name, path.stat().st_size, digest,
                                       src.shard_metadata(name)))
    return files


def partition(src: source.LocalSource, adapter: adapters.Adapter, out_dir: Path, repo_id: str,
              revision: str, experts_per_file: int = constants.DEFAULT_EXPERTS_PER_FILE,
              expected_hashes: dict[str, str] | None = None) -> models.Manifest:
    if len(revision) != constants.REVISION_LENGTH:
        raise ValueError('revision must be a full commit SHA')
    config = src.config()
    specs = src.specs()
    units = plan(specs, adapter, config, experts_per_file)
    out_dir = Path(out_dir)
    source_files = _source_files(src, expected_hashes)
    by_file = defaultdict(list)
    for unit in units:
        by_file[unit.file].append(unit)
    pieces = []
    for file, file_units in by_file.items():
        pieces.extend(_write_file(src, out_dir, file, file_units))
    files = {file: models.FileInfo((out_dir / file).stat().st_size,
                                   tensor_utils.file_sha256(out_dir / file))
             for file in by_file}
    manifest = models.Manifest(
        family=adapter.family, repo_id=repo_id, revision=revision, config=config,
        experts_per_file=experts_per_file, source_files=source_files,
        source_tensors={s.key: models.SourceTensor(list(s.shape), s.dtype, s.nbytes, s.file)
                        for s in specs},
        assets=source.copy_assets(src, out_dir / constants.ASSETS_DIR), files=files,
        units=[models.Unit(u.id, u.role, u.file, u.layer, u.expert, u.component, u.depends_on,
                           [p.name for p in u.pieces]) for u in units],
        tensors=pieces)
    (out_dir / constants.MANIFEST_NAME).write_text(json.dumps(manifest.to_dict(), indent=1))
    return manifest
