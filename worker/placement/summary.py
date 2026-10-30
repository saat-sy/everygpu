"""Readable one-line-per-layer summaries of unit selections."""

from model_partitioner.utils import constants as partitioner_constants
from model_partitioner.utils import models
from worker.placement import order


def describe(manifest: models.Manifest, unit_ids) -> list[str]:
    """One readable line per role or layer, with expert ranges compacted."""
    units = {unit.id: unit for unit in manifest.units}
    shared: set[str] = set()
    experts: dict[str, list[int]] = {}
    names: list[str] = []
    for unit_id in order.model_order(manifest):
        if unit_id not in unit_ids:
            continue
        unit = units[unit_id]
        name = unit.id if unit.role not in order.LAYER_ROLES else f"layer {unit.layer}"
        if name not in names:
            names.append(name)
        if unit.role == partitioner_constants.ROLE_LAYER:
            shared.add(name)
        elif unit.role == partitioner_constants.ROLE_EXPERT:
            experts.setdefault(name, []).append(unit.expert)
    return [
        name + _layer_parts(name in shared, experts.get(name, [])) for name in names
    ]


def _layer_parts(has_shared: bool, experts: list[int]) -> str:
    parts = (["shared"] if has_shared else []) + (
        [f"experts {_ranges(experts)}"] if experts else []
    )
    return ": " + ", ".join(parts) if parts else ""


def _ranges(numbers: list[int]) -> str:
    spans: list[list[int]] = []
    for n in numbers:
        if spans and n == spans[-1][1] + 1:
            spans[-1][1] = n
        else:
            spans.append([n, n])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)
