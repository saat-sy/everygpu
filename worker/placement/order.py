"""Model order of manifest units: sizes, a flat order and per-layer groups."""

from model_partitioner.utils import constants as partitioner_constants
from model_partitioner.utils import models

_ROLE_RANK = {
    partitioner_constants.ROLE_EMBEDDING: 0,
    partitioner_constants.ROLE_LAYER: 1,
    partitioner_constants.ROLE_EXPERT: 1,
    partitioner_constants.ROLE_FINAL: 2,
}
LAYER_ROLES = frozenset(
    (partitioner_constants.ROLE_LAYER, partitioner_constants.ROLE_EXPERT)
)


def unit_bytes(manifest: models.Manifest) -> dict[str, int]:
    """Total tensor bytes of each unit."""
    nbytes = {piece.name: piece.nbytes for piece in manifest.tensors}
    return {unit.id: sum(nbytes[t] for t in unit.tensors) for unit in manifest.units}


def model_order(manifest: models.Manifest) -> list[str]:
    """Embedding, then each layer's shared unit followed by its experts, then final."""
    units = sorted(
        manifest.units,
        key=lambda u: (
            _ROLE_RANK.get(u.role, len(_ROLE_RANK)),
            -1 if u.layer is None else u.layer,
            -1 if u.expert is None else u.expert,
            u.id,
        ),
    )
    return [unit.id for unit in units]


def model_groups(manifest: models.Manifest) -> list[list[str]]:
    """Model order split into groups: a layer with its experts, or one other unit."""
    units = {unit.id: unit for unit in manifest.units}
    groups: list[list[str]] = []
    last = None
    for unit_id in model_order(manifest):
        unit = units[unit_id]
        key = unit.layer if unit.role in LAYER_ROLES else unit.id
        if groups and key == last:
            groups[-1].append(unit_id)
        else:
            groups.append([unit_id])
        last = key
    return groups
