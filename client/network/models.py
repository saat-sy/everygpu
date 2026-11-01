"""Client network data structures: checkpoint layout, route hops and failures. No I/O."""

import dataclasses

from model_partitioner.utils import models as partitioner_models
from worker.execution import models as execution_models


class RouteError(Exception):
    """No admitted route is available; the message says why."""


class GenerationError(Exception):
    """An admitted route failed while generating; the message says where."""


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    """The pinned checkpoint's shape and the ids of its units."""

    repo: str
    revision: str
    layers: int
    experts: int
    embedding: str
    final: str
    layer_units: tuple[str, ...]
    expert_units: tuple[tuple[str, ...], ...]
    config: dict
    assets: dict[str, partitioner_models.FileInfo]

    def unit_ids(self) -> list[str]:
        experts = [unit for layer in self.expert_units for unit in layer]
        return [self.embedding, *self.layer_units, *experts, self.final]


Hop = execution_models.Hop


@dataclasses.dataclass(frozen=True)
class Route:
    """Ordered hops that together execute the whole model."""

    repo: str
    revision: str
    hops: tuple[Hop, ...]
