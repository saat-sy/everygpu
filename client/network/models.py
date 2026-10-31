"""Client network data structures: checkpoint layout, route hops and failures. No I/O."""

import dataclasses

from worker.execution import models as execution_models


class RouteError(Exception):
    """No admitted route is available; the message says why."""


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

    def unit_ids(self) -> list[str]:
        experts = [unit for layer in self.expert_units for unit in layer]
        return [self.embedding, *self.layer_units, *experts, self.final]


@dataclasses.dataclass(frozen=True)
class Hop:
    """A worker that executes one stage, reachable at its execution endpoint."""

    peer: str
    endpoint: execution_models.Endpoint
    stage: execution_models.StageDescriptor


@dataclasses.dataclass(frozen=True)
class Route:
    """Ordered hops that together execute the whole model."""

    repo: str
    revision: str
    hops: tuple[Hop, ...]
