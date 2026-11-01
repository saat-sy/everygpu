"""Request-time discovery: ready workers whose advertised executable stage fits the checkpoint."""

from libp2p.kad_dht.kad_dht import KadDHT

from client.network import models
from worker.discovery import discovery
from worker.discovery import models as discovery_models
from worker.execution import models as execution_models


def forget_cached(dht: KadDHT) -> None:
    """Drop lookups the library cached, so a long-lived client sees current workers."""
    # shortcut: library caches lookups locally
    dht.value_store.store.clear()
    dht.provider_store.providers.clear()


async def find_hops(dht: KadDHT, checkpoint: models.Checkpoint) -> list[models.Hop]:
    """Executable workers ordered by peer id; weight coverage alone never qualifies one."""
    found = await discovery.find_coverage(
        dht, checkpoint.repo, checkpoint.revision, checkpoint.unit_ids()
    )
    provided: dict[str, set[str]] = {}
    statuses: dict[str, discovery_models.PeerStatus] = {}
    for unit_id, peers in found.items():
        for peer, status in peers.items():
            provided.setdefault(peer, set()).add(unit_id)
            statuses[peer] = status
    return [
        models.Hop(peer, status.execution, status.stage)
        for peer, status in sorted(statuses.items())
        if _executable(checkpoint, status, provided[peer])
    ]


def _executable(
    checkpoint: models.Checkpoint,
    status: discovery_models.PeerStatus,
    provided: set[str],
) -> bool:
    stage = status.stage
    return (
        status.execution is not None
        and stage is not None
        and (stage.layers, stage.experts) == (checkpoint.layers, checkpoint.experts)
        and _stage_units(checkpoint, stage) <= provided
    )


def _stage_units(
    checkpoint: models.Checkpoint, stage: execution_models.StageDescriptor
) -> set[str]:
    units = set(checkpoint.layer_units[stage.first : stage.stop])
    for layer in checkpoint.expert_units[stage.first : stage.stop]:
        units.update(layer)
    if stage.embedding:
        units.add(checkpoint.embedding)
    if stage.final:
        units.add(checkpoint.final)
    return units
