"""The network's pinned checkpoint: its pin from the DHT and its layout from the manifest."""

import trio
from libp2p.kad_dht.kad_dht import KadDHT

from client.network import models
from model_partitioner.utils import constants as partitioner_constants
from model_partitioner.utils import models as partitioner_models
from worker.discovery import discovery
from worker.loading import manifest as manifest_loading

_FAMILY = "olmoe"
Cache = dict[tuple[str, str], models.Checkpoint]


async def load(dht: KadDHT, cache: Cache) -> models.Checkpoint:
    """The checkpoint the network pinned; a pin's manifest is fetched once."""
    repo, revision = await discovery.read_network_pin(dht)
    if (repo, revision) not in cache:
        manifest = await trio.to_thread.run_sync(
            manifest_loading.fetch_manifest, repo, revision, abandon_on_cancel=True
        )
        cache[(repo, revision)] = _checkpoint(repo, revision, manifest)
    return cache[(repo, revision)]


def _checkpoint(
    repo: str, revision: str, manifest: partitioner_models.Manifest
) -> models.Checkpoint:
    by_role: dict[str, list] = {}
    for unit in manifest.units:
        by_role.setdefault(unit.role, []).append(unit)
    layers = manifest.config["num_hidden_layers"]
    layer_units = sorted(
        by_role.get(partitioner_constants.ROLE_LAYER, []), key=lambda u: u.layer
    )
    embedding = by_role.get(partitioner_constants.ROLE_EMBEDDING, [])
    final = by_role.get(partitioner_constants.ROLE_FINAL, [])
    experts = manifest.config["num_experts"]
    expert_units = [
        sorted(
            (
                u
                for u in by_role.get(partitioner_constants.ROLE_EXPERT, [])
                if u.layer == i
            ),
            key=lambda u: u.expert,
        )
        for i in range(layers)
    ]
    if (
        manifest.family != _FAMILY
        or len(embedding) != 1
        or len(final) != 1
        or [unit.layer for unit in layer_units] != list(range(layers))
        or any(
            [u.expert for u in group] != list(range(experts)) for group in expert_units
        )
        or len(by_role.get(partitioner_constants.ROLE_EXPERT, [])) != layers * experts
    ):
        raise models.RouteError("the pinned manifest does not describe a whole model")
    return models.Checkpoint(
        repo,
        revision,
        layers,
        experts,
        embedding[0].id,
        final[0].id,
        tuple(unit.id for unit in layer_units),
        tuple(tuple(unit.id for unit in group) for group in expert_units),
    )
