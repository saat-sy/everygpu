"""Discovery: read the network pin and look up which ready peers provide which pieces."""

import json
import logging
import time

import trio
from libp2p.kad_dht.kad_dht import KadDHT

from worker.discovery import lookup
from worker.discovery import models
from worker.discovery import records
from worker.execution import models as execution_models
from worker.utils import constants

logger = logging.getLogger(__name__)
PeerStatuses = dict[str, models.PeerStatus]  # peer id -> ready status
Coverage = dict[str, PeerStatuses]  # unit id -> its ready providers


async def read_network_pin(dht: KadDHT) -> tuple[str, str]:
    """The repo and revision this network pinned; never resolved locally."""
    raw = await lookup.get_value(dht, constants.NETWORK_KEY)
    try:
        records.RecordValidator().validate(constants.NETWORK_KEY, raw)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "no valid network pin found through bootstrap peers"
        ) from error
    data = json.loads(raw)
    return data["repo"], data["revision"]


async def find_coverage(
    dht: KadDHT, repo: str, revision: str, unit_ids: list[str]
) -> Coverage:
    """Providers of each unit whose status is ready, unexpired and on this revision."""
    keys = {records.piece_key(repo, revision, uid): uid for uid in unit_ids}
    limiter = trio.CapacityLimiter(constants.LOOKUP_CONCURRENCY)
    logger.debug("looking up providers units=%d", len(keys))
    providers = await lookup.find_providers(dht, list(keys), limiter)

    # Each provider's status is fetched once, however many pieces it provides.
    peers = set().union(*providers.values())
    logger.debug("fetching peer statuses peers=%d", len(peers))
    ready_statuses: PeerStatuses = {}
    async with trio.open_nursery() as nursery:
        for peer in peers:
            nursery.start_soon(
                _fetch_ready_statuses,
                dht,
                repo,
                revision,
                peer,
                limiter,
                ready_statuses,
            )

    return {
        unit_id: {
            peer: ready_statuses[peer]
            for peer in providers[key]
            if peer in ready_statuses
        }
        for key, unit_id in keys.items()
    }


async def _fetch_ready_statuses(
    dht: KadDHT,
    repo: str,
    revision: str,
    peer: str,
    limiter: trio.CapacityLimiter,
    ready_statuses: PeerStatuses,
) -> None:
    """Record the peer's status only if it is ready, unexpired and on this revision."""
    key = records.peer_key(peer)
    async with limiter:
        raw = await lookup.get_value(dht, key)
    try:
        records.RecordValidator().validate(key, raw)
    except (TypeError, ValueError):
        logger.debug("peer status missing or invalid peer=%s", peer)
        return
    status = json.loads(raw)
    if (
        status["ready"]
        and status["expires"] > time.time()
        and (status["repo"], status["revision"]) == (repo, revision)
    ):
        execution, stage = status.get("exec"), status.get("stage")
        ready_statuses[peer] = models.PeerStatus(
            status["addrs"],
            execution
            and execution_models.Endpoint(execution["host"], execution["port"]),
            stage and execution_models.StageDescriptor(**stage),
        )
    else:
        logger.debug("peer status not ready, expired or on another pin peer=%s", peer)
