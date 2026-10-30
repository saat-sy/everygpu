"""Advertising: this worker's network pin, readiness status and piece provider records."""

import json
import time

import trio
from libp2p.kad_dht.kad_dht import KadDHT

from worker.discovery import records
from worker.execution import models as execution_models
from worker.utils import constants


class Advertiser:
    """Owns this worker's readiness and every record it writes to the DHT."""

    def __init__(self, dht: KadDHT, repo: str, revision: str):
        self.dht = dht
        self.repo = repo
        self.revision = revision
        self.ready = False
        self.execution: execution_models.Endpoint | None = None

    async def publish(self) -> None:
        """Store the network pin and, once ready, the ready status."""
        await self._publish_network_pin()
        if self.ready:
            await self._publish_peer_status(ready=True)

    async def announce_pieces(self, unit_ids: list[str]) -> list[str]:
        """Add this peer as a provider of each piece; return pieces no peer was sent."""
        keys = {
            records.piece_key(self.repo, self.revision, unit_id): unit_id
            for unit_id in unit_ids
        }
        # A lone first worker holds its own records, so no peer needs to be reached.
        is_first_worker = self.dht.get_routing_table_size() == 0
        reached_peer: dict[str, bool] = {}
        limiter = trio.CapacityLimiter(constants.LOOKUP_CONCURRENCY)
        async with trio.open_nursery() as nursery:
            for key in keys:
                nursery.start_soon(self._announce_piece, key, limiter, reached_peer)
        if is_first_worker:
            return []
        return [unit_id for key, unit_id in keys.items() if not reached_peer[key]]

    async def refresh_forever(self) -> None:
        while True:
            await trio.sleep(constants.REFRESH_SECONDS)
            try:
                await self.publish()
            except Exception as error:
                print(f"warning: discovery degraded: {error}")

    async def withdraw(self) -> None:
        self.ready = False
        await self._publish_peer_status(ready=False)

    async def _publish_network_pin(self) -> None:
        pin = {"repo": self.repo, "revision": self.revision}
        value = json.dumps(pin, sort_keys=True)
        await self.dht.put_value(constants.NETWORK_KEY, value.encode())

    async def _publish_peer_status(self, ready: bool) -> None:
        status = {
            "repo": self.repo,
            "revision": self.revision,
            "addrs": [str(address) for address in self.dht.host.get_addrs()],
            "ready": ready,
            "exec": self.execution
            and {"host": self.execution.host, "port": self.execution.port},
            "expires": time.time() + constants.EXPIRATION_SECONDS,
            "seq": time.time_ns(),
        }
        key = records.peer_key(str(self.dht.host.get_id()))
        await self.dht.put_value(key, json.dumps(status).encode())

    async def _announce_piece(
        self, key: str, limiter: trio.CapacityLimiter, reached_peer: dict[str, bool]
    ) -> None:
        async with limiter:
            reached_peer[key] = await self.dht.provide(key)
