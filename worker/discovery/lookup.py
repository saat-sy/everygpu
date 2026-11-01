"""Record lookups: connected peers answer first; a bounded DHT search covers what they lack."""

import time

import trio
from libp2p.kad_dht.kad_dht import KadDHT

from worker.discovery import records
from worker.utils import constants


async def find_providers(
    dht: KadDHT, keys: list[str], limiter: trio.CapacityLimiter
) -> dict[str, set[str]]:
    """Provider peer ids per key; keys no search answered in time map to no providers."""
    found: dict[str, set[str]] = {key: set() for key in keys}
    started = time.monotonic()
    connected_limiter = trio.CapacityLimiter(constants.CONNECTED_CONCURRENCY)
    async with trio.open_nursery() as nursery:
        for key in keys:
            nursery.start_soon(
                _ask_connected_providers, dht, key, connected_limiter, found
            )
    missing = [key for key in keys if not found[key]]
    print(
        f"discovery: connected peers know providers for {len(keys) - len(missing)}/"
        f"{len(keys)} units ({time.monotonic() - started:.1f}s)",
        flush=True,
    )
    if missing:
        with trio.move_on_after(constants.SEARCH_PHASE_SECONDS):
            async with trio.open_nursery() as nursery:
                for key in missing:
                    nursery.start_soon(_search_providers, dht, key, limiter, found)
        print(
            f"discovery: DHT search left {sum(not found[k] for k in missing)}/"
            f"{len(missing)} units without providers ({time.monotonic() - started:.1f}s)",
            flush=True,
        )
    return found


async def get_value(dht: KadDHT, key: str) -> bytes | None:
    """The newest valid record connected peers hold for a key, else a bounded DHT search."""
    values: list[bytes] = []
    async with trio.open_nursery() as nursery:
        for peer in dht.host.get_connected_peers():
            nursery.start_soon(_ask_connected_value, dht, peer, key, values)
    valid = [value for value in values if _is_valid(key, value)]
    if valid:
        return valid[records.RecordValidator().select(key, valid)]
    with trio.move_on_after(constants.SEARCH_PHASE_SECONDS):
        return await dht.get_value(key)
    return None


def _is_valid(key: str, value: bytes) -> bool:
    try:
        records.RecordValidator().validate(key, value)
    except (TypeError, ValueError):
        return False
    return True


async def _ask_connected_providers(
    dht: KadDHT, key: str, limiter: trio.CapacityLimiter, found: dict[str, set[str]]
) -> None:
    async with limiter:
        async with trio.open_nursery() as nursery:
            for peer in dht.host.get_connected_peers():
                nursery.start_soon(_ask_provider, dht, peer, key, found)


async def _ask_provider(dht, peer, key: str, found: dict[str, set[str]]) -> None:
    with trio.move_on_after(constants.CONNECTED_QUERY_SECONDS):
        # shortcut: private provider RPC, libp2p pinned; avoids the global search
        infos, _ = await dht.provider_store._get_providers_from_peer_with_closers(
            peer, key.encode()
        )
        found[key].update(str(info.peer_id) for info in infos)


async def _search_providers(
    dht: KadDHT, key: str, limiter: trio.CapacityLimiter, found: dict[str, set[str]]
) -> None:
    async with limiter:
        infos = await dht.find_providers(key, constants.PROVIDER_LOOKUP_COUNT)
    found[key].update(str(info.peer_id) for info in infos)


async def _ask_connected_value(dht, peer, key: str, values: list[bytes]) -> None:
    with trio.move_on_after(constants.CONNECTED_QUERY_SECONDS):
        # shortcut: private value RPC, libp2p pinned; avoids the global search
        value = await dht.value_store._get_from_peer(peer, key.encode())
        if value is not None:
            values.append(value)
