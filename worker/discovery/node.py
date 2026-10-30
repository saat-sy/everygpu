"""libp2p host and Kademlia DHT lifecycle: listening, bootstrap and routing-table upkeep."""

from contextlib import asynccontextmanager

import trio
from libp2p import new_host
from libp2p.abc import INotifee
from libp2p.kad_dht.kad_dht import DHTMode, KadDHT
from libp2p.tools.anyio_service import background_trio_service
from libp2p.tools.utils import info_from_p2p_addr
from libp2p.utils import address_validation
from multiaddr import Multiaddr

from worker.discovery import records
from worker.utils import constants


@asynccontextmanager
async def open_node(bootstrap: list[str] | None, client: bool = False):
    """Run a host and DHT; with bootstrap peers, fail rather than found a separate network."""
    host = new_host()
    port = address_validation.find_free_port()
    async with host.run(listen_addrs=address_validation.get_available_interfaces(port)):
        if bootstrap:
            await _connect_any(host, bootstrap)
        dht = KadDHT(host, DHTMode.CLIENT if client else DHTMode.SERVER)
        dht.register_validator(constants.RECORD_NAMESPACE, records.RecordValidator())
        # shortcut: private attribute; default drops bulk ADD_PROVIDER
        dht._provider_rate_max = constants.PROVIDER_RATE_MAX
        # Departed peers otherwise cost each query a timeout.
        host.get_network().register_notifee(_DepartedPeerRemover(dht))
        for peer in host.get_connected_peers():
            await dht.routing_table.add_peer(peer)
        async with background_trio_service(dht):
            yield dht


def peer_addresses(dht: KadDHT) -> list[str]:
    """Addresses other peers can dial this node on."""
    return [str(address) for address in dht.host.get_addrs()]


async def _connect_any(host, bootstrap: list[str]) -> None:
    errors = []
    for address in bootstrap:
        try:
            with trio.fail_after(constants.BOOTSTRAP_TIMEOUT_SECONDS):
                await host.connect(info_from_p2p_addr(Multiaddr(address)))
        except Exception as error:
            errors.append(f"{address}: {type(error).__name__}")
    if len(errors) == len(bootstrap):
        raise RuntimeError(f"cannot join through any bootstrap peer: {errors}")


class _DepartedPeerRemover(INotifee):
    """Network listener: drops a peer from the routing table when its last connection closes."""

    def __init__(self, dht: KadDHT):
        self.dht = dht

    async def disconnected(self, network, conn) -> None:
        peer = conn.muxed_conn.peer_id
        if not network.get_connections(peer):
            self.dht.routing_table.remove_peer(peer)

    # The remaining INotifee methods are required by the interface and unused.
    async def opened_stream(self, network, stream) -> None:
        pass

    async def closed_stream(self, network, stream) -> None:
        pass

    async def connected(self, network, conn) -> None:
        pass

    async def listen(self, network, multiaddr) -> None:
        pass

    async def listen_close(self, network, multiaddr) -> None:
        pass
