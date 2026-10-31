"""Network service: a long-lived DHT client and request-time route discovery and admission."""

import contextlib

import trio

from client.network import admission
from client.network import checkpoint
from client.network import discovery
from client.network import models
from client.network import routing
from client.network import runtime as runtime_lib
from client.utils import constants
from worker.discovery import node


class Service:
    """Keeps a DHT client joined through the bootstrap peers; `session` admits a route per request."""

    def __init__(self, bootstrap: list[str] | None):
        self.bootstrap = bootstrap
        self.runtime = runtime_lib.TrioRuntime()
        self._dht = None
        self._error = "joining the network"
        self._checkpoints: checkpoint.Cache = {}
        self._lookup = trio.Lock()

    async def start(self) -> None:
        """Begin joining in the background; never waits on the network."""
        if self.bootstrap:
            self.runtime.start()
            self.runtime.spawn(self._maintain)

    async def stop(self) -> None:
        if self.bootstrap:
            await self.runtime.stop()

    def session(self):
        """Async context manager: discover, select and admit a route; release on exit."""
        if not self.bootstrap:
            raise models.RouteError("no bootstrap peer is configured")
        return self.runtime.lease(self._open)

    async def _maintain(self) -> None:
        while True:
            try:
                async with node.open_node(self.bootstrap, client=True) as dht:
                    self._dht = dht
                    while dht.get_routing_table_size():
                        await trio.sleep(constants.RECONNECT_SECONDS)
                    self._error = "lost every peer"
            except Exception as error:
                self._error = f"{type(error).__name__}: {error}"
            finally:
                self._dht = None
            await trio.sleep(constants.RECONNECT_SECONDS)

    @contextlib.asynccontextmanager
    async def _open(self):
        route = await self._route()
        async with admission.open_session(route) as session:
            yield session

    async def _route(self) -> models.Route:
        dht = self._dht
        if dht is None:
            raise models.RouteError(f"not connected to the network ({self._error})")
        try:
            async with self._lookup:
                with trio.fail_after(constants.DISCOVERY_TIMEOUT_SECONDS):
                    discovery.forget_cached(dht)
                    pinned = await checkpoint.load(dht, self._checkpoints)
                    hops = await discovery.find_hops(dht, pinned)
        except trio.TooSlowError:
            raise models.RouteError("network discovery timed out") from None
        except models.RouteError:
            raise
        except Exception as error:
            raise models.RouteError(
                f"network discovery failed: {type(error).__name__}: {error}"
            ) from None
        return routing.select(pinned, hops)
