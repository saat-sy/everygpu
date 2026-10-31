"""An admitted session: its route and the worker connections held for its lifetime."""

import trio

from client.network import models
from client.utils import constants
from worker.execution import channel as channel_lib
from worker.execution import models as execution_models
from worker.execution import route as route_lib
from worker.utils import constants as worker_constants

CONNECTION_ERRORS = (
    OSError,
    trio.BrokenResourceError,
    trio.ClosedResourceError,
    execution_models.ProtocolError,
)


class Session:
    """Holds one owner connection per admitted hop; `channels` follows route order."""

    def __init__(self, session_id: str, route: models.Route):
        self.session_id = session_id
        self.route = route
        self.wire_route = route_lib.encode(route.hops)
        self.channels: list[channel_lib.Channel] = []
        self.opened: list[channel_lib.Channel] = []

    async def release(self) -> None:
        """Release every admitted hop on a fresh connection, then close ours; bounded, uncancellable.

        A fresh connection never queues behind an in-flight step on the owner connection.
        """
        with trio.CancelScope(shield=True):
            with trio.move_on_after(constants.RELEASE_TIMEOUT_SECONDS):
                async with trio.open_nursery() as nursery:
                    for hop in self.route.hops[: len(self.channels)]:
                        nursery.start_soon(self._release_one, hop)
            for channel in self.opened:
                channel.abort()
        self.channels.clear()
        self.opened.clear()

    async def _release_one(self, hop: models.Hop) -> None:
        frame = execution_models.Frame(
            worker_constants.MSG_RELEASE, {"session_id": self.session_id}
        )
        try:
            stream = await trio.open_tcp_stream(hop.endpoint.host, hop.endpoint.port)
            channel = channel_lib.Channel(stream)
            try:
                await channel.send(frame)
                await channel.receive()
            finally:
                channel.abort()
        except CONNECTION_ERRORS:
            pass
