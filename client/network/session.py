"""An admitted session: its route and the worker connections held for its lifetime."""

import trio

from client.network import models
from client.utils import constants
from worker.execution import channel as channel_lib
from worker.execution import models as execution_models
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
        self.channels: list[channel_lib.Channel] = []
        self.opened: list[channel_lib.Channel] = []

    async def release(self) -> None:
        """Release admitted hops, then close every connection; bounded and uncancellable."""
        with trio.move_on_after(constants.RELEASE_TIMEOUT_SECONDS, shield=True):
            async with trio.open_nursery() as nursery:
                for channel in self.channels:
                    nursery.start_soon(self._release_one, channel)
        with trio.CancelScope(shield=True):
            for channel in self.opened:
                with trio.move_on_after(constants.RELEASE_TIMEOUT_SECONDS, shield=True):
                    await channel.stream.aclose()
        self.channels.clear()
        self.opened.clear()

    async def _release_one(self, channel: channel_lib.Channel) -> None:
        frame = execution_models.Frame(
            worker_constants.MSG_RELEASE, {"session_id": self.session_id}
        )
        try:
            await channel.send(frame)
            await channel.receive()
        except CONNECTION_ERRORS:
            pass
