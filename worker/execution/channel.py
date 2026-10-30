"""A framed connection: whole-frame sends and receives serialised across tasks."""

import trio

from worker.execution import models
from worker.execution import wire


class Channel:
    """Wraps a trio stream so concurrent tasks never interleave frames."""

    def __init__(self, stream: trio.abc.HalfCloseableStream):
        self.stream = stream
        self._send_lock = trio.Lock()
        self._receive_lock = trio.Lock()

    async def send(self, frame: models.Frame) -> None:
        data = wire.encode(frame)
        async with self._send_lock:
            await self.stream.send_all(data)

    async def receive(self) -> models.Frame | None:
        async with self._receive_lock:
            return await wire.read_frame(self.stream)
