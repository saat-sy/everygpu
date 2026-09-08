"""Connected inference runtimes."""

import asyncio
import json

from fastapi import HTTPException, WebSocket


class RuntimePool:
    def __init__(self, required: int):
        self.required = required
        self._connections: dict[int, WebSocket] = {}
        self._inboxes: dict[int, asyncio.Queue[str | bytes]] = {}
        self._ready: set[int] = set()

    @property
    def connected(self) -> int:
        return len(self._connections)

    @property
    def ready(self) -> int:
        return len(self._ready)

    @property
    def ready_stages(self) -> list[int]:
        return sorted(self._ready)

    def connect(self, websocket: WebSocket) -> int | None:
        available = set(range(self.required)) - set(self._connections)
        if not available:
            return None

        stage = min(available)
        self._connections[stage] = websocket
        self._inboxes[stage] = asyncio.Queue()
        print(f"Runtime connected ({self.connected}/{self.required})")
        return stage

    def socket(self, stage: int) -> WebSocket | None:
        return self._connections.get(stage)

    def connections(self) -> list[tuple[int, WebSocket]]:
        return list(self._connections.items())

    async def listen(self, stage: int) -> None:
        websocket = self._connections[stage]
        inbox = self._inboxes[stage]
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return

            text = message.get("text")
            if text == f"ready {stage}":
                self._ready.add(stage)
                print(f"Runtime {stage} ready ({self.ready}/{self.required})")
            elif text is not None:
                await inbox.put(text)
            elif message.get("bytes") is not None:
                await inbox.put(message["bytes"])

    async def receive(self, stage: int) -> str | bytes:
        inbox = self._inboxes.get(stage)
        if inbox is None:
            raise HTTPException(503, f"runtime {stage} disconnected")
        return await inbox.get()

    async def disconnect(self, stage: int) -> None:
        inbox = self._inboxes.pop(stage, None)
        if inbox is not None:
            await inbox.put(
                json.dumps(
                    {"type": "error", "message": f"runtime {stage} disconnected"}
                )
            )
        self._connections.pop(stage, None)
        self._ready.discard(stage)
        print(f"Runtime disconnected ({self.connected}/{self.required})")
