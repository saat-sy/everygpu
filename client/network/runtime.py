"""Bridge from asyncio to a dedicated trio thread, where the libp2p runtime lives."""

import asyncio
import contextlib
import threading

import trio

from client.network import models
from client.utils import constants

_ITEM, _ERROR, _END = "item", "error", "end"


class TrioRuntime:
    """Runs one trio loop in a thread; streams of trio work are cancelled with their asyncio reader."""

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._token: trio.lowlevel.TrioToken | None = None
        self._nursery: trio.Nursery | None = None
        self._started = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=trio.run, args=(self._main,), name="everygpu-network", daemon=True
        )
        self._thread.start()
        self._started.wait()

    async def stop(self) -> None:
        """Cancel everything running and wait for the thread to finish cleanup."""
        if self._thread is None:
            return
        with contextlib.suppress(trio.RunFinishedError):
            self._token.run_sync_soon(self._nursery.cancel_scope.cancel)
        await asyncio.to_thread(self._thread.join, constants.STOP_TIMEOUT_SECONDS)

    def spawn(self, function, *args) -> None:
        """Run a trio function in the background until the runtime stops."""
        try:
            self.call_soon(self._nursery.start_soon, function, *args)
        except trio.RunFinishedError:
            raise models.RouteError("the network service is not running") from None

    def call_soon(self, function, *args) -> None:
        """Run a plain function on the trio thread; raises trio.RunFinishedError once stopped."""
        self._token.run_sync_soon(function, *args)

    def stream(self, function, *args) -> "Stream":
        """Start `function(emit, *args)` in trio; its emitted items are read through the Stream."""
        stream = Stream(self)
        self.spawn(stream.run, function, *args)
        return stream

    async def _main(self) -> None:
        async with trio.open_nursery() as nursery:
            self._nursery = nursery
            self._token = trio.lowlevel.current_trio_token()
            self._started.set()
            await trio.sleep_forever()


class Stream:
    """Asyncio reader of a trio producer, which stays a bounded number of items ahead."""

    def __init__(self, runtime: TrioRuntime):
        self._runtime = runtime
        self._loop = asyncio.get_running_loop()
        self._items: asyncio.Queue = asyncio.Queue()
        self._credits = trio.Semaphore(constants.STREAM_BUFFER_ITEMS)
        self._scope = trio.CancelScope()
        self._finished = self._loop.create_future()

    async def next(self):
        """The next item; re-raises the producer's failure."""
        kind, value = await self._items.get()
        if kind == _ERROR:
            raise value
        if kind == _END:
            raise models.GenerationError("generation ended without a result")
        with contextlib.suppress(trio.RunFinishedError):
            self._runtime.call_soon(self._credits.release)
        return value

    async def aclose(self) -> None:
        """Cancel the producer now, then wait a bounded time for its cleanup. Idempotent."""
        try:
            self._runtime.call_soon(self._scope.cancel)
        except trio.RunFinishedError:
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(self._finished), constants.CLEANUP_WAIT_SECONDS
            )

    async def run(self, function, *args) -> None:
        """Trio side: drive the producer, then report how it ended."""
        try:
            with self._scope:
                await function(self._emit, *args)
            self._put(_END, None)
        except Exception as error:
            self._put(_ERROR, error)
        finally:
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._settle)

    async def _emit(self, item) -> None:
        try:
            with trio.fail_after(constants.STREAM_STALL_SECONDS):
                await self._credits.acquire()
        except trio.TooSlowError:
            raise models.GenerationError(
                "the client stopped reading the response"
            ) from None
        self._put(_ITEM, item)

    def _put(self, kind: str, value) -> None:
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._items.put_nowait, (kind, value))

    def _settle(self) -> None:
        if not self._finished.done():
            self._finished.set_result(None)
