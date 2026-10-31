"""Bridge from asyncio to a dedicated trio thread, where the libp2p runtime lives."""

import asyncio
import contextlib
import threading

import trio

from client.network import models
from client.utils import constants


class TrioRuntime:
    """Runs one trio loop in a thread; leased trio work is cancelled with its asyncio caller."""

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
            self._token.run_sync_soon(self._nursery.start_soon, function, *args)
        except trio.RunFinishedError:
            raise models.RouteError("the network service is not running") from None

    @contextlib.asynccontextmanager
    async def lease(self, factory):
        """Enter a trio async context manager and hold it until this block ends or is cancelled."""
        loop = asyncio.get_running_loop()
        entered, finished = loop.create_future(), loop.create_future()
        scope = trio.CancelScope()
        self.spawn(self._hold, factory, loop, entered, finished, scope)
        try:
            value = await entered
            yield value
        finally:
            await self._cancel(scope, finished)

    async def _main(self) -> None:
        async with trio.open_nursery() as nursery:
            self._nursery = nursery
            self._token = trio.lowlevel.current_trio_token()
            self._started.set()
            await trio.sleep_forever()

    async def _hold(self, factory, loop, entered, finished, scope) -> None:
        try:
            with scope:
                async with factory() as value:
                    _settle(loop, entered, value)
                    await trio.sleep_forever()
        except Exception as error:
            _settle(loop, entered, error=error)
        finally:
            _settle(loop, entered, error=models.RouteError("the request was cancelled"))
            _settle(loop, finished)

    async def _cancel(self, scope: trio.CancelScope, finished: asyncio.Future) -> None:
        try:
            self._token.run_sync_soon(scope.cancel)
        except trio.RunFinishedError:
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(finished), constants.CLEANUP_WAIT_SECONDS
            )


def _settle(loop, future: asyncio.Future, value=None, error=None) -> None:
    """Complete an asyncio future from the trio thread unless it is already done."""

    with contextlib.suppress(RuntimeError):
        loop.call_soon_threadsafe(_apply, future, value, error)


def _apply(future: asyncio.Future, value, error) -> None:
    if not future.done():
        future.set_exception(error) if error else future.set_result(value)
