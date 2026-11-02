"""Phase timing: records a duration metric, and the current span attribute, for a wait or a block."""

import contextlib
import time

from monitoring.telemetry import lifecycle
from monitoring.telemetry import metrics
from monitoring.telemetry import tracing


def _finish(key: str, started: float, labels: dict) -> None:
    seconds = time.monotonic() - started
    metrics.record(key, seconds, **labels)
    from opentelemetry import trace

    trace.get_current_span().set_attribute("everygpu." + key, seconds)


@contextlib.contextmanager
def timed(key: str, *, traced: bool = False, **labels):
    """Time the block on the monotonic clock, optionally as its own span; free while disabled."""
    if lifecycle.active() is None:
        yield
        return
    started = time.monotonic()
    with contextlib.ExitStack() as stack:
        if traced:
            stack.enter_context(tracing.span(key))
        try:
            yield
        finally:
            _finish(key, started, labels)


@contextlib.asynccontextmanager
async def waited(lock, key: str, **labels):
    """Hold `lock` for the block, recording only how long acquiring it took."""
    started = time.monotonic()
    async with lock:
        if lifecycle.active() is not None:
            _finish(key, started, labels)
        yield
