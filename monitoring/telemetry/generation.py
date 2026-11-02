"""Sampled-token arrival clock: TTFT, arrival intervals and per-request TPOT."""

import time

from monitoring.telemetry import metrics


class TokenClock:
    """Timestamps each sampled token (EOS included) on the client's monotonic clock."""

    def __init__(self, request_started: float | None):
        self._request_started = request_started
        self._first = self._last = 0.0
        self.count = 0

    def arrive(self) -> None:
        now = time.monotonic()
        if self.count == 0:
            self._first = now
            if self._request_started:
                metrics.record("request.ttft", now - self._request_started)
        else:
            metrics.record("token.interval", now - self._last)
        self._last = now
        self.count += 1

    def finish(self) -> None:
        """Record the sampled count and, from two tokens on, the mean time per output token."""
        metrics.record("tokens.sampled", self.count)
        if self.count >= 2:
            metrics.record(
                "request.tpot", (self._last - self._first) / (self.count - 1)
            )
