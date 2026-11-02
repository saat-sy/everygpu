"""Pure timing helpers: validation of peer-reported timing and power-of-two bucketing."""

from monitoring.telemetry import constants
from monitoring.telemetry import models


def phase(step: int) -> str:
    """Prefill for the first step, else decode."""
    return constants.PHASE_PREFILL if step == 0 else constants.PHASE_DECODE


def bucket(value: int) -> int:
    """The next power of two at or above a positive size; zero stays zero."""
    return 1 << (value - 1).bit_length() if value > 0 else 0


def parse_reply(exchange_seconds: float, meta: dict) -> models.Timing | None:
    """Validated timing from a reply, or None when absent or inconsistent with the exchange."""
    raw = meta.get(constants.TELEMETRY_KEY) if isinstance(meta, dict) else None
    if not isinstance(raw, dict):
        return None
    handling = raw.get("handling_seconds")
    limit = exchange_seconds + constants.TOLERANCE_SECONDS
    if not _seconds(handling, limit):
        return None
    below = raw.get("communication_seconds", 0.0)
    edges, incomplete = raw.get("edges", 0), raw.get("incomplete", 0)
    if (
        _seconds(below, handling + constants.TOLERANCE_SECONDS)
        and _count(edges, constants.MAX_ROUTE_EDGES)
        and _count(incomplete, edges)
    ):
        return models.Timing(handling, models.Downstream(below, edges, incomplete))
    return models.Timing(handling, models.Downstream(0.0, 1, 1))


def _seconds(value, limit: float) -> bool:
    return type(value) in (int, float) and 0 <= value <= limit


def _count(value, limit: int) -> bool:
    return type(value) is int and 0 <= value <= limit
