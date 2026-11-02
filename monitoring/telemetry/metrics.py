"""Metric recording facade: one call per observation, a no-op while telemetry is disabled."""

from monitoring.telemetry import constants
from monitoring.telemetry import lifecycle


def record(key: str, value: float = 1, **attributes) -> None:
    """Record a histogram value, or add to a counter, with bounded attributes."""
    state = lifecycle.active()
    if state is None:
        return
    instrument = state.instruments[key]
    if constants.METRICS[key][0] == constants.HISTOGRAM:
        instrument.record(value, attributes)
    else:
        instrument.add(value, attributes)


def incomplete(reason: str) -> None:
    """Count a measurement that could not be made."""
    record("telemetry.incomplete", reason=reason)


def code_label(code) -> str:
    """A remote-supplied error code reduced to the known set."""
    return (
        code
        if isinstance(code, str) and code in constants.KNOWN_CODES
        else constants.OTHER
    )
