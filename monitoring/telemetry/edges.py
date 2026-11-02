"""Per-edge exchange accounting: paired timing, residual, bytes, throughput and slow-link checks."""

from monitoring.telemetry import constants
from monitoring.telemetry import metrics
from monitoring.telemetry import models
from monitoring.telemetry import slow
from monitoring.telemetry import timing


def record(
    edge: models.Edge,
    span,
    seconds: float,
    request_bytes: int,
    reply_bytes: int,
    reply_meta: dict,
    expected: bool,
) -> models.Downstream:
    """Record one successful exchange; returns communication totals for this edge and below it."""
    labels = {
        "edge": edge.kind,
        "phase": edge.phase,
        "connection": edge.connection,
    }
    metrics.record("hop.exchange", seconds, **labels)
    metrics.record("hop.request_bytes", request_bytes, **labels)
    metrics.record("hop.reply_bytes", reply_bytes, **labels)
    if seconds > 0:
        metrics.record(
            "hop.throughput", (request_bytes + reply_bytes) / seconds, **labels
        )
    span.set_attribute("everygpu.exchange_seconds", seconds)
    span.set_attribute("everygpu.request_bytes", request_bytes)
    span.set_attribute("everygpu.reply_bytes", reply_bytes)
    if not expected:
        return models.Downstream()
    parsed = timing.parse_reply(seconds, reply_meta)
    if parsed is None:
        metrics.incomplete("missing_or_invalid_timing")
        span.set_attribute("everygpu.timing_available", False)
        return models.Downstream(0.0, 1, 1)
    residual = max(0.0, seconds - parsed.handling_seconds)
    metrics.record("hop.handling", parsed.handling_seconds, **labels)
    metrics.record("hop.communication", residual, **labels)
    span.set_attribute("everygpu.timing_available", True)
    span.set_attribute("everygpu.handling_seconds", parsed.handling_seconds)
    span.set_attribute("everygpu.communication_estimate_seconds", residual)
    slow.check(
        constants.KIND_COMMUNICATION,
        (
            edge.source,
            edge.target,
            edge.phase,
            edge.connection,
            timing.bucket(request_bytes),
        ),
        residual,
        span,
        {
            "source": edge.source,
            "target": edge.target,
            "step": edge.step,
            "request_bytes": request_bytes,
            "connection": edge.connection,
            "phase": edge.phase,
        },
        phase=edge.phase,
        connection=edge.connection,
    )
    below = parsed.downstream
    if below.incomplete:
        metrics.incomplete("downstream_timing")
    return models.Downstream(
        residual + below.communication_seconds, 1 + below.edges, below.incomplete
    )
