"""Slow-operation reporting: detector verdict to span event, warning and bounded counter."""

import logging

from monitoring.telemetry import lifecycle
from monitoring.telemetry import metrics

logger = logging.getLogger(__name__)


def check(kind: str, group: tuple, seconds: float, span, identifiers: dict, **labels):
    """Report a heuristic outlier; `labels` are bounded metric labels, `identifiers` are trace-only."""
    state = lifecycle.active()
    if state is None:
        return None
    outlier = state.slow.observe((kind, *group), seconds)
    if outlier is None:
        return None
    span.add_event(
        "slow_operation",
        {
            "kind": kind,
            "seconds": outlier.value,
            "baseline_median_seconds": outlier.median,
            "threshold_seconds": outlier.threshold,
            "baseline_samples": outlier.samples,
            **identifiers,
        },
    )
    metrics.record("slow_operations", kind=kind, **labels)
    logger.warning(
        "slow %s heuristic outlier seconds=%.4f median=%.4f threshold=%.4f %s",
        kind,
        outlier.value,
        outlier.median,
        outlier.threshold,
        " ".join(f"{key}={value}" for key, value in identifiers.items()),
    )
    return outlier
