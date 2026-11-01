"""Executable coverage: which ready workers run which embedding, layer and final groups."""

from worker.discovery import models

Spans = dict[
    str, tuple[int, int]
]  # peer id -> its stage as group indices [start, stop)


def executable_spans(
    groups: list[list[str]], found: dict[str, dict[str, models.PeerStatus]]
) -> Spans:
    """Stage span of each ready worker that advertises an endpoint and provides every unit it runs."""
    layers = len(groups) - 2
    statuses = {p: s for peers in found.values() for p, s in peers.items()}
    spans = {}
    for peer, status in statuses.items():
        stage = status.stage
        if status.execution is None or stage is None or stage.layers != layers:
            continue
        start = 0 if stage.embedding else 1 + stage.first
        stop = 1 + stage.stop + stage.final
        if start < stop <= len(groups) and all(
            peer in found[unit_id] for group in groups[start:stop] for unit_id in group
        ):
            spans[peer] = (start, stop)
    return spans


def group_copies(spans: Spans, count: int) -> list[int]:
    """Per group, how many workers' executable stages include it."""
    return [sum(a <= i < b for a, b in spans.values()) for i in range(count)]
