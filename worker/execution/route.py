"""Route metadata: encoding and strict validation of the stage chain a session is admitted with."""

import dataclasses

from worker.execution import endpoint
from worker.execution import models
from worker.utils import constants

_HOP_FIELDS = frozenset({"peer", "endpoint", "stage"})
_STAGE_FIELDS = frozenset(f.name for f in dataclasses.fields(models.StageDescriptor))
_INT_FIELDS = ("first", "stop", "layers", "experts")
_BOOL_FIELDS = ("embedding", "final")


def encode(hops: tuple[models.Hop, ...]) -> list[dict]:
    """The wire form of a route's hops."""
    return [
        {
            "peer": hop.peer,
            "endpoint": str(hop.endpoint),
            "stage": dataclasses.asdict(hop.stage),
        }
        for hop in hops
    ]


def parse(meta: dict) -> tuple[tuple[models.Hop, ...], int]:
    """The validated hops and this worker's index; refuses anything but one whole acyclic chain."""
    raw, index = meta.get("route"), meta.get("index")
    if not (isinstance(raw, list) and 0 < len(raw) <= constants.MAX_ROUTE_HOPS) or not (
        type(index) is int and 0 <= index < len(raw)
    ):
        raise _bad("route must be a bounded list with this worker's index")
    hops = tuple(_hop(item) for item in raw)
    _check_chain(hops)
    return hops, index


def _hop(item) -> models.Hop:
    if not (
        isinstance(item, dict)
        and item.keys() == _HOP_FIELDS
        and isinstance(item["peer"], str)
        and 0 < len(item["peer"]) <= constants.SESSION_ID_MAX_CHARS
        and isinstance(item["endpoint"], str)
    ):
        raise _bad("malformed route hop")
    stage = item["stage"]
    if not (
        isinstance(stage, dict)
        and stage.keys() == _STAGE_FIELDS
        and all(type(stage[name]) is int for name in _INT_FIELDS)
        and all(type(stage[name]) is bool for name in _BOOL_FIELDS)
    ):
        raise _bad("malformed route stage")
    try:
        parsed = endpoint.parse(item["endpoint"])
    except ValueError:
        raise _bad("malformed route endpoint") from None
    return models.Hop(item["peer"], parsed, models.StageDescriptor(**stage))


def _check_chain(hops: tuple[models.Hop, ...]) -> None:
    stages = [hop.stage for hop in hops]
    position = 0
    for i, stage in enumerate(stages):
        empty = stage.stop == stage.first and not (stage.embedding or stage.final)
        if (
            stage.first != position
            or stage.stop < stage.first
            or empty
            or stage.embedding != (i == 0)
            or stage.final != (i == len(stages) - 1)
            or (stage.layers, stage.experts) != (stages[0].layers, stages[0].experts)
        ):
            raise _bad("route stages are not one contiguous chain")
        position = stage.stop
    if position != stages[0].layers:
        raise _bad("route stages do not cover every layer")
    if len({hop.peer for hop in hops}) != len(hops) or len(
        {hop.endpoint for hop in hops}
    ) != len(hops):
        raise _bad("route repeats a worker")


def _bad(message: str) -> models.RequestError:
    return models.RequestError("bad_route", message)
