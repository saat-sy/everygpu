"""Route selection: the fewest contiguous executable stages from embedding to final."""

from client.network import models


def select(checkpoint: models.Checkpoint, hops: list[models.Hop]) -> models.Route:
    """Deterministic shortest chain; ties go to the lowest peer id. Hops must be peer-ordered."""
    chain = _chain(hops, checkpoint.layers, 0, False, {})
    if chain is None:
        raise models.RouteError(
            "no complete executable route: need ready workers whose stages run "
            f"the embedding, layers 0-{checkpoint.layers - 1} with all "
            f"{checkpoint.experts} experts each, and the final head"
        )
    return models.Route(checkpoint.repo, checkpoint.revision, chain)


def _chain(
    hops: list[models.Hop],
    layers: int,
    position: int,
    embedded: bool,
    memo: dict,
) -> tuple[models.Hop, ...] | None:
    """Shortest hops from this position to the final head, or None."""
    key = (position, embedded)
    if key not in memo:
        best = None
        for hop in hops:
            rest = _rest(hops, layers, hop, position, embedded, memo)
            if rest is not None and (best is None or 1 + len(rest) < len(best)):
                best = (hop, *rest)
        memo[key] = best
    return memo[key]


def _rest(hops, layers, hop, position, embedded, memo) -> tuple | None:
    """Hops after this one when it can run next, else None."""
    stage = hop.stage
    if stage.first != position or stage.embedding == embedded:
        return None
    if stage.stop == stage.first and not (stage.embedding or stage.final):
        return None
    if stage.final:
        return () if stage.stop == layers else None
    return _chain(hops, layers, stage.stop, True, memo)
