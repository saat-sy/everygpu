"""Placement policy: the stage to load, aligned with the executable stages already advertised."""


def choose(
    groups: list[list[str]],
    sizes: dict[str, int],
    spans: list[tuple[int, int]],
    budget: int,
) -> list[str]:
    """Units of a run of uncovered groups, else of the scarcest advertised stage that fits."""
    weights = [sum(sizes[unit_id] for unit_id in group) for group in groups]
    copies = [sum(a <= i < b for a, b in spans) for i in range(len(groups))]
    if 0 in copies:
        best = _widest_uncovered_run(copies, weights, budget)
        if best is None:
            raise ValueError(
                f"weight budget {budget:,} bytes cannot hold the smallest uncovered "
                f"stage component; raise --max-vram or use a larger GPU"
            )
    else:
        fitting = [s for s in set(spans) if sum(weights[s[0] : s[1]]) <= budget]
        if not fitting:
            raise ValueError(
                f"every group is covered and weight budget {budget:,} bytes fits "
                "no advertised stage to replicate"
            )
        best = min(fitting, key=lambda s: (sum(copies[s[0] : s[1]]), s))
    return [unit_id for group in groups[best[0] : best[1]] for unit_id in group]


def _widest_uncovered_run(copies, weights, budget) -> tuple[int, int] | None:
    best = None
    for first in range(len(copies)):
        for stop in range(first + 1, len(copies) + 1):
            if copies[stop - 1] or sum(weights[first:stop]) > budget:
                break
            if best is None or stop - first > best[1] - best[0]:
                best = (first, stop)
    return best
