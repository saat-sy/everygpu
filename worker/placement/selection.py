"""Placement policy: which units to load given coverage and a byte budget."""


def choose(
    groups: list[list[str]],
    sizes: dict[str, int],
    copies: dict[str, int],
    budget: int,
) -> list[str]:
    """Missing whole groups first, then other missing units; scarcest copies once covered."""
    order = [unit_id for group in groups for unit_id in group]
    chosen: list[str] = []
    left = budget
    for group in groups:
        if not any(copies.get(u) for u in group):
            if sum(sizes[u] for u in group) <= left:
                left = _take(group, sizes, chosen, left)
    left = _take([u for u in order if not copies.get(u)], sizes, chosen, left)
    if all(copies.get(u) or u in chosen for u in order):
        rank = {unit_id: i for i, unit_id in enumerate(order)}
        scarcest = sorted(order, key=lambda u: (copies.get(u, 0), rank[u]))
        _take(scarcest, sizes, chosen, left)
    return chosen


def _take(candidates: list[str], sizes: dict[str, int], chosen: list, left: int) -> int:
    """Append each unchosen candidate that still fits; return the bytes left."""
    for unit_id in candidates:
        if unit_id not in chosen and sizes[unit_id] <= left:
            chosen.append(unit_id)
            left -= sizes[unit_id]
    return left
