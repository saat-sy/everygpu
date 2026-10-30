"""Onboarding: choose, load and announce the pieces this worker will serve."""

import sys

import trio

from worker.discovery import advertiser
from worker.discovery import discovery
from worker.execution import forward
from worker.loading import manifest as manifest_loading
from worker.loading import resident
from worker.placement import order
from worker.placement import selection
from worker.placement import summary


async def run(
    dht, ads: advertiser.Advertiser, weight_budget: int, runner: forward.StageRunner
) -> dict:
    """Choose, load and announce pieces; ready only if every step succeeded; return resident units."""
    repo, revision = ads.repo, ads.revision
    manifest = await trio.to_thread.run_sync(
        manifest_loading.fetch_manifest, repo, revision, abandon_on_cancel=True
    )
    groups = order.model_groups(manifest)
    unit_ids = [unit_id for group in groups for unit_id in group]
    found = await discovery.find_coverage(dht, repo, revision, unit_ids)
    copies = {unit_id: len(peers) for unit_id, peers in found.items()}
    print(
        f"discovered coverage: {sum(map(bool, copies.values()))}/{len(unit_ids)} units"
    )
    chosen = selection.choose(groups, order.unit_bytes(manifest), copies, weight_budget)
    if not chosen:
        raise ValueError("the weight budget fits no missing or useful piece")
    print(f"selected {len(chosen)} units:")
    print("  " + "\n  ".join(summary.describe(manifest, set(chosen))))
    weights = await trio.to_thread.run_sync(
        resident.load, repo, revision, manifest, chosen, abandon_on_cancel=True
    )
    print(
        f"resident: {sum(t.nbytes for u in weights.values() for t in u.values()):,} bytes"
    )
    try:
        runner.install(manifest.config, weights)
    except ValueError as error:
        print(f"warning: forward execution disabled: {error}", file=sys.stderr)
    failed = await ads.announce_pieces(chosen)
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(chosen)} piece announcements not sent"
        )
    ads.ready = True
    await ads.publish()
    return weights
