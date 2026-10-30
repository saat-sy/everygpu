"""The inspect command: report network coverage, missing pieces and serving workers."""

import trio

from model_partitioner.utils import constants as partitioner_constants
from worker.discovery import discovery
from worker.discovery import node
from worker.loading import manifest as manifest_loading
from worker.placement import order
from worker.placement import summary


async def run(args) -> int:
    async with node.open_node(args.bootstrap, client=True) as dht:
        repo, revision = await discovery.read_network_pin(dht)
        print(f"artifact: {repo} @ {revision}")
        manifest = await trio.to_thread.run_sync(
            manifest_loading.fetch_manifest, repo, revision, abandon_on_cancel=True
        )
        unit_ids = order.model_order(manifest)
        found = await discovery.find_coverage(dht, repo, revision, unit_ids)
    missing = [unit_id for unit_id in unit_ids if not found[unit_id]]
    print(f"covered: {len(unit_ids) - len(missing)}/{len(unit_ids)} units")
    if missing:
        print("not found in the DHT now:")
        print("  " + "\n  ".join(summary.describe(manifest, set(missing))))
    workers: dict[str, list[str]] = {}
    addrs: dict[str, list[str]] = {}
    for unit_id in unit_ids:
        for peer, peer_addrs in found[unit_id].items():
            workers.setdefault(peer, []).append(unit_id)
            addrs[peer] = peer_addrs
    for peer, served in workers.items():
        print(f"worker {peer} ({len(served)} units) at {', '.join(addrs[peer])}")
        print("  " + "\n  ".join(summary.describe(manifest, set(served))))
    return partitioner_constants.EXIT_OK
