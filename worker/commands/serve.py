"""The worker command: join, choose and load pieces, advertise until stopped."""

import sys

import trio

from model_partitioner.artifact import hub
from model_partitioner.utils import constants as partitioner_constants
from worker.commands import onboard
from worker.discovery import advertiser
from worker.discovery import discovery
from worker.discovery import node
from worker.execution import endpoint
from worker.execution import server as execution_server
from worker.execution import sessions
from worker.loading import budget
from worker.utils import constants


async def run(args) -> int:
    name, free, total, weight_budget = budget.gpu_budget(args.max_vram)
    print(
        f"GPU: {name}, {free / constants.GIB:.2f} GiB free of {total / constants.GIB:.2f} GiB; "
        f"weight budget {weight_budget / constants.GIB:.2f} GiB"
    )
    async with node.open_node(args.bootstrap) as dht:
        print("bootstrap addresses:\n  " + "\n  ".join(node.peer_addresses(dht)))
        if args.new_network:
            repo = constants.DEFAULT_REPO
            revision = await trio.to_thread.run_sync(
                hub.resolve_revision, repo, abandon_on_cancel=True
            )
        else:
            repo, revision = await discovery.read_network_pin(dht)
        ads = advertiser.Advertiser(dht, repo, revision)
        await ads.publish()
        print(f"artifact: {repo} @ {revision}")
        try:
            async with trio.open_nursery() as nursery:
                executor = execution_server.ExecutionServer(sessions.SessionTable())
                port = await nursery.start(
                    executor.serve, args.exec_host, args.exec_port
                )
                ads.execution = endpoint.advertised(
                    args.exec_host,
                    port,
                    args.advertise_exec,
                    node.peer_addresses(dht),
                )
                print(
                    f"execution server: {args.exec_host}:{port}; advertised: "
                    f"{ads.execution or 'none (pass --advertise-exec HOST:PORT)'}"
                )
                nursery.start_soon(ads.refresh_forever)
                units = await onboard.run(dht, ads, weight_budget)
                print(f"READY: {len(units)} units resident; Ctrl+C withdraws and stops")
                await trio.sleep_forever()
        finally:
            if ads.ready:
                try:
                    with trio.move_on_after(
                        constants.WITHDRAW_TIMEOUT_SECONDS, shield=True
                    ) as withdrawal:
                        await ads.withdraw()
                    if withdrawal.cancelled_caught:
                        print(
                            "warning: availability withdrawal timed out",
                            file=sys.stderr,
                        )
                    else:
                        print("availability withdrawn")
                except Exception as error:
                    print(
                        f"warning: availability withdrawal failed: {error}",
                        file=sys.stderr,
                    )
    return partitioner_constants.EXIT_OK
