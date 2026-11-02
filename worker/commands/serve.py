"""The worker command: join, choose and load pieces, advertise until stopped."""

import logging
import time

import trio

from monitoring.logging import logger as log_utils
from monitoring.telemetry import device as device_telemetry
from monitoring.telemetry import lifecycle
from model_partitioner.artifact import hub
from model_partitioner.utils import constants as partitioner_constants
from worker.commands import onboard
from worker.discovery import advertiser
from worker.discovery import discovery
from worker.discovery import node
from worker.execution import endpoint
from worker.execution import forward
from worker.execution import identity
from worker.execution import server as execution_server
from worker.execution import sessions
from worker.loading import budget
from worker.utils import constants

logger = logging.getLogger(__name__)


async def run(args) -> int:
    started = time.monotonic()
    name, free, total, weight_budget = budget.gpu_budget(args.max_vram)
    logger.info(
        "gpu selected name=%s free_gib=%.2f total_gib=%.2f weight_budget_gib=%.2f",
        name,
        free / constants.GIB,
        total / constants.GIB,
        weight_budget / constants.GIB,
    )
    device_telemetry.register()
    async with node.open_node(args.bootstrap) as dht:
        lifecycle.identify(**{"everygpu.peer_id": str(dht.host.get_id())})
        for address in node.peer_addresses(dht):
            logger.info("bootstrap address=%s", address)
        if args.new_network:
            repo = constants.DEFAULT_REPO
            revision = await trio.to_thread.run_sync(
                hub.resolve_revision, repo, abandon_on_cancel=True
            )
        else:
            repo, revision = await discovery.read_network_pin(dht)
        ads = advertiser.Advertiser(dht, repo, revision)
        await ads.publish()
        logger.info("checkpoint pinned repo=%s revision=%s", repo, revision)
        try:
            async with trio.open_nursery() as nursery:
                runner = forward.StageRunner()
                executor = execution_server.ExecutionServer(
                    sessions.SessionTable(),
                    runner.forward,
                    lambda: identity.build(
                        repo, revision, str(dht.host.get_id()), runner.descriptor()
                    ),
                )
                port = await nursery.start(
                    executor.serve, args.exec_host, args.exec_port
                )
                ads.execution = endpoint.advertised(
                    args.exec_host,
                    port,
                    args.advertise_exec,
                    node.peer_addresses(dht),
                )
                logger.info(
                    "execution server listening host=%s port=%d advertised=%s",
                    args.exec_host,
                    port,
                    ads.execution,
                )
                if ads.execution is None:
                    logger.warning(
                        "no execution endpoint advertised: pass --advertise-exec HOST:PORT"
                    )
                nursery.start_soon(ads.refresh_forever)
                units = await onboard.run(dht, ads, weight_budget, runner)
                logger.info(
                    "worker ready units=%d elapsed_ms=%d; Ctrl+C withdraws and stops",
                    len(units),
                    log_utils.elapsed_ms(started),
                )
                await trio.sleep_forever()
        finally:
            if ads.ready:
                try:
                    with trio.move_on_after(
                        constants.WITHDRAW_TIMEOUT_SECONDS, shield=True
                    ) as withdrawal:
                        await ads.withdraw()
                    if withdrawal.cancelled_caught:
                        logger.warning(
                            "availability withdrawal timed out after %ds",
                            constants.WITHDRAW_TIMEOUT_SECONDS,
                        )
                    else:
                        logger.info("availability withdrawn")
                except Exception as error:
                    logger.warning("availability withdrawal failed: %s", error)
    return partitioner_constants.EXIT_OK
