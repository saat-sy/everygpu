"""Onboarding: choose, load and announce the pieces this worker will serve."""

import logging
import time

import trio

from monitoring.logging import logger as log_utils
from worker.discovery import advertiser
from worker.discovery import discovery
from worker.execution import forward
from worker.loading import manifest as manifest_loading
from worker.loading import resident
from worker.placement import coverage
from worker.placement import order
from worker.placement import selection
from worker.placement import summary

logger = logging.getLogger(__name__)


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
    spans = coverage.executable_spans(groups, found)
    copies = coverage.group_copies(spans, len(groups))
    logger.info(
        "coverage discovered executable_groups=%d/%d",
        sum(map(bool, copies)),
        len(groups),
    )
    chosen = selection.choose(
        groups, order.unit_bytes(manifest), list(spans.values()), weight_budget
    )
    logger.info("placement selected units=%d of %d", len(chosen), len(unit_ids))
    for line in summary.describe(manifest, set(chosen)):
        logger.debug("placement %s", line)
    started = time.monotonic()
    weights = await trio.to_thread.run_sync(
        resident.load, repo, revision, manifest, chosen, abandon_on_cancel=True
    )
    logger.info(
        "weights resident units=%d bytes=%d elapsed_ms=%d",
        len(weights),
        sum(t.nbytes for u in weights.values() for t in u.values()),
        log_utils.elapsed_ms(started),
    )
    try:
        runner.install(manifest.config, weights)
    except ValueError as error:
        raise RuntimeError(
            f"selected pieces form no executable stage: {error}"
        ) from error
    ads.stage = runner.descriptor()
    logger.info(
        "stage installed layers=%d-%d embedding=%s final=%s",
        ads.stage.first,
        ads.stage.stop,
        ads.stage.embedding,
        ads.stage.final,
    )
    started = time.monotonic()
    failed = await ads.announce_pieces(chosen)
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(chosen)} piece announcements not sent"
        )
    logger.info(
        "pieces announced units=%d elapsed_ms=%d",
        len(chosen),
        log_utils.elapsed_ms(started),
    )
    ads.ready = True
    await ads.publish()
    return weights
