"""Command-line entrypoint: run a worker or inspect network coverage."""

import argparse
import logging
import math
import sys
import time

import huggingface_hub.errors
import trio

from monitoring.logging import logger as log_utils
from monitoring.telemetry import constants as telemetry_constants
from monitoring.telemetry import lifecycle
from model_partitioner.utils import constants as partitioner_constants
from worker.commands import report
from worker.commands import serve
from worker.execution import endpoint
from worker.utils import constants
from worker.utils import models

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="worker",
        description="Joins the network, loads model pieces that fit this GPU and advertises them until Ctrl+C.",
    )
    network = parser.add_mutually_exclusive_group()
    network.add_argument(
        "--new-network",
        action="store_true",
        help=f"start a network pinned to the current {constants.DEFAULT_REPO} revision",
    )
    network.add_argument(
        "--bootstrap", nargs="+", metavar="ADDRESS", help="join through these peers"
    )
    parser.add_argument(
        "--max-vram",
        type=float,
        metavar="GIB",
        help="cap on GPU memory this worker plans to use, in GiB",
    )
    parser.add_argument(
        "--exec-host",
        default=constants.EXEC_BIND_HOST,
        metavar="HOST",
        help="address the execution server binds to (default %(default)s)",
    )
    parser.add_argument(
        "--exec-port",
        type=_port,
        default=constants.EXEC_PORT,
        metavar="PORT",
        help="TCP port the execution server binds to; 0 picks a free one (default %(default)s)",
    )
    parser.add_argument(
        "--advertise-exec",
        type=endpoint.parse,
        metavar="HOST:PORT",
        help="execution endpoint peers should dial, e.g. a tunnel's public address "
        "(default: bind host, or this node's first non-loopback IPv4)",
    )
    sub = parser.add_subparsers(dest="command")
    inspect = sub.add_parser(
        "inspect",
        help="report coverage, missing pieces and serving workers",
        description="Joins as a non-storing DHT client and reports the network's piece coverage.",
    )
    inspect.add_argument(
        "--bootstrap", nargs="+", required=True, metavar="ADDRESS", help="peers to ask"
    )
    return parser


def _port(text: str) -> int:
    if not text.isdigit() or int(text) > 65535:
        raise argparse.ArgumentTypeError(f"invalid port {text!r}")
    return int(text)


def main() -> None:
    try:
        args = _parser().parse_args()
    except SystemExit as usage:
        sys.exit(usage.code and partitioner_constants.EXIT_FAILED)
    try:
        log_utils.configure()
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(partitioner_constants.EXIT_FAILED)
    if args.command is None and args.new_network == bool(args.bootstrap):
        logger.error("pass --new-network or --bootstrap ADDRESS")
        sys.exit(partitioner_constants.EXIT_FAILED)
    if args.max_vram is not None and (
        not math.isfinite(args.max_vram) or args.max_vram <= 0
    ):
        logger.error("--max-vram must be positive and finite")
        sys.exit(partitioner_constants.EXIT_FAILED)
    handler = report.run if args.command == "inspect" else serve.run
    command = args.command or "serve"
    status = partitioner_constants.EXIT_FAILED
    started = time.monotonic()
    logger.info("%s starting", command)
    if args.command is None:
        lifecycle.configure(telemetry_constants.SERVICE_WORKER)
    try:
        status = trio.run(handler, args)
    except* KeyboardInterrupt:
        logger.info("interrupted; shutting down")
        status = partitioner_constants.EXIT_OK
    except* models.NoGPU as group:
        _log_errors(group)
        status = partitioner_constants.EXIT_INCOMPLETE
    except* (
        ValueError,
        RuntimeError,
        OSError,
        huggingface_hub.errors.HfHubHTTPError,
    ) as group:
        _log_errors(group)
    finally:
        lifecycle.shutdown()
    logger.info(
        "%s finished status=%d elapsed_ms=%d",
        command,
        status,
        log_utils.elapsed_ms(started),
    )
    sys.exit(status)


def _log_errors(group: BaseExceptionGroup) -> None:
    for error in group.exceptions:
        if isinstance(error, BaseExceptionGroup):
            _log_errors(error)
        else:
            logger.error("%s", error)
