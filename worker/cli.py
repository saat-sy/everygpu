"""Command-line entrypoint: run a worker or inspect network coverage."""

import argparse
import logging
import math
import sys

import huggingface_hub.errors
import trio

from model_partitioner.utils import constants as partitioner_constants
from worker.commands import report
from worker.commands import serve
from worker.execution import endpoint
from worker.utils import constants
from worker.utils import models


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
    if args.command is None and args.new_network == bool(args.bootstrap):
        print("error: pass --new-network or --bootstrap ADDRESS", file=sys.stderr)
        sys.exit(partitioner_constants.EXIT_FAILED)
    if args.max_vram is not None and (
        not math.isfinite(args.max_vram) or args.max_vram <= 0
    ):
        print("error: --max-vram must be positive and finite", file=sys.stderr)
        sys.exit(partitioner_constants.EXIT_FAILED)
    logging.getLogger(constants.LIBP2P_LOGGER).setLevel(logging.CRITICAL)
    handler = report.run if args.command == "inspect" else serve.run
    status = partitioner_constants.EXIT_FAILED
    try:
        status = trio.run(handler, args)
    except* KeyboardInterrupt:
        print("\nstopped")
        status = partitioner_constants.EXIT_OK
    except* models.NoGPU as group:
        _print_errors(group)
        status = partitioner_constants.EXIT_INCOMPLETE
    except* (
        ValueError,
        RuntimeError,
        OSError,
        huggingface_hub.errors.HfHubHTTPError,
    ) as group:
        _print_errors(group)
    sys.exit(status)


def _print_errors(group: BaseExceptionGroup) -> None:
    for error in group.exceptions:
        if isinstance(error, BaseExceptionGroup):
            _print_errors(error)
        else:
            print(f"error: {error}", file=sys.stderr)
