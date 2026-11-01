"""Starts the client HTTP server from command-line and environment configuration."""

import argparse
import logging
import os
import sys

import uvicorn

from client.network import bootstrap
from client.network import service
from client.server import app
from client.server import config
from client.utils import constants
from monitoring.logging import logger as log_utils

logger = logging.getLogger(__name__)


def run(argv: list[str]) -> int:
    """Bind the configured address and serve until interrupted."""
    args = _parser().parse_args(argv)
    try:
        log_utils.configure()
    except ValueError as exc:
        print(f"client: {exc}", file=sys.stderr)
        return 2
    value = os.environ.get(constants.HOST_ENV, constants.DEFAULT_ADDRESS)
    try:
        host, port = config.parse_address(value)
        peers = bootstrap.resolve(args.bootstrap, os.environ)
    except ValueError as exc:
        logger.error("invalid configuration: %s", exc)
        return 2
    try:
        sock = config.bind(host, port)
    except OSError as exc:
        logger.error("cannot listen on %s: %s", value, exc)
        return 1
    with sock:
        logger.info(
            "listening url=%s bootstrap_peers=%d",
            config.url(host, port),
            len(peers or ()),
        )
        if not peers:
            logger.warning(
                "no bootstrap peers: pass --bootstrap or set %s; chat requests return 503",
                constants.BOOTSTRAP_ENV,
            )
        server = uvicorn.Server(
            uvicorn.Config(
                app.create_app(service.Service(peers)),
                log_config=None,
                access_log=False,
            )
        )
        try:
            server.run(sockets=[sock])
        except KeyboardInterrupt:
            pass
    if not server.started:
        logger.error("server did not start")
        return 1
    logger.info("client stopped")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="client",
        description="Serves the OpenAI-compatible API over workers found through the DHT.",
    )
    parser.add_argument(
        "--bootstrap",
        nargs="+",
        metavar="ADDRESS",
        help=f"peers to discover workers through (default: {constants.BOOTSTRAP_ENV})",
    )
    return parser
