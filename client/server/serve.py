"""Starts the client HTTP server from command-line and environment configuration."""

import argparse
import os
import sys

import uvicorn

from client.network import bootstrap
from client.network import service
from client.server import app
from client.server import config
from client.utils import constants


def run(argv: list[str]) -> int:
    """Bind the configured address and serve until interrupted."""
    args = _parser().parse_args(argv)
    value = os.environ.get(constants.HOST_ENV, constants.DEFAULT_ADDRESS)
    try:
        host, port = config.parse_address(value)
        peers = bootstrap.resolve(args.bootstrap, os.environ)
    except ValueError as exc:
        print(f"client: {exc}", file=sys.stderr)
        return 2
    try:
        sock = config.bind(host, port)
    except OSError as exc:
        print(f"client: cannot listen on {value}: {exc}", file=sys.stderr)
        return 1
    with sock:
        print(f"Listening on {config.url(host, port)}", flush=True)
        print(
            (
                f"Bootstrap peers: {len(peers)}"
                if peers
                else f"No bootstrap peers: pass --bootstrap or set {constants.BOOTSTRAP_ENV}; "
                "chat requests return 503"
            ),
            flush=True,
        )
        server = uvicorn.Server(uvicorn.Config(app.create_app(service.Service(peers))))
        try:
            server.run(sockets=[sock])
        except KeyboardInterrupt:
            pass
    return 0 if server.started else 1


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
