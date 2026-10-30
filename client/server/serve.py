"""Starts the client HTTP server from environment configuration."""

import os
import sys

import uvicorn

from client.server import app
from client.server import config
from client.utils import constants


def run() -> int:
    """Bind the configured address and serve until interrupted."""
    value = os.environ.get(constants.HOST_ENV, constants.DEFAULT_ADDRESS)
    try:
        host, port = config.parse_address(value)
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
        server = uvicorn.Server(uvicorn.Config(app.create_app()))
        try:
            server.run(sockets=[sock])
        except KeyboardInterrupt:
            pass
    return 0 if server.started else 1
