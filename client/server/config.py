"""Listening address parsing and socket binding."""

import socket

from client.utils import constants


def parse_address(value: str) -> tuple[str, int]:
    """Split host:port, requiring paired brackets around IPv6 hosts."""
    host, sep, port = value.rpartition(":")
    bracketed = host.startswith("[") and host.endswith("]")
    if bracketed:
        host = host[1:-1]
    if (
        not sep
        or not host
        or "[" in host
        or "]" in host
        or (":" in host and not bracketed)
        or not port.isascii()
        or not port.isdigit()
    ):
        raise ValueError(f"{constants.HOST_ENV} must be host:port, got {value!r}")
    if not 1 <= int(port) <= 65535:
        raise ValueError(f"{constants.HOST_ENV} port must be 1-65535, got {port}")
    return host, int(port)


def bind(host: str, port: int) -> socket.socket:
    """Bind a listening socket; raises OSError if unavailable."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    return socket.create_server((host, port), family=family)


def url(host: str, port: int) -> str:
    """Return the base URL, bracketing IPv6 hosts."""
    shown = f"[{host}]" if ":" in host else host
    return f"http://{shown}:{port}"
