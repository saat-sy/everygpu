"""Execution endpoints: parsing user input and choosing the address peers should dial."""

import ipaddress
import re

import multiaddr

from worker.execution import models
from worker.utils import constants

_BRACKETED = re.compile(r"\[([^\[\]]+)\]:(\d+)")
_PLAIN = re.compile(r"([^\[\]:]+):(\d+)")
_HOSTNAME = re.compile(
    r"(?=.{1,253}$)[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(\.[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*"
)


def parse(text: str) -> models.Endpoint:
    """Parse HOST:PORT with a hostname or IPv4 host, or [ADDR]:PORT with an IPv6 host."""
    bracketed = _BRACKETED.fullmatch(text)
    match = bracketed or _PLAIN.fullmatch(text)
    if match is None or not text.isascii():
        raise ValueError(f"expected HOST:PORT or [IPV6]:PORT, got {text!r}")
    host, port = match.group(1), int(match.group(2))
    if bracketed:
        valid = _is_ip(host, ipaddress.IPv6Address)
    else:
        valid = _is_ip(host, ipaddress.IPv4Address) or (
            _HOSTNAME.fullmatch(host) and not host.replace(".", "").isdigit()
        )
    if not valid or not 0 < port < 65536:
        raise ValueError(f"invalid endpoint {text!r}")
    return models.Endpoint(host, port)


def advertised(
    bind_host: str,
    bound_port: int,
    override: models.Endpoint | None,
    node_addrs: list[str],
) -> models.Endpoint | None:
    """Endpoint to publish: the override, else the bind host, else this node's first routable IPv4."""
    if override is not None:
        return override
    if bind_host not in constants.EXEC_WILDCARD_HOSTS:
        return models.Endpoint(bind_host, bound_port)
    for address in node_addrs:
        if "/ip4/" not in address:
            continue
        host = ipaddress.ip_address(
            multiaddr.Multiaddr(address).value_for_protocol("ip4")
        )
        if not host.is_loopback and not host.is_unspecified:
            return models.Endpoint(str(host), bound_port)
    return None


def _is_ip(text: str, version: type) -> bool:
    try:
        version(text)
    except ValueError:
        return False
    return True
