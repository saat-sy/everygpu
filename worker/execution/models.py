"""Execution data structures: wire frames, endpoints, sessions and request errors."""

import dataclasses


class ProtocolError(ValueError):
    """A frame violates the wire format; the connection cannot be resynchronised."""


class RequestError(Exception):
    """A well-formed request the worker refuses; reported to the sender as an error response."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclasses.dataclass(frozen=True)
class Frame:
    msg_type: int
    meta: dict
    payload: bytes = b""


@dataclasses.dataclass(frozen=True)
class Endpoint:
    host: str
    port: int

    def __str__(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{host}:{self.port}"


@dataclasses.dataclass
class Session:
    session_id: str
    owner: int
    state: dict = dataclasses.field(default_factory=dict)
