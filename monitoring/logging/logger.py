"""Shared logging: startup configuration, event format, bound context fields, timing and progress."""

import contextlib
import contextvars
import logging
import math
import os
import time
import types

from monitoring.logging import constants

_FIELDS: contextvars.ContextVar[types.MappingProxyType] = contextvars.ContextVar(
    "everygpu_log_fields", default=types.MappingProxyType({})
)


class Formatter(logging.Formatter):
    """Appends the context fields to the message line, ahead of any traceback."""

    converter = time.gmtime

    def formatMessage(self, record: logging.LogRecord) -> str:
        fields = "".join(f" {key}={value}" for key, value in current().items())
        return super().formatMessage(record) + fields


def configure() -> None:
    """Install the one stderr handler and set levels; raises ValueError on a bad level."""
    name = os.environ.get(constants.LEVEL_ENV, constants.DEFAULT_LEVEL)
    level = constants.LEVELS.get(name.upper())
    if level is None:
        raise ValueError(
            f"{constants.LEVEL_ENV} must be one of {', '.join(constants.LEVELS)}, "
            f"got {name!r}"
        )
    handler = logging.StreamHandler()
    handler.setFormatter(Formatter(constants.LINE_FORMAT, constants.TIME_FORMAT))
    third_party = max(level, constants.THIRD_PARTY_LEVEL)
    logging.basicConfig(level=third_party, handlers=[handler], force=True)
    for namespace in constants.NAMESPACES:
        logging.getLogger(namespace).setLevel(level)
    # libp2p detaches itself on import.
    libp2p = logging.getLogger(constants.LIBP2P_LOGGER)
    libp2p.handlers.clear()
    libp2p.propagate = True
    libp2p.setLevel(level if level == logging.DEBUG else third_party)
    for name in list(logging.root.manager.loggerDict):
        if name.startswith(constants.LIBP2P_LOGGER + "."):
            logging.getLogger(name).setLevel(logging.NOTSET)


def current() -> types.MappingProxyType:
    """The fields bound in the current context."""
    return _FIELDS.get()


@contextlib.contextmanager
def bind(**fields):
    """Add fields to every event logged inside the block, including its child tasks."""
    token = _FIELDS.set(types.MappingProxyType({**_FIELDS.get(), **fields}))
    try:
        yield
    finally:
        _FIELDS.reset(token)


def elapsed_ms(started: float) -> int:
    """Whole milliseconds since a time.monotonic() reading."""
    return int((time.monotonic() - started) * 1000)


def progress_due(done: int, total: int) -> bool:
    """Whether to log after `done` of `total` items, counting from 1."""
    return done == total or done % math.ceil(total / constants.PROGRESS_STEPS) == 0
