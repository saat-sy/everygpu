"""Constants for the shared logging configuration."""

import logging
import types

LEVEL_ENV = "EVERYGPU_LOG_LEVEL"
DEFAULT_LEVEL = "INFO"
LEVELS = types.MappingProxyType(
    {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
)
NAMESPACES = ("client", "worker", "model_partitioner")
LIBP2P_LOGGER = "libp2p"
THIRD_PARTY_LEVEL = logging.WARNING
LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
PROGRESS_STEPS = 10
