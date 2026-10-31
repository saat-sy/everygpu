"""Shared constants and defaults for the worker."""

import struct
import types

DEFAULT_REPO = "saatwiksy/olmoe-partitions"
RECORD_NAMESPACE = "everygpu"
NETWORK_KEY = "/everygpu/network"
PEER_KEY_PREFIX = "/everygpu/peer/"
REFRESH_SECONDS = 60
EXPIRATION_SECONDS = 180
BOOTSTRAP_TIMEOUT_SECONDS = 15
WITHDRAW_TIMEOUT_SECONDS = 15
LOOKUP_CONCURRENCY = 8
PROVIDER_LOOKUP_COUNT = 1000
PROVIDER_RATE_MAX = 1_000_000
LIBP2P_LOGGER = "libp2p"
WEIGHT_FRACTION = 0.8
DEVICE = "cuda"
GIB = 1 << 30
EXEC_BIND_HOST = "0.0.0.0"
EXEC_PORT = 42851
EXEC_MAX_SESSIONS = 16
RELAY_DIAL_TIMEOUT_SECONDS = 5
RELAY_TIMEOUT_SECONDS = 120
MAX_ROUTE_HOPS = 64
SESSION_LINK_KEY = "link"
EXEC_WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::"})
MSG_ADMIT = 1
MSG_FORWARD = 2
MSG_RELEASE = 3
MSG_RESPONSE = 4
MSG_TYPES = frozenset({MSG_ADMIT, MSG_FORWARD, MSG_RELEASE, MSG_RESPONSE})
FRAME_HEADER = struct.Struct(">BI")
MAX_META_BYTES = 1 << 16
MAX_PAYLOAD_BYTES = 64 << 20
MAX_SHAPE_DIMS = 8
READ_CHUNK_BYTES = 1 << 16
SESSION_ID_MAX_CHARS = 128
DTYPE_BYTES = types.MappingProxyType(
    {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int64": 8,
        "int32": 4,
        "int8": 1,
        "uint8": 1,
    }
)
