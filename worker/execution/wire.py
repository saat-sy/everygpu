"""Execution wire codec: frame layout, payload-size validation, encoding and stream decoding.

Layout: msg_type (1B) | meta_len (4B big-endian) | UTF-8 JSON object | payload.
The payload length is implied by meta: `shape` and `dtype` together give
prod(shape) * itemsize bytes; without them the payload is empty.
"""

import json
import math

import trio

from worker.execution import models
from worker.utils import constants


def encode(frame: models.Frame) -> bytes:
    """Serialise a frame; the payload must match what its metadata declares."""
    if frame.msg_type not in constants.MSG_TYPES:
        raise models.ProtocolError(f"unknown message type {frame.msg_type}")
    meta = json.dumps(frame.meta, separators=(",", ":")).encode()
    if len(meta) > constants.MAX_META_BYTES:
        raise models.ProtocolError("metadata too large")
    if payload_size(frame.meta) != len(frame.payload):
        raise models.ProtocolError("payload length does not match shape and dtype")
    header = constants.FRAME_HEADER.pack(frame.msg_type, len(meta))
    return header + meta + frame.payload


def payload_size(meta: dict) -> int:
    """Payload bytes declared by `shape` and `dtype`; zero when neither is present."""
    if "shape" not in meta and "dtype" not in meta:
        return 0
    shape, dtype = meta.get("shape"), meta.get("dtype")
    if not isinstance(dtype, str) or dtype not in constants.DTYPE_BYTES:
        raise models.ProtocolError("unsupported or missing dtype")
    if not (
        isinstance(shape, list)
        and len(shape) <= constants.MAX_SHAPE_DIMS
        and all(type(dim) is int and dim >= 0 for dim in shape)
    ):
        raise models.ProtocolError("invalid or missing shape")
    size = math.prod(shape) * constants.DTYPE_BYTES[dtype]
    if size > constants.MAX_PAYLOAD_BYTES:
        raise models.ProtocolError("payload too large")
    return size


async def read_frame(stream: trio.abc.ReceiveStream) -> models.Frame | None:
    """Next frame, or None on a clean EOF between frames."""
    header = await _read_exactly(stream, constants.FRAME_HEADER.size, eof_ok=True)
    if header is None:
        return None
    msg_type, meta_len = constants.FRAME_HEADER.unpack(header)
    if msg_type not in constants.MSG_TYPES:
        raise models.ProtocolError(f"unknown message type {msg_type}")
    if meta_len > constants.MAX_META_BYTES:
        raise models.ProtocolError("metadata too large")
    raw = await _read_exactly(stream, meta_len)
    try:
        meta = json.loads(raw)
    except ValueError as error:
        raise models.ProtocolError("metadata is not valid UTF-8 JSON") from error
    if not isinstance(meta, dict):
        raise models.ProtocolError("metadata is not a JSON object")
    payload = await _read_exactly(stream, payload_size(meta))
    return models.Frame(msg_type, meta, payload)


async def _read_exactly(
    stream: trio.abc.ReceiveStream, size: int, eof_ok: bool = False
) -> bytes | None:
    data = bytearray()
    while len(data) < size:
        chunk = await stream.receive_some(
            min(size - len(data), constants.READ_CHUNK_BYTES)
        )
        if not chunk:
            if eof_ok and not data:
                return None
            raise models.ProtocolError("connection closed mid-frame")
        data += chunk
    return bytes(data)
