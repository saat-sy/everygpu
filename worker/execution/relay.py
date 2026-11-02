"""Relays a stage's activations to the next route hop over a persistent per-session connection."""

import logging
import time

import trio

from monitoring.logging import logger as log_utils
from monitoring.telemetry import constants as telemetry_constants
from monitoring.telemetry import edges
from monitoring.telemetry import metrics
from monitoring.telemetry import models as telemetry_models
from monitoring.telemetry import phases
from monitoring.telemetry import timing
from monitoring.telemetry import tracing
from worker.execution import channel as channel_lib
from worker.execution import models
from worker.utils import constants

logger = logging.getLogger(__name__)
_CONNECTION_ERRORS = (
    OSError,
    trio.BrokenResourceError,
    trio.ClosedResourceError,
    models.ProtocolError,
)


async def forward(
    session: models.Session, meta: dict, payload: bytes, incoming_trace: dict | None
) -> dict:
    """Send the activations to the next hop and return its reply (the sampled token)."""
    hop = session.route[session.index + 1]
    step = meta["step"]
    cold = session.state.get(constants.SESSION_LINK_KEY) is None
    attributes = {
        "everygpu.step": step,
        "everygpu.target_peer": hop.peer,
        "everygpu.connection": _connection(cold),
    }
    with tracing.span("worker.relay", attributes, kind="client") as span:
        carrier = tracing.carrier(incoming_trace)
        if carrier:
            meta = {**meta, telemetry_constants.TRACE_KEY: carrier}
        started = time.monotonic()
        response, seconds, request_bytes = await _exchange(session, hop, meta, payload)
        logger.debug(
            "relayed step=%s peer=%s elapsed_ms=%d",
            step,
            hop.peer,
            log_utils.elapsed_ms(started),
        )
        reply = _reply(hop, response, step)
        if carrier:
            below = edges.record(
                telemetry_models.Edge(
                    telemetry_constants.EDGE_RELAY,
                    session.route[session.index].peer,
                    hop.peer,
                    timing.phase(step),
                    _connection(cold),
                    step,
                ),
                span,
                seconds,
                request_bytes,
                response.size,
                response.meta,
                True,
            )
            reply[telemetry_constants.TELEMETRY_KEY] = below.wire()
        return reply


async def _exchange(session, hop, meta: dict, payload: bytes):
    """Dial if needed, send and await the reply; times only the send-to-reply exchange."""
    link = session.state.get(constants.SESSION_LINK_KEY)
    sent = False
    try:
        with trio.fail_after(constants.RELAY_TIMEOUT_SECONDS):
            if link is None:
                link = await _dial(session, hop)
            frame = models.Frame(constants.MSG_FORWARD, meta, payload)
            started = time.monotonic()
            request_bytes = await link.send(frame)
            response = await link.receive()
            seconds = time.monotonic() - started
        sent = True
    except trio.TooSlowError:
        metrics.record("timeouts", phase="relay")
        raise _failed("downstream_timeout", hop, "did not answer in time") from None
    except _CONNECTION_ERRORS as error:
        raise _failed("downstream_unreachable", hop, str(error)) from None
    finally:
        if not sent and link is not None:
            link.abort()
            session.state.pop(constants.SESSION_LINK_KEY, None)
    return response, seconds, request_bytes


def _connection(cold: bool) -> str:
    return (
        telemetry_constants.CONNECTION_COLD
        if cold
        else telemetry_constants.CONNECTION_REUSED
    )


async def _dial(session: models.Session, hop: models.Hop) -> channel_lib.Channel:
    with phases.timed(
        "hop.dial",
        edge=telemetry_constants.EDGE_RELAY,
        connection=telemetry_constants.CONNECTION_COLD,
    ):
        with trio.fail_after(constants.RELAY_DIAL_TIMEOUT_SECONDS):
            stream = await trio.open_tcp_stream(hop.endpoint.host, hop.endpoint.port)
    link = channel_lib.Channel(stream)
    logger.debug("relay connection opened peer=%s endpoint=%s", hop.peer, hop.endpoint)
    if session.closed:
        link.abort()
        raise models.RequestError("unknown_session", "session released")
    session.state[constants.SESSION_LINK_KEY] = link
    return link


def _reply(hop: models.Hop, response: models.Frame | None, step: int) -> dict:
    if response is None or response.msg_type != constants.MSG_RESPONSE:
        raise _failed("downstream_failed", hop, "connection closed")
    reply = response.meta
    if reply.get("status") != "ok":
        raise models.RequestError(
            str(reply.get("code", "downstream_failed")),
            f"worker {hop.peer}: {reply.get('message')}",
        )
    if reply.get("step") != step or type(reply.get("token_id")) is not int:
        raise _failed("downstream_failed", hop, "reply does not match the step")
    return {"token_id": reply["token_id"], "step": step}


def _failed(code: str, hop: models.Hop, detail: str) -> models.RequestError:
    """The downstream failure as a request error, logged as degraded operation."""
    logger.warning(
        "downstream failed peer=%s code=%s detail=%s", hop.peer, code, detail
    )
    return models.RequestError(code, f"worker {hop.peer} at {hop.endpoint}: {detail}")
