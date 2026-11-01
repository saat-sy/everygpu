"""Relays a stage's activations to the next route hop over a persistent per-session connection."""

import trio

from worker.execution import channel as channel_lib
from worker.execution import models
from worker.utils import constants

_CONNECTION_ERRORS = (
    OSError,
    trio.BrokenResourceError,
    trio.ClosedResourceError,
    models.ProtocolError,
)


async def forward(session: models.Session, meta: dict, payload: bytes) -> dict:
    """Send the activations to the next hop and return its reply (the sampled token)."""
    hop = session.route[session.index + 1]
    link = session.state.get(constants.SESSION_LINK_KEY)
    sent = False
    try:
        with trio.fail_after(constants.RELAY_TIMEOUT_SECONDS):
            if link is None:
                link = await _dial(session, hop)
            await link.send(models.Frame(constants.MSG_FORWARD, meta, payload))
            response = await link.receive()
        sent = True
    except trio.TooSlowError:
        raise _failed("downstream_timeout", hop, "did not answer in time") from None
    except _CONNECTION_ERRORS as error:
        raise _failed("downstream_unreachable", hop, str(error)) from None
    finally:
        if not sent and link is not None:
            link.abort()
            session.state.pop(constants.SESSION_LINK_KEY, None)
    return _reply(hop, response, meta["step"])


async def _dial(session: models.Session, hop: models.Hop) -> channel_lib.Channel:
    with trio.fail_after(constants.RELAY_DIAL_TIMEOUT_SECONDS):
        stream = await trio.open_tcp_stream(hop.endpoint.host, hop.endpoint.port)
    link = channel_lib.Channel(stream)
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
    return models.RequestError(code, f"worker {hop.peer} at {hop.endpoint}: {detail}")
