"""Route admission: dial each hop in order, admit the session, roll back on any failure."""

import contextlib
import secrets

import trio

from client.network import models
from client.network import session as session_lib
from client.utils import constants
from worker.execution import channel as channel_lib
from worker.execution import identity
from worker.execution import models as execution_models
from worker.utils import constants as worker_constants


@contextlib.asynccontextmanager
async def open_session(route: models.Route):
    """Admit a new session on every hop; release it all on exit, error or cancellation."""
    session = session_lib.Session(secrets.token_hex(constants.SESSION_ID_BYTES), route)
    try:
        for hop in route.hops:
            await _admit(session, hop)
        yield session
    finally:
        await session.release()


async def _admit(session: session_lib.Session, hop: models.Hop) -> None:
    route = session.route
    request = execution_models.Frame(
        worker_constants.MSG_ADMIT,
        {
            "session_id": session.session_id,
            "expect": identity.build(route.repo, route.revision, hop.peer, hop.stage),
        },
    )
    try:
        with trio.fail_after(constants.DIAL_TIMEOUT_SECONDS):
            stream = await trio.open_tcp_stream(hop.endpoint.host, hop.endpoint.port)
        channel = channel_lib.Channel(stream)
        session.opened.append(channel)
        with trio.fail_after(constants.ADMIT_TIMEOUT_SECONDS):
            await channel.send(request)
            response = await channel.receive()
    except trio.TooSlowError:
        raise models.RouteError(
            f"worker {hop.peer} at {hop.endpoint} did not answer in time"
        ) from None
    except session_lib.CONNECTION_ERRORS as error:
        raise models.RouteError(
            f"worker {hop.peer} at {hop.endpoint} is unreachable: {error}"
        ) from None
    if (
        response is None
        or response.msg_type != worker_constants.MSG_RESPONSE
        or response.meta.get("status") != "ok"
    ):
        detail = (
            response and f"{response.meta.get('code')}: {response.meta.get('message')}"
        )
        raise models.RouteError(f"worker {hop.peer} refused admission ({detail})")
    session.channels.append(channel)
