"""Route admission: dial each hop in order, admit the session, roll back on any failure."""

import contextlib
import logging
import secrets
import time

import trio

from client.network import models
from client.network import session as session_lib
from client.utils import constants
from monitoring.logging import logger as log_utils
from monitoring.telemetry import constants as telemetry_constants
from monitoring.telemetry import edges
from monitoring.telemetry import metrics
from monitoring.telemetry import models as telemetry_models
from monitoring.telemetry import phases
from monitoring.telemetry import tracing
from worker.execution import channel as channel_lib
from worker.execution import identity
from worker.execution import models as execution_models
from worker.utils import constants as worker_constants

logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def open_session(route: models.Route):
    """Admit a new session on every hop; release it all on exit, error or cancellation."""
    session = session_lib.Session(secrets.token_hex(constants.SESSION_ID_BYTES), route)
    started = time.monotonic()
    admitted = False
    with log_utils.bind(session_id=session.session_id):
        try:
            with tracing.span("client.admission", {"everygpu.hops": len(route.hops)}):
                for index, hop in enumerate(route.hops):
                    await _admit(session, hop, index)
                admitted = True
                metrics.record("sessions.active", 1, side="client")
                metrics.record("setup.admission", time.monotonic() - started)
                logger.info(
                    "route admitted hops=%d elapsed_ms=%d",
                    len(route.hops),
                    log_utils.elapsed_ms(started),
                )
            yield session
        finally:
            await session.release()
            if admitted:
                metrics.record("sessions.active", -1, side="client")


async def _admit(session: session_lib.Session, hop: models.Hop, index: int) -> None:
    attributes = {"everygpu.target_peer": hop.peer, "everygpu.index": index}
    with tracing.span("client.admit", attributes, kind="client") as span:
        route = session.route
        meta = {
            "session_id": session.session_id,
            "expect": identity.build(route.repo, route.revision, hop.peer, hop.stage),
            "route": session.wire_route,
            "index": index,
        }
        started = time.monotonic()
        channel = await _dial(session, hop)
        carrier = tracing.carrier()
        if carrier:
            meta[telemetry_constants.TRACE_KEY] = carrier
        request = execution_models.Frame(worker_constants.MSG_ADMIT, meta)
        try:
            with trio.fail_after(constants.ADMIT_TIMEOUT_SECONDS):
                exchange_started = time.monotonic()
                request_bytes = await channel.send(request)
                response = await channel.receive()
                seconds = time.monotonic() - exchange_started
        except trio.TooSlowError:
            metrics.record("timeouts", phase="admission")
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
                response
                and f"{response.meta.get('code')}: {response.meta.get('message')}"
            )
            raise models.RouteError(f"worker {hop.peer} refused admission ({detail})")
        session.channels.append(channel)
        edges.record(
            telemetry_models.Edge(
                telemetry_constants.EDGE_CLIENT,
                telemetry_constants.CLIENT_PEER,
                hop.peer,
                telemetry_constants.PHASE_ADMIT,
                telemetry_constants.CONNECTION_COLD,
                index,
            ),
            span,
            seconds,
            request_bytes,
            response.size,
            response.meta,
            carrier is not None,
        )
        logger.debug(
            "worker admitted peer=%s stage=%d index=%d elapsed_ms=%d",
            hop.peer,
            hop.stage.first,
            index,
            log_utils.elapsed_ms(started),
        )


async def _dial(session: session_lib.Session, hop: models.Hop):
    """Open the owner connection to a hop; its time is the cold-connection cost."""
    try:
        with phases.timed(
            "hop.dial",
            traced=True,
            edge=telemetry_constants.EDGE_CLIENT,
            connection=telemetry_constants.CONNECTION_COLD,
        ):
            with trio.fail_after(constants.DIAL_TIMEOUT_SECONDS):
                stream = await trio.open_tcp_stream(
                    hop.endpoint.host, hop.endpoint.port
                )
    except trio.TooSlowError:
        metrics.record("timeouts", phase="admission")
        raise models.RouteError(
            f"worker {hop.peer} at {hop.endpoint} did not answer in time"
        ) from None
    except session_lib.CONNECTION_ERRORS as error:
        raise models.RouteError(
            f"worker {hop.peer} at {hop.endpoint} is unreachable: {error}"
        ) from None
    channel = channel_lib.Channel(stream)
    session.opened.append(channel)
    return channel
