"""Trio TCP execution server: request dispatch for ADMIT, FORWARD and RELEASE."""

import dataclasses
import itertools
import logging
import collections.abc

import trio

from monitoring.logging import logger as log_utils
from worker.execution import channel as channel_lib
from worker.execution import models
from worker.execution import route as route_lib
from worker.execution import sessions as sessions_lib
from worker.utils import constants

logger = logging.getLogger(__name__)

ForwardHandler = collections.abc.Callable[
    [models.Session, dict, bytes],
    collections.abc.Awaitable[tuple[dict, bytes]],
]


async def unavailable_forward(session: models.Session, meta: dict, payload: bytes):
    raise models.RequestError("unavailable", "forward execution is not implemented")


class ExecutionServer:
    """Serves framed requests; a session lives until released or its admitting connection closes."""

    def __init__(
        self,
        sessions: sessions_lib.SessionTable,
        forward: ForwardHandler = unavailable_forward,
        identity: collections.abc.Callable[[], dict] | None = None,
    ):
        self.sessions = sessions
        self.forward = forward
        self.identity = identity
        self._owners = itertools.count()

    async def serve(
        self, host: str, port: int, *, task_status=trio.TASK_STATUS_IGNORED
    ) -> None:
        """Listen until cancelled; reports the bound port once listening."""
        listeners = await trio.open_tcp_listeners(port, host=host)
        task_status.started(listeners[0].socket.getsockname()[1])
        await trio.serve_listeners(self._serve_connection, listeners)

    async def _serve_connection(self, stream: trio.SocketStream) -> None:
        channel = channel_lib.Channel(stream)
        owner = next(self._owners)
        logger.debug("connection opened owner=%d", owner)
        try:
            while True:
                try:
                    frame = await channel.receive()
                except models.ProtocolError as error:
                    logger.warning("malformed frame owner=%d error=%s", owner, error)
                    await channel.send(_error("malformed_frame", str(error)))
                    return
                if frame is None:
                    return
                await channel.send(await self._respond(frame, owner))
        except (trio.BrokenResourceError, trio.ClosedResourceError):
            pass
        except Exception:
            logger.exception("execution connection failed owner=%d", owner)
        finally:
            released = self.sessions.release_owned(owner)
            logger.debug(
                "connection closed owner=%d sessions_released=%d", owner, released
            )
            await stream.aclose()

    async def _respond(self, frame: models.Frame, owner: int) -> models.Frame:
        try:
            meta, payload = await self._dispatch(frame, owner)
        except models.RequestError as error:
            return _error(error.code, error.message)
        except Exception:
            return _error("internal", "request failed")
        return models.Frame(constants.MSG_RESPONSE, {"status": "ok", **meta}, payload)

    async def _dispatch(self, frame: models.Frame, owner: int) -> tuple[dict, bytes]:
        if frame.msg_type == constants.MSG_RESPONSE:
            raise models.RequestError("bad_request", "RESPONSE is not a request")
        session_id = _session_id(frame.meta)
        with log_utils.bind(session_id=session_id):
            try:
                return await self._handle(frame, session_id, owner)
            except models.RequestError as error:
                admitting = frame.msg_type == constants.MSG_ADMIT
                logger.log(
                    logging.WARNING if admitting else logging.DEBUG,
                    "request refused type=%d code=%s",
                    frame.msg_type,
                    error.code,
                )
                raise
            except Exception:
                logger.exception("execution request failed type=%d", frame.msg_type)
                raise

    async def _handle(
        self, frame: models.Frame, session_id: str, owner: int
    ) -> tuple[dict, bytes]:
        if frame.msg_type == constants.MSG_ADMIT:
            hops, index = route_lib.parse(frame.meta)
            self._check_identity(frame.meta, hops[index])
            self.sessions.admit(session_id, owner, hops, index)
            stage = hops[index].stage
            logger.debug(
                "session admitted layers=%d-%d index=%d active=%d",
                stage.first,
                stage.stop,
                index,
                len(self.sessions),
            )
            return {}, b""
        if frame.msg_type == constants.MSG_RELEASE:
            released = self.sessions.release(session_id)
            logger.debug(
                "session released found=%s active=%d", released, len(self.sessions)
            )
            return {"released": released}, b""
        session = self.sessions.get(session_id)
        return await self.forward(session, frame.meta, frame.payload)

    def _check_identity(self, meta: dict, own: models.Hop) -> None:
        """Refuse admission unless the sender expects exactly this stage, at its own route slot."""
        if self.identity is None:
            return
        current = self.identity()
        if current["stage"] is None:
            raise models.RequestError("unavailable", "no executable stage is resident")
        if meta.get("expect") != current:
            raise models.RequestError(
                "stale_stage", "pin, peer or stage differs from the expected one"
            )
        if (current["peer"], current["stage"]) != (
            own.peer,
            dataclasses.asdict(own.stage),
        ):
            raise models.RequestError(
                "bad_route", "the route slot is not this worker's peer and stage"
            )


def _session_id(meta: dict) -> str:
    session_id = meta.get("session_id")
    if (
        not isinstance(session_id, str)
        or not 0 < len(session_id) <= constants.SESSION_ID_MAX_CHARS
    ):
        raise models.RequestError("bad_request", "invalid session_id")
    return session_id


def _error(code: str, message: str) -> models.Frame:
    meta = {"status": "error", "code": code, "message": message}
    return models.Frame(constants.MSG_RESPONSE, meta)
