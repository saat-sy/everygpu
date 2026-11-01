"""Session table: admission, lookup and release of per-session execution state."""

from worker.execution import models
from worker.utils import constants


class SessionTable:
    """Sessions admitted on this worker; each owns isolated state cleared on release."""

    def __init__(self, capacity: int = constants.EXEC_MAX_SESSIONS):
        self.capacity = capacity
        self._sessions: dict[str, models.Session] = {}

    def admit(
        self,
        session_id: str,
        owner: int,
        route: tuple[models.Hop, ...] = (),
        index: int = 0,
    ) -> models.Session:
        if session_id in self._sessions:
            raise models.RequestError("duplicate_session", "session already admitted")
        if len(self._sessions) >= self.capacity:
            raise models.RequestError("at_capacity", "no free session slot")
        session = models.Session(session_id, owner, route, index)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> models.Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise models.RequestError(
                "unknown_session", "session not admitted"
            ) from None

    def release(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.closed = True
            if session.scope is not None:
                session.scope.cancel()
            link = session.state.get(constants.SESSION_LINK_KEY)
            if link is not None:
                link.abort()
            session.state.clear()
        return session is not None

    def release_owned(self, owner: int) -> None:
        for session_id in [
            s.session_id for s in self._sessions.values() if s.owner == owner
        ]:
            self.release(session_id)
