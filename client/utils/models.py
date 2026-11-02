"""Data structures for client request failures."""

import dataclasses

from client.utils import constants


class ApiError(Exception):
    """A request failure carrying its OpenAI-style error fields."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        param: str | None = None,
        code: str | None = None,
        error_type: str = constants.INVALID_REQUEST,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code
        self.error_type = error_type
        self.headers = headers


@dataclasses.dataclass(frozen=True)
class ChatRequest:
    """A validated chat completion request."""

    messages: list[dict]
    stream: bool
    max_tokens: int
    temperature: float
    exchange: "Exchange | None" = None


@dataclasses.dataclass
class Exchange:
    """One chat request's identity and outcome so far, logged once when it ends."""

    request_id: str
    started: float
    status: int = 200
    outcome: str = constants.OUTCOME_COMPLETED
    streaming: bool = False
    chunks: int = 0
    completion_tokens: int | None = None
    error: str = ""
    exc: BaseException | None = None
    span: object = None
    context: object = None
    phase: str = constants.PHASE_VALIDATION
    content_chunks: int = 0
    content_sent: bool = False
