"""Data structures for client request failures."""

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
