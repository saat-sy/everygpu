"""Chat completion request validation."""

from client.utils import constants
from client.utils import models


def validate_chat_request(body: object) -> None:
    """Raise ApiError (400 or 404) unless body is a supported text chat request."""
    if not isinstance(body, dict):
        raise models.ApiError(400, "Request body must be a JSON object.")
    for field in body:
        if field not in constants.REQUEST_FIELDS:
            raise models.ApiError(400, f"Unsupported parameter: {field}.", param=field)
    if not isinstance(body.get("model"), str):
        raise models.ApiError(400, "model must be a string.", param="model")
    if not isinstance(body.get("stream", False), bool):
        raise models.ApiError(400, "stream must be a boolean.", param="stream")
    _validate_messages(body.get("messages"))
    if body["model"] != constants.MODEL_ID:
        raise models.ApiError(
            404,
            f"The model {body['model']!r} does not exist.",
            param="model",
            code="model_not_found",
        )


def _validate_messages(messages: object) -> None:
    if not isinstance(messages, list) or not messages:
        raise models.ApiError(
            400, "messages must be a nonempty list.", param="messages"
        )
    for i, message in enumerate(messages):
        param = f"messages[{i}]"
        if not isinstance(message, dict):
            raise models.ApiError(400, "Each message must be an object.", param=param)
        for field in message:
            if field not in constants.MESSAGE_FIELDS:
                raise models.ApiError(
                    400, f"Unsupported parameter: {field}.", param=f"{param}.{field}"
                )
        role = message.get("role")
        if not isinstance(role, str) or role not in constants.ROLES:
            raise models.ApiError(
                400,
                "role must be system, user or assistant.",
                param=f"{param}.role",
            )
        if not isinstance(message.get("content"), str):
            raise models.ApiError(
                400, "content must be a text string.", param=f"{param}.content"
            )
