"""Starlette application: routes for model listing and chat completions."""

import json

from starlette import applications
from starlette import exceptions
from starlette import requests
from starlette import responses
from starlette import routing

from client.server import errors
from client.server import validation
from client.utils import constants
from client.utils import models


async def list_models(request: requests.Request) -> responses.JSONResponse:
    """Return the intended model; it does not imply available workers."""
    model = {
        "id": constants.MODEL_ID,
        "object": "model",
        "created": 0,
        "owned_by": constants.MODEL_OWNER,
    }
    return responses.JSONResponse({"object": "list", "data": [model]})


async def chat_completions(request: requests.Request) -> responses.JSONResponse:
    """Validate the request, then report that generation is unavailable."""
    raw = await _read_body(request)
    try:
        body = json.loads(raw)
    except ValueError:
        raise models.ApiError(400, "Request body must be valid JSON.") from None
    validation.validate_chat_request(body)
    raise models.ApiError(
        501,
        constants.NOT_IMPLEMENTED_MESSAGE,
        code=constants.NOT_IMPLEMENTED_CODE,
        error_type=constants.SERVER_ERROR,
    )


async def _read_body(request: requests.Request) -> bytes:
    """Read the body, rejecting it once it exceeds the size limit."""
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > constants.MAX_BODY_BYTES:
            raise models.ApiError(
                413, f"Request body exceeds {constants.MAX_BODY_BYTES} bytes."
            )
    return bytes(body)


def create_app() -> applications.Starlette:
    """Build the application with OpenAI-style error handlers."""
    return applications.Starlette(
        routes=[
            routing.Route("/v1/models", list_models, methods=["GET"]),
            routing.Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        ],
        exception_handlers={
            models.ApiError: errors.api_error_response,
            exceptions.HTTPException: errors.http_error_response,
            Exception: errors.internal_error_response,
        },
    )
