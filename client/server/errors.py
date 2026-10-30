"""Renders request failures as OpenAI-style error responses."""

from starlette import exceptions
from starlette import requests
from starlette import responses

from client.utils import constants
from client.utils import models


async def api_error_response(
    request: requests.Request, exc: models.ApiError
) -> responses.JSONResponse:
    """Render an ApiError as an OpenAI error object."""
    body = {
        "error": {
            "message": exc.message,
            "type": exc.error_type,
            "param": exc.param,
            "code": exc.code,
        }
    }
    return responses.JSONResponse(body, status_code=exc.status, headers=exc.headers)


async def http_error_response(
    request: requests.Request, exc: exceptions.HTTPException
) -> responses.JSONResponse:
    """Render routing errors such as 404 and 405 in the same shape."""
    error = models.ApiError(exc.status_code, exc.detail, headers=exc.headers)
    return await api_error_response(request, error)


async def internal_error_response(
    request: requests.Request, exc: Exception
) -> responses.JSONResponse:
    """Render unexpected failures without exposing their details."""
    error = models.ApiError(
        500, constants.INTERNAL_ERROR_MESSAGE, error_type=constants.SERVER_ERROR
    )
    return await api_error_response(request, error)
