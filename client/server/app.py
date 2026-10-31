"""Starlette application: model listing and chat completions as JSON or server-sent events."""

import asyncio
import contextlib
import json
import secrets
import time

import anyio
from starlette import applications
from starlette import exceptions
from starlette import requests
from starlette import responses
from starlette import routing

from client.generation import models as generation_models
from client.server import errors
from client.network import models as network_models
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


async def chat_completions(request: requests.Request) -> responses.Response:
    """Validate, then generate over an admitted route as one JSON body or an SSE stream."""
    raw = await _read_body(request)
    try:
        body = json.loads(raw)
    except ValueError:
        raise models.ApiError(400, "Request body must be valid JSON.") from None
    params = validation.validate_chat_request(body)
    try:
        stream = request.app.state.service.generate(params)
    except network_models.RouteError as error:
        raise _translate(error) from None
    watcher = asyncio.ensure_future(_wait_for_disconnect(request))
    handed_off = False
    try:
        first = await _next(stream, watcher)
        if first is _DISCONNECTED:
            return responses.Response(status_code=499)
        if params.stream:
            handed_off = True
            return _StreamingResponse(
                stream,
                _events(stream, first),
                media_type="text/event-stream",
                headers=constants.SSE_HEADERS,
            )
        return await _complete(stream, first, watcher)
    finally:
        watcher.cancel()
        if not handed_off:
            await stream.aclose()


async def _complete(stream, item, watcher) -> responses.Response:
    """Collect the whole response into one chat.completion body."""
    pieces = []
    while not isinstance(item, generation_models.Finish):
        pieces.append(item)
        item = await _next(stream, watcher)
        if item is _DISCONNECTED:
            return responses.Response(status_code=499)
    return responses.JSONResponse(
        {
            **_envelope("chat.completion"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(pieces)},
                    "finish_reason": item.reason,
                }
            ],
            "usage": {
                "prompt_tokens": item.prompt_tokens,
                "completion_tokens": item.completion_tokens,
                "total_tokens": item.prompt_tokens + item.completion_tokens,
            },
        }
    )


async def _events(stream, item):
    """SSE frames for a response already known to have started."""
    envelope = _envelope("chat.completion.chunk")
    try:
        yield _sse(envelope, {"role": "assistant", "content": ""}, None)
        while not isinstance(item, generation_models.Finish):
            if item:
                yield _sse(envelope, {"content": item}, None)
            item = await stream.next()
        yield _sse(envelope, {}, item.reason)
        yield "data: [DONE]\n\n"
    except (network_models.RouteError, network_models.GenerationError) as error:
        yield "data: " + json.dumps(errors.error_body(_translate(error))) + "\n\n"
    except models.ApiError as error:
        yield "data: " + json.dumps(errors.error_body(error)) + "\n\n"
    finally:
        await stream.aclose()


class _StreamingResponse(responses.StreamingResponse):
    """Streaming response that also stops the generation when the connection or server ends."""

    def __init__(self, stream, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stream = stream

    async def __call__(self, scope, receive, send) -> None:
        try:
            if _spec_version(scope) < (2, 4):
                await super().__call__(scope, receive, send)
                return
            async with anyio.create_task_group() as group:
                group.start_soon(_cancel_on_disconnect, receive, group.cancel_scope)
                await super().__call__(scope, receive, send)
                group.cancel_scope.cancel()
        finally:
            await self._stream.aclose()


def _spec_version(scope) -> tuple[int, ...]:
    version = scope.get("asgi", {}).get("spec_version", "2.0")
    return tuple(int(part) for part in version.split("."))


async def _cancel_on_disconnect(receive, cancel_scope) -> None:
    """Starlette only notices a disconnect on send from ASGI 2.4, so a stalled body never would."""
    while (await receive())["type"] != "http.disconnect":
        pass
    cancel_scope.cancel()


_DISCONNECTED = object()


async def _next(stream, watcher):
    """The stream's next item, translated errors, or _DISCONNECTED if the client left first."""
    getter = asyncio.ensure_future(stream.next())
    try:
        await asyncio.wait({getter, watcher}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        getter.cancel()
        raise
    if not getter.done():
        getter.cancel()
        return _DISCONNECTED
    try:
        return getter.result()
    except (network_models.RouteError, network_models.GenerationError) as error:
        raise _translate(error) from None


async def _wait_for_disconnect(request: requests.Request) -> None:
    while (await request.receive())["type"] != "http.disconnect":
        pass


def _translate(error: Exception) -> models.ApiError:
    if isinstance(error, network_models.RouteError):
        return models.ApiError(
            503,
            str(error),
            code=constants.ROUTE_UNAVAILABLE_CODE,
            error_type=constants.SERVER_ERROR,
        )
    return models.ApiError(
        502,
        str(error),
        code=constants.GENERATION_FAILED_CODE,
        error_type=constants.SERVER_ERROR,
    )


def _envelope(kind: str) -> dict:
    return {
        "id": constants.COMPLETION_ID_PREFIX
        + secrets.token_hex(constants.COMPLETION_ID_BYTES),
        "object": kind,
        "created": int(time.time()),
        "model": constants.MODEL_ID,
    }


def _sse(envelope: dict, delta: dict, finish_reason: str | None) -> str:
    choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    return "data: " + json.dumps({**envelope, "choices": [choice]}) + "\n\n"


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


@contextlib.asynccontextmanager
async def _lifespan(app: applications.Starlette):
    """Run the network service for the application's lifetime."""
    await app.state.service.start()
    try:
        yield
    finally:
        await app.state.service.stop()


def create_app(service) -> applications.Starlette:
    """Build the application over a network service, with OpenAI-style error handlers."""
    app = applications.Starlette(
        routes=[
            routing.Route("/v1/models", list_models, methods=["GET"]),
            routing.Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        ],
        exception_handlers={
            models.ApiError: errors.api_error_response,
            exceptions.HTTPException: errors.http_error_response,
            Exception: errors.internal_error_response,
        },
        lifespan=_lifespan,
    )
    app.state.service = service
    return app
