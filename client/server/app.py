"""Starlette application: model listing and chat completions as JSON or server-sent events."""

import asyncio
import contextlib
import json
import logging
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
from monitoring.logging import logger as log_utils

logger = logging.getLogger(__name__)


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
    """Serve one chat request, logging how it ended exactly once."""
    exchange = models.Exchange(
        secrets.token_hex(constants.COMPLETION_ID_BYTES), time.monotonic()
    )
    with log_utils.bind(request_id=exchange.request_id):
        try:
            return await _chat(request, exchange)
        except models.ApiError as error:
            exchange.status = error.status
            if error.status < 500:
                exchange.outcome = constants.OUTCOME_REJECTED
            else:
                exchange.outcome, exchange.error = (
                    constants.OUTCOME_FAILED,
                    error.message,
                )
                exchange.exc = error.__cause__
            raise
        except requests.ClientDisconnect:
            return _disconnected(exchange)
        except asyncio.CancelledError:
            exchange.outcome = constants.OUTCOME_CANCELLED
            raise
        except Exception as error:
            exchange.status, exchange.outcome = 500, constants.OUTCOME_FAILED
            exchange.exc = error
            return await errors.internal_error_response(request, error)
        finally:
            if not exchange.streaming:
                _log_end(exchange)


async def _chat(request: requests.Request, exchange: models.Exchange):
    """Validate, then generate over an admitted route as one JSON body or an SSE stream."""
    raw = await _read_body(request)
    try:
        body = json.loads(raw)
    except ValueError:
        raise models.ApiError(400, "Request body must be valid JSON.") from None
    params = validation.validate_chat_request(body)
    logger.info(
        "chat request accepted stream=%s messages=%d max_tokens=%d",
        params.stream,
        len(params.messages),
        params.max_tokens,
    )
    try:
        stream = request.app.state.service.generate(params)
    except network_models.RouteError as error:
        raise _translate(error) from None
    watcher = asyncio.ensure_future(_wait_for_disconnect(request))
    handed_off = False
    try:
        first = await _next(stream, watcher)
        if first is _DISCONNECTED:
            return _disconnected(exchange)
        if params.stream:
            handed_off = exchange.streaming = True
            exchange.outcome = constants.OUTCOME_DISCONNECTED
            return _StreamingResponse(
                stream,
                exchange,
                _events(stream, first, exchange),
                media_type="text/event-stream",
                headers=constants.SSE_HEADERS,
            )
        return await _complete(stream, first, watcher, exchange)
    finally:
        watcher.cancel()
        if not handed_off:
            await stream.aclose()


def _disconnected(exchange: models.Exchange) -> responses.Response:
    exchange.status = constants.STATUS_CLIENT_CLOSED
    exchange.outcome = constants.OUTCOME_DISCONNECTED
    return responses.Response(status_code=exchange.status)


def _log_end(exchange: models.Exchange) -> None:
    level = (
        logging.ERROR if exchange.outcome == constants.OUTCOME_FAILED else logging.INFO
    )
    tokens = exchange.completion_tokens
    logger.log(
        level,
        "chat request %s status=%d chunks=%d%s elapsed_ms=%d%s",
        exchange.outcome,
        exchange.status,
        exchange.chunks,
        "" if tokens is None else f" completion_tokens={tokens}",
        log_utils.elapsed_ms(exchange.started),
        f" error={exchange.error}" if exchange.error else "",
        exc_info=exchange.exc,
    )


async def _complete(stream, item, watcher, exchange) -> responses.Response:
    """Collect the whole response into one chat.completion body."""
    pieces = []
    while not isinstance(item, generation_models.Finish):
        pieces.append(item)
        exchange.chunks = len(pieces)
        item = await _next(stream, watcher)
        if item is _DISCONNECTED:
            return _disconnected(exchange)
    exchange.completion_tokens = item.completion_tokens
    return responses.JSONResponse(
        {
            **_envelope("chat.completion", exchange.request_id),
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


async def _events(stream, item, exchange):
    """SSE frames for a response already known to have started."""
    envelope = _envelope("chat.completion.chunk", exchange.request_id)
    try:
        yield _sse(envelope, {"role": "assistant", "content": ""}, None)
        while not isinstance(item, generation_models.Finish):
            exchange.chunks += 1
            if item:
                yield _sse(envelope, {"content": item}, None)
            item = await stream.next()
        exchange.completion_tokens = item.completion_tokens
        yield _sse(envelope, {}, item.reason)
        yield "data: [DONE]\n\n"
        exchange.outcome = constants.OUTCOME_COMPLETED
    except (network_models.RouteError, network_models.GenerationError) as error:
        exchange.outcome, exchange.error = constants.OUTCOME_FAILED, str(error)
        exchange.exc = error.__cause__
        yield "data: " + json.dumps(errors.error_body(_translate(error))) + "\n\n"
    except models.ApiError as error:
        exchange.outcome = constants.OUTCOME_REJECTED
        yield "data: " + json.dumps(errors.error_body(error)) + "\n\n"
    except Exception as error:
        exchange.outcome, exchange.exc = constants.OUTCOME_FAILED, error
        yield "data: " + json.dumps(errors.error_body(errors.internal_error())) + "\n\n"
    finally:
        await stream.aclose()


class _StreamingResponse(responses.StreamingResponse):
    """Streaming response that also stops the generation when the connection or server ends."""

    def __init__(self, stream, exchange, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stream = stream
        self._exchange = exchange

    async def __call__(self, scope, receive, send) -> None:
        """Log how the stream ended once it has, not when its headers went out."""
        with log_utils.bind(request_id=self._exchange.request_id):
            try:
                await self._send_body(scope, receive, send)
            except asyncio.CancelledError:
                self._exchange.outcome = constants.OUTCOME_CANCELLED
                raise
            except Exception:
                self._exchange.outcome = constants.OUTCOME_FAILED
                raise
            finally:
                _log_end(self._exchange)
                await self.body_iterator.aclose()
                await self._stream.aclose()

    async def _send_body(self, scope, receive, send) -> None:
        if _spec_version(scope) < (2, 4):
            await super().__call__(scope, receive, send)
            return
        try:
            async with anyio.create_task_group() as group:
                group.start_soon(_cancel_on_disconnect, receive, group.cancel_scope)
                await super().__call__(scope, receive, send)
                group.cancel_scope.cancel()
        except* requests.ClientDisconnect:
            self._exchange.outcome = constants.OUTCOME_DISCONNECTED


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
        raise _translate(error) from error.__cause__


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


def _envelope(kind: str, request_id: str) -> dict:
    return {
        "id": constants.COMPLETION_ID_PREFIX + request_id,
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
        logger.info("network service stopped")


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
