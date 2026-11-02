"""The generation loop: one prefill, then cached single-token decode steps through the route."""

import collections.abc
import logging
import time

import trio

from client.generation import text
from client.network import models as network_models
from client.network import session as session_lib
from client.utils import constants
from monitoring.logging import logger as log_utils
from monitoring.telemetry import constants as telemetry_constants
from monitoring.telemetry import edges
from monitoring.telemetry import generation
from monitoring.telemetry import metrics
from monitoring.telemetry import timing
from monitoring.telemetry import models as telemetry_models
from monitoring.telemetry import tracing
from worker.execution import models as execution_models
from worker.utils import constants as worker_constants

logger = logging.getLogger(__name__)


async def run(
    session: session_lib.Session,
    prompt: list[int],
    budget: int,
    temperature: float,
    eos: frozenset[int],
    vocab: int,
    emit: collections.abc.Callable[[str], collections.abc.Awaitable[None]],
    detokenizer: text.Detokenizer,
    request_started: float = 0.0,
) -> tuple[str, int]:
    """Generate up to `budget` tokens, emitting each non-EOS delta. Returns the finish reason and sampled count."""
    clock = generation.TokenClock(request_started)
    started = time.monotonic()
    try:
        return await _generate(
            session, prompt, budget, temperature, eos, vocab, emit, detokenizer, clock
        )
    finally:
        clock.finish()
        metrics.record("request.generation_duration", time.monotonic() - started)


async def _generate(
    session, prompt, budget, temperature, eos, vocab, emit, detokenizer, clock
) -> tuple[str, int]:
    started = time.monotonic()
    ids, position = prompt, 0
    for step in range(budget):
        step_started = time.monotonic()
        token = await _step(session, ids, position, step, temperature, vocab)
        clock.arrive()
        logger.debug(
            "step done step=%d tokens_in=%d position=%d elapsed_ms=%d",
            step,
            len(ids),
            position,
            log_utils.elapsed_ms(step_started),
        )
        if step == 0:
            logger.info(
                "first token generated elapsed_ms=%d", log_utils.elapsed_ms(started)
            )
        position += len(ids)
        if token in eos:
            return constants.FINISH_STOP, step + 1
        await emit(detokenizer.add(token))
        ids = [token]
    return constants.FINISH_LENGTH, budget


async def _step(session, ids, position, step, temperature, vocab) -> int:
    """One request/response on the first worker's connection; only the sampled id comes back."""
    phase = timing.phase(step)
    attributes = {
        "everygpu.step": step,
        "everygpu.phase": phase,
        "everygpu.tokens_in": len(ids),
        "everygpu.position": position,
    }
    with tracing.span("client.step", attributes):
        with tracing.span("client.exchange", kind="client") as span:
            return await _exchange(
                session, span, ids, position, step, temperature, vocab, phase
            )


async def _exchange(session, span, ids, position, step, temperature, vocab, phase):
    channel = session.channels[0]
    meta = {
        "session_id": session.session_id,
        "target": "embedding",
        "token_ids": ids,
        "position": position,
        "temperature": temperature,
        "step": step,
    }
    carrier = tracing.carrier()
    if carrier:
        meta[telemetry_constants.TRACE_KEY] = carrier
    frame = execution_models.Frame(worker_constants.MSG_FORWARD, meta)
    try:
        with trio.fail_after(constants.STEP_TIMEOUT_SECONDS):
            started = time.monotonic()
            request_bytes = await channel.send(frame)
            response = await channel.receive()
            seconds = time.monotonic() - started
    except trio.TooSlowError:
        metrics.record("timeouts", phase="generation")
        raise network_models.GenerationError(
            "the route did not answer in time"
        ) from None
    except session_lib.CONNECTION_ERRORS as error:
        raise network_models.GenerationError(
            f"the connection to the first worker failed: {error}"
        ) from None
    token = _token(response, step, vocab)
    below = edges.record(
        telemetry_models.Edge(
            telemetry_constants.EDGE_CLIENT,
            telemetry_constants.CLIENT_PEER,
            session.route.hops[0].peer,
            phase,
            telemetry_constants.CONNECTION_REUSED,
            step,
        ),
        span,
        seconds,
        request_bytes,
        response.size,
        response.meta,
        carrier is not None,
    )
    metrics.record("route.step_exchange", seconds, phase=phase)
    if carrier and not below.incomplete:
        metrics.record(
            "route.step_communication", below.communication_seconds, phase=phase
        )
        span.set_attribute(
            "everygpu.route_communication_seconds", below.communication_seconds
        )
    return token


def _token(response: execution_models.Frame | None, step: int, vocab: int) -> int:
    if response is None or response.msg_type != worker_constants.MSG_RESPONSE:
        raise network_models.GenerationError("the first worker closed the connection")
    meta = response.meta
    if meta.get("status") != "ok":
        raise network_models.GenerationError(
            f"{meta.get('message')} ({meta.get('code')})"
        )
    token = meta.get("token_id")
    if meta.get("step") != step or type(token) is not int or not 0 <= token < vocab:
        raise network_models.GenerationError("the route returned an invalid token")
    return token
