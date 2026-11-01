"""The generation loop: one prefill, then cached single-token decode steps through the route."""

import collections.abc

import trio

from client.generation import text
from client.network import models as network_models
from client.network import session as session_lib
from client.utils import constants
from worker.execution import models as execution_models
from worker.utils import constants as worker_constants


async def run(
    session: session_lib.Session,
    prompt: list[int],
    budget: int,
    temperature: float,
    eos: frozenset[int],
    vocab: int,
    emit: collections.abc.Callable[[str], collections.abc.Awaitable[None]],
    detokenizer: text.Detokenizer,
) -> tuple[str, int]:
    """Generate up to `budget` tokens, emitting each non-EOS delta. Returns the finish reason and sampled count."""
    ids, position = prompt, 0
    for step in range(budget):
        token = await _step(session, ids, position, step, temperature, vocab)
        position += len(ids)
        if token in eos:
            return constants.FINISH_STOP, step + 1
        await emit(detokenizer.add(token))
        ids = [token]
    return constants.FINISH_LENGTH, budget


async def _step(session, ids, position, step, temperature, vocab) -> int:
    """One request/response on the first worker's connection; only the sampled id comes back."""
    channel = session.channels[0]
    frame = execution_models.Frame(
        worker_constants.MSG_FORWARD,
        {
            "session_id": session.session_id,
            "target": "embedding",
            "token_ids": ids,
            "position": position,
            "temperature": temperature,
            "step": step,
        },
    )
    try:
        with trio.fail_after(constants.STEP_TIMEOUT_SECONDS):
            await channel.send(frame)
            response = await channel.receive()
    except trio.TooSlowError:
        raise network_models.GenerationError(
            "the route did not answer in time"
        ) from None
    except session_lib.CONNECTION_ERRORS as error:
        raise network_models.GenerationError(
            f"the connection to the first worker failed: {error}"
        ) from None
    return _token(response, step, vocab)


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
