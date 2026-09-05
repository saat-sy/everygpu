import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter_ns
from typing import NoReturn
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from opentelemetry.trace import SpanKind
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import constants
from download import download_laptop
from telemetry import (
    RequestTelemetry,
    RuntimeMeasurement,
    elapsed_ms,
    otel,
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    otel.configure_telemetry()
    try:
        yield
    finally:
        otel.shutdown_telemetry()


app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.router.redirect_slashes = False

runtimes: dict[int, WebSocket] = {}
runtime_inboxes: dict[int, asyncio.Queue[str | bytes]] = {}
ready_runtimes: set[int] = set()
download_started = False
tokenizer: PreTrainedTokenizerBase | None = None
pipeline_lock = asyncio.Lock()


@app.get("/health")
async def health():
    return {
        "connected_runtimes": len(runtimes),
        "ready_runtimes": len(ready_runtimes),
        "ready_stages": sorted(ready_runtimes),
        "required_runtimes": constants.REQUIRED_RUNTIMES,
        "download_started": download_started,
        "pipeline_ready": (
            len(ready_runtimes) == constants.REQUIRED_RUNTIMES
            and tokenizer is not None
        ),
    }


@app.post("/v1/completions")
async def completions(request: dict):
    with otel.tracer.start_as_current_span(
        constants.PIPELINE_TRACE_NAME,
        kind=SpanKind.SERVER,
    ):
        return await complete_request(request)


async def complete_request(request: dict):
    request_started_ns = perf_counter_ns()
    request_id = uuid4().hex
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise HTTPException(400, "prompt must be a non-empty string")
    model = request.get("model", constants.MODEL_NAME)
    if not isinstance(model, str) or not model:
        raise HTTPException(400, "model must be a non-empty string")

    request_bytes = len(json.dumps(request, separators=(",", ":")).encode("utf-8"))
    request_telemetry = RequestTelemetry.start(
        request_id=request_id,
        model=model,
        request_started_ns=request_started_ns,
        request_bytes=request_bytes,
    )
    queue_started_ns = perf_counter_ns()

    async with pipeline_lock:
        queue_wait_ms = elapsed_ms(queue_started_ns)
        active_tokenizer = tokenizer
        runtime_zero = runtimes.get(0)
        runtime_one = runtimes.get(1)
        if (
            len(ready_runtimes) != constants.REQUIRED_RUNTIMES
            or active_tokenizer is None
            or runtime_zero is None
            or runtime_one is None
        ):
            raise HTTPException(503, "pipeline is not ready")

        tokenize_started_ns = perf_counter_ns()
        input_ids = list(active_tokenizer.encode(prompt, add_special_tokens=False))
        tokenize_ms = elapsed_ms(tokenize_started_ns)
        prompt_tokens = len(input_ids)
        generated_tokens = []

        for step_id in range(constants.MAX_NEW_TOKENS):
            phase = "prefill" if step_id == 0 else "decode"
            command = json.dumps(
                {
                    "type": "forward_tokens",
                    "request_id": request_id,
                    "step_id": step_id,
                    "phase": phase,
                    "input_ids": input_ids,
                },
                separators=(",", ":"),
            )

            exchange_started_ns = perf_counter_ns()
            send_started_ns = perf_counter_ns()
            await runtime_zero.send_text(command)
            send_ms = elapsed_ms(send_started_ns)

            runtime_zero_response = parse_runtime_response(
                await receive_runtime_message(0),
                expected_type="hidden_states",
                request_id=request_id,
                step_id=step_id,
                stage=0,
            )
            hidden_states = await receive_runtime_message(0)
            if not isinstance(hidden_states, bytes):
                raise_runtime_error(hidden_states)
            stage_zero_exchange_ms = elapsed_ms(exchange_started_ns)
            runtime_zero_metrics = parse_runtime_measurement(
                runtime_zero_response["runtime"]
            )
            request_telemetry.record_runtime(
                runtime_zero_metrics,
                phase,
                request_message_type="token_ids",
                request_latency_ms=send_ms,
                request_bytes=len(command.encode("utf-8")),
                response_message_type="hidden_states",
                response_bytes=len(hidden_states),
                exchange_ms=stage_zero_exchange_ms,
            )

            command = json.dumps(
                {
                    "type": "forward_hidden",
                    "request_id": request_id,
                    "step_id": step_id,
                    "phase": phase,
                },
                separators=(",", ":"),
            )
            exchange_started_ns = perf_counter_ns()
            command_send_started_ns = perf_counter_ns()
            await runtime_one.send_text(command)
            command_send_ms = elapsed_ms(command_send_started_ns)
            payload_send_started_ns = perf_counter_ns()
            await runtime_one.send_bytes(hidden_states)
            payload_send_ms = elapsed_ms(payload_send_started_ns)
            stage_one_send_ms = command_send_ms + payload_send_ms

            raw_response = await receive_runtime_message(1)
            stage_one_exchange_ms = elapsed_ms(exchange_started_ns)
            if not isinstance(raw_response, str):
                raise HTTPException(500, "runtime 1 returned an invalid response")
            response_data = parse_runtime_response(
                raw_response,
                expected_type="token",
                request_id=request_id,
                step_id=step_id,
                stage=1,
            )
            runtime_one_metrics = parse_runtime_measurement(response_data["runtime"])
            response_bytes = len(raw_response.encode("utf-8"))
            request_telemetry.record_runtime(
                runtime_one_metrics,
                phase,
                request_message_type="hidden_states",
                request_latency_ms=stage_one_send_ms,
                request_bytes=len(hidden_states),
                response_message_type="token_id",
                response_bytes=response_bytes,
                exchange_ms=stage_one_exchange_ms,
            )

            token_id = response_data.get("token_id")
            if not isinstance(token_id, int):
                raise HTTPException(500, "runtime 1 returned an invalid token")
            input_ids.append(token_id)
            generated_tokens.append(token_id)
            request_telemetry.record_token_ready()

            if token_id == active_tokenizer.eos_token_id:
                break

        detokenize_started_ns = perf_counter_ns()
        text = active_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        detokenize_ms = elapsed_ms(detokenize_started_ns)
        request_e2e_ms = elapsed_ms(request_started_ns)
        request_telemetry.finish(
            prompt_tokens=prompt_tokens,
            request_e2e_ms=request_e2e_ms,
            queue_wait_ms=queue_wait_ms,
            tokenize_ms=tokenize_ms,
            detokenize_ms=detokenize_ms,
        )

    return {"text": text}


def parse_runtime_response(
    message,
    *,
    expected_type: str,
    request_id: str,
    step_id: int,
    stage: int,
):
    if not isinstance(message, str):
        raise HTTPException(500, f"runtime {stage} returned an invalid response")
    try:
        response = json.loads(message)
    except json.JSONDecodeError:
        raise HTTPException(500, f"runtime {stage} returned invalid JSON") from None

    if not isinstance(response, dict):
        raise HTTPException(500, f"runtime {stage} returned an invalid response")
    if response.get("type") == "error":
        raise HTTPException(500, response.get("message", f"runtime {stage} failed"))
    if (
        response.get("type") != expected_type
        or response.get("request_id") != request_id
        or response.get("step_id") != step_id
        or not isinstance(response.get("runtime"), dict)
    ):
        raise HTTPException(500, f"runtime {stage} returned mismatched metadata")
    return response


def parse_runtime_measurement(runtime: dict) -> RuntimeMeasurement:
    try:
        return RuntimeMeasurement.from_payload(runtime)
    except (TypeError, ValueError) as error:
        raise HTTPException(500, "runtime returned invalid metrics") from error


async def receive_runtime_message(stage: int) -> str | bytes:
    if stage not in runtime_inboxes:
        raise HTTPException(503, f"runtime {stage} disconnected")
    return await runtime_inboxes[stage].get()


def raise_runtime_error(message: str) -> NoReturn:
    try:
        response = json.loads(message)
        detail = response.get("message", "runtime returned an invalid response")
    except (json.JSONDecodeError, TypeError):
        detail = "runtime returned an invalid response"
    raise HTTPException(500, detail)


async def start_downloads():
    global download_started, tokenizer
    if download_started:
        return
    download_started = True

    print(
        f"\n{constants.REQUIRED_RUNTIMES} runtimes connected. "
        "Triggering downloads..."
    )
    for stage, websocket in list(runtimes.items()):
        await websocket.send_text(f"download {stage}")
        print(f"Assigned stage {stage} to runtime {stage}")

    await asyncio.to_thread(download_laptop)
    tokenizer = await asyncio.to_thread(
        AutoTokenizer.from_pretrained, ".", local_files_only=True
    )
    print("Laptop tokenizer ready")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    available_stages = set(range(constants.REQUIRED_RUNTIMES)) - set(runtimes)
    if not available_stages:
        await websocket.close()
        return

    stage = min(available_stages)
    runtimes[stage] = websocket
    runtime_inboxes[stage] = asyncio.Queue()
    print(f"Runtime connected ({len(runtimes)}/{constants.REQUIRED_RUNTIMES})")

    if download_started:
        await websocket.send_text(f"download {stage}")
    elif len(runtimes) == constants.REQUIRED_RUNTIMES:
        asyncio.create_task(start_downloads())

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            text = message.get("text")
            if text == f"ready {stage}":
                ready_runtimes.add(stage)
                print(
                    f"Runtime {stage} ready "
                    f"({len(ready_runtimes)}/{constants.REQUIRED_RUNTIMES})"
                )
            elif text is not None:
                await runtime_inboxes[stage].put(text)
            else:
                await runtime_inboxes[stage].put(message["bytes"])
    finally:
        inbox = runtime_inboxes.pop(stage, None)
        if inbox is not None:
            await inbox.put(
                json.dumps(
                    {"type": "error", "message": f"runtime {stage} disconnected"}
                )
            )
        runtimes.pop(stage, None)
        ready_runtimes.discard(stage)
        print(
            "Runtime disconnected "
            f"({len(runtimes)}/{constants.REQUIRED_RUNTIMES})"
        )


if __name__ == "__main__":
    print(
        "Server running on 0.0.0.0:8765. Waiting for "
        f"{constants.REQUIRED_RUNTIMES} runtimes..."
    )
    uvicorn.run(app, host="0.0.0.0", port=8765, ws_max_size=64 * 1024 * 1024)
