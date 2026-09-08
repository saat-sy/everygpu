"""Distributed text-generation pipeline."""

import asyncio
import json
from time import perf_counter_ns
from typing import NoReturn
from uuid import uuid4

from fastapi import HTTPException
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import config
from coordinator.runtimes import RuntimePool
from model.download import download_coordinator
from telemetry.recorder import RequestTelemetry, RuntimeMeasurement, elapsed_ms


class Pipeline:
    def __init__(self, runtimes: RuntimePool):
        self.runtimes = runtimes
        self.download_started = False
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return (
            self.runtimes.ready == self.runtimes.required
            and self.tokenizer is not None
        )

    def status(self) -> dict:
        return {
            "connected_runtimes": self.runtimes.connected,
            "ready_runtimes": self.runtimes.ready,
            "ready_stages": self.runtimes.ready_stages,
            "required_runtimes": self.runtimes.required,
            "download_started": self.download_started,
            "pipeline_ready": self.ready,
        }

    async def prepare(self) -> None:
        if self.download_started:
            return
        self.download_started = True

        print(f"\n{self.runtimes.required} runtimes connected. Triggering downloads...")
        for stage, websocket in self.runtimes.connections():
            await websocket.send_text(f"download {stage}")
            print(f"Assigned stage {stage} to runtime {stage}")

        await asyncio.to_thread(download_coordinator)
        self.tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained, ".", local_files_only=True
        )
        print("Coordinator tokenizer ready")

    async def complete(self, request: dict) -> dict[str, str]:
        request_started_ns = perf_counter_ns()
        request_id = uuid4().hex
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise HTTPException(400, "prompt must be a non-empty string")
        model = request.get("model", config.MODEL_NAME)
        if not isinstance(model, str) or not model:
            raise HTTPException(400, "model must be a non-empty string")

        request_bytes = len(json.dumps(request, separators=(",", ":")).encode())
        request_telemetry = RequestTelemetry.start(
            request_id=request_id,
            model=model,
            request_started_ns=request_started_ns,
            request_bytes=request_bytes,
        )
        queue_started_ns = perf_counter_ns()

        async with self._lock:
            queue_wait_ms = elapsed_ms(queue_started_ns)
            tokenizer = self.tokenizer
            runtime_zero = self.runtimes.socket(0)
            runtime_one = self.runtimes.socket(1)
            if (
                not self.ready
                or tokenizer is None
                or runtime_zero is None
                or runtime_one is None
            ):
                raise HTTPException(503, "pipeline is not ready")

            tokenize_started_ns = perf_counter_ns()
            input_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
            tokenize_ms = elapsed_ms(tokenize_started_ns)
            prompt_tokens = len(input_ids)
            generated_tokens = []

            for step_id in range(config.MAX_NEW_TOKENS):
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
                    await self.runtimes.receive(0),
                    expected_type="hidden_states",
                    request_id=request_id,
                    step_id=step_id,
                    stage=0,
                )
                hidden_states = await self.runtimes.receive(0)
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
                    request_bytes=len(command.encode()),
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

                raw_response = await self.runtimes.receive(1)
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
                runtime_one_metrics = parse_runtime_measurement(
                    response_data["runtime"]
                )
                request_telemetry.record_runtime(
                    runtime_one_metrics,
                    phase,
                    request_message_type="hidden_states",
                    request_latency_ms=stage_one_send_ms,
                    request_bytes=len(hidden_states),
                    response_message_type="token_id",
                    response_bytes=len(raw_response.encode()),
                    exchange_ms=stage_one_exchange_ms,
                )

                token_id = response_data.get("token_id")
                if not isinstance(token_id, int):
                    raise HTTPException(500, "runtime 1 returned an invalid token")
                input_ids.append(token_id)
                generated_tokens.append(token_id)
                request_telemetry.record_token_ready()

                if token_id == tokenizer.eos_token_id:
                    break

            detokenize_started_ns = perf_counter_ns()
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            if not isinstance(text, str):
                raise HTTPException(500, "tokenizer returned invalid decoded text")
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
    message: str | bytes,
    *,
    expected_type: str,
    request_id: str,
    step_id: int,
    stage: int,
) -> dict:
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


def raise_runtime_error(message: str) -> NoReturn:
    try:
        response = json.loads(message)
        detail = response.get("message", "runtime returned an invalid response")
    except (json.JSONDecodeError, TypeError):
        detail = "runtime returned an invalid response"
    raise HTTPException(500, detail)
