import asyncio
import json
from datetime import datetime
from time import perf_counter_ns
from typing import NoReturn
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import uvicorn

from download import download_laptop
from metrics import append_metric_record, average, elapsed_ms, percentile, rounded

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.router.redirect_slashes = False

runtimes: dict[int, WebSocket] = {}
runtime_inboxes: dict[int, asyncio.Queue[str | bytes]] = {}
ready_runtimes: set[int] = set()
REQUIRED_RUNTIMES = 2
MAX_NEW_TOKENS = 20
MODEL_NAME = "olmoe-pipeline-fp16"
download_started = False
tokenizer: PreTrainedTokenizerBase | None = None
pipeline_lock = asyncio.Lock()
metrics_write_lock = asyncio.Lock()


@app.get("/health")
async def health():
    return {
        "connected_runtimes": len(runtimes),
        "ready_runtimes": len(ready_runtimes),
        "ready_stages": sorted(ready_runtimes),
        "required_runtimes": REQUIRED_RUNTIMES,
        "download_started": download_started,
        "pipeline_ready": (
            len(ready_runtimes) == REQUIRED_RUNTIMES and tokenizer is not None
        ),
    }


@app.post("/v1/completions")
async def completions(request: dict):
    request_started_ns = perf_counter_ns()
    created_at = datetime.now().astimezone().isoformat()
    request_id = uuid4().hex
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise HTTPException(400, "prompt must be a non-empty string")
    model = request.get("model", MODEL_NAME)
    if not isinstance(model, str) or not model:
        raise HTTPException(400, "model must be a non-empty string")

    request_bytes = len(
        json.dumps(request, separators=(",", ":")).encode("utf-8")
    )
    queue_started_ns = perf_counter_ns()

    async with pipeline_lock:
        queue_wait_ms = elapsed_ms(queue_started_ns)
        active_tokenizer = tokenizer
        runtime_zero = runtimes.get(0)
        runtime_one = runtimes.get(1)
        if (
            len(ready_runtimes) != REQUIRED_RUNTIMES
            or active_tokenizer is None
            or runtime_zero is None
            or runtime_one is None
        ):
            raise HTTPException(503, "pipeline is not ready")

        tokenize_started_ns = perf_counter_ns()
        input_ids = list(
            active_tokenizer.encode(prompt, add_special_tokens=False)
        )
        tokenize_ms = elapsed_ms(tokenize_started_ns)
        prompt_tokens = len(input_ids)
        generated_tokens = []
        token_ready_times_ns = []
        runtime_measurements = {}
        edge_measurements = {}
        hidden_payload_sizes = []
        runtime_exchange_ms = 0.0

        for step_id in range(MAX_NEW_TOKENS):
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
            add_edge_measurement(
                edge_measurements,
                "laptop",
                "stage:0",
                "token_ids",
                send_ms,
                len(command.encode("utf-8")),
            )

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
            runtime_exchange_ms += stage_zero_exchange_ms
            stage_zero_processing_ms = runtime_processing_ms(runtime_zero_response)
            add_edge_measurement(
                edge_measurements,
                "stage:0",
                "laptop",
                "hidden_states",
                max(stage_zero_exchange_ms - send_ms - stage_zero_processing_ms, 0.0),
                len(hidden_states),
            )
            hidden_payload_sizes.append(len(hidden_states))
            record_runtime_measurement(
                runtime_measurements, runtime_zero_response["runtime"], phase
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
            add_edge_measurement(
                edge_measurements,
                "laptop",
                "stage:1",
                "hidden_states",
                stage_one_send_ms,
                len(hidden_states),
            )

            raw_response = await receive_runtime_message(1)
            stage_one_exchange_ms = elapsed_ms(exchange_started_ns)
            runtime_exchange_ms += stage_one_exchange_ms
            if not isinstance(raw_response, str):
                raise HTTPException(500, "runtime 1 returned an invalid response")
            response_data = parse_runtime_response(
                raw_response,
                expected_type="token",
                request_id=request_id,
                step_id=step_id,
                stage=1,
            )
            stage_one_processing_ms = runtime_processing_ms(response_data)
            response_bytes = len(raw_response.encode("utf-8"))
            add_edge_measurement(
                edge_measurements,
                "stage:1",
                "laptop",
                "token_id",
                max(
                    stage_one_exchange_ms
                    - stage_one_send_ms
                    - stage_one_processing_ms,
                    0.0,
                ),
                response_bytes,
            )
            record_runtime_measurement(
                runtime_measurements, response_data["runtime"], phase
            )

            token_id = response_data.get("token_id")
            if not isinstance(token_id, int):
                raise HTTPException(500, "runtime 1 returned an invalid token")
            input_ids.append(token_id)
            generated_tokens.append(token_id)
            token_ready_times_ns.append(perf_counter_ns())

            if token_id == active_tokenizer.eos_token_id:
                break

        detokenize_started_ns = perf_counter_ns()
        text = active_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        detokenize_ms = elapsed_ms(detokenize_started_ns)
        request_e2e_ms = elapsed_ms(request_started_ns)
        orchestration_overhead_ms = max(
            request_e2e_ms
            - queue_wait_ms
            - tokenize_ms
            - detokenize_ms
            - runtime_exchange_ms,
            0.0,
        )

        metrics_record = build_metrics_record(
            request_id=request_id,
            model=model,
            created_at=created_at,
            prompt_tokens=prompt_tokens,
            output_tokens=len(generated_tokens),
            request_e2e_ms=request_e2e_ms,
            token_ready_times_ns=token_ready_times_ns,
            request_started_ns=request_started_ns,
            queue_wait_ms=queue_wait_ms,
            orchestration_overhead_ms=orchestration_overhead_ms,
            tokenize_ms=tokenize_ms,
            detokenize_ms=detokenize_ms,
            runtime_measurements=runtime_measurements,
            edge_measurements=edge_measurements,
            request_bytes=request_bytes,
            hidden_payload_sizes=hidden_payload_sizes,
        )

    response = {"text": text, "metrics": metrics_record}
    for _ in range(3):
        response_bytes = len(
            json.dumps(response, separators=(",", ":")).encode("utf-8")
        )
        metrics_record["edges"][-1]["metrics"]["bytes"] = response_bytes

    async with metrics_write_lock:
        await asyncio.to_thread(append_metric_record, metrics_record)

    return response


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


def runtime_processing_ms(response: dict) -> float:
    processing_ms = response["runtime"].get("processing_ms")
    if not isinstance(processing_ms, (int, float)):
        return 0.0
    return float(processing_ms)


def add_edge_measurement(
    measurements,
    src: str,
    dst: str,
    message: str,
    latency_ms: float,
    byte_count: int,
):
    edge = measurements.setdefault(
        (src, dst, message), {"latencies_ms": [], "bytes": []}
    )
    edge["latencies_ms"].append(latency_ms)
    edge["bytes"].append(byte_count)


def record_runtime_measurement(measurements, runtime: dict, phase: str):
    stage_id = runtime.get("stage_id")
    node_id = runtime.get("node_id")
    if not isinstance(stage_id, int) or not isinstance(node_id, str):
        raise HTTPException(500, "runtime returned invalid metrics metadata")

    entry = measurements.setdefault(
        stage_id,
        {
            "node_id": node_id,
            "stage_id": stage_id,
            "device": runtime.get("device", "unknown"),
            "layers": runtime.get("layers", {}),
            "prefill_gpu_ms": [],
            "decode_gpu_ms": [],
            "sample_ms": [],
            "gpu_memory_allocated_bytes": 0,
            "gpu_memory_reserved_bytes": 0,
        },
    )
    gpu_ms = runtime.get("gpu_ms")
    if isinstance(gpu_ms, (int, float)):
        entry[f"{phase}_gpu_ms"].append(float(gpu_ms))
    sample_ms = runtime.get("sample_ms")
    if isinstance(sample_ms, (int, float)):
        entry["sample_ms"].append(float(sample_ms))
    entry["gpu_memory_allocated_bytes"] = max(
        entry["gpu_memory_allocated_bytes"],
        int(runtime.get("gpu_memory_allocated_bytes", 0)),
    )
    entry["gpu_memory_reserved_bytes"] = max(
        entry["gpu_memory_reserved_bytes"],
        int(runtime.get("gpu_memory_reserved_bytes", 0)),
    )


def build_runtime_nodes(runtime_measurements):
    nodes = []
    for stage_id in sorted(runtime_measurements):
        runtime = runtime_measurements[stage_id]
        prefill = runtime["prefill_gpu_ms"]
        decode = runtime["decode_gpu_ms"]
        node_metrics = {
            "prefill_gpu_ms": rounded(sum(prefill) if prefill else None),
            "decode_gpu_ms_avg": rounded(average(decode)),
            "decode_gpu_ms_p50": rounded(percentile(decode, 0.50)),
            "decode_gpu_ms_p95": rounded(percentile(decode, 0.95)),
        }
        if runtime["sample_ms"]:
            node_metrics["sample_ms_avg"] = rounded(average(runtime["sample_ms"]))
        node_metrics.update(
            {
                "gpu_memory_allocated_bytes": runtime[
                    "gpu_memory_allocated_bytes"
                ],
                "gpu_memory_reserved_bytes": runtime["gpu_memory_reserved_bytes"],
            }
        )
        nodes.append(
            {
                "node_id": runtime["node_id"],
                "node_type": "model_stage",
                "stage_id": stage_id,
                "device": runtime["device"],
                "layers": runtime["layers"],
                "metrics": node_metrics,
            }
        )
    return nodes


def aggregate_edge(edge_measurements, src, dst, message):
    samples = edge_measurements.get(
        (src, dst, message), {"latencies_ms": [], "bytes": []}
    )
    return {
        "src": src,
        "dst": dst,
        "message": message,
        "metrics": {
            "latency_ms_avg": rounded(average(samples["latencies_ms"])),
            "latency_ms_p95": rounded(percentile(samples["latencies_ms"], 0.95)),
            "bytes_avg": rounded(average(samples["bytes"])),
        },
    }


def build_metrics_record(
    *,
    request_id,
    model,
    created_at,
    prompt_tokens,
    output_tokens,
    request_e2e_ms,
    token_ready_times_ns,
    request_started_ns,
    queue_wait_ms,
    orchestration_overhead_ms,
    tokenize_ms,
    detokenize_ms,
    runtime_measurements,
    edge_measurements,
    request_bytes,
    hidden_payload_sizes,
):
    ttft_ms = (
        (token_ready_times_ns[0] - request_started_ns) / 1_000_000
        if token_ready_times_ns
        else None
    )
    tpot_ms = (
        (token_ready_times_ns[-1] - token_ready_times_ns[0])
        / 1_000_000
        / (len(token_ready_times_ns) - 1)
        if len(token_ready_times_ns) > 1
        else None
    )
    output_tokens_per_sec = (
        output_tokens / (request_e2e_ms / 1000) if request_e2e_ms > 0 else 0.0
    )

    runtime_nodes = build_runtime_nodes(runtime_measurements)
    nodes = [
        {
            "node_id": "laptop",
            "node_type": "orchestrator",
            "device": "cpu",
            "metrics": {
                "queue_wait_ms": rounded(queue_wait_ms),
                "orchestration_overhead_ms": rounded(orchestration_overhead_ms),
            },
        },
        {
            "node_id": "laptop",
            "node_type": "tokenizer",
            "device": "cpu",
            "metrics": {
                "tokenize_ms": rounded(tokenize_ms),
                "detokenize_ms": rounded(detokenize_ms),
            },
        },
        *runtime_nodes,
    ]

    edges = [
        {
            "src": "client",
            "dst": "laptop",
            "message": "http_request",
            "metrics": {"latency_ms": None, "bytes": request_bytes},
        },
        aggregate_edge(edge_measurements, "laptop", "stage:0", "token_ids"),
        aggregate_edge(edge_measurements, "stage:0", "laptop", "hidden_states"),
        aggregate_edge(edge_measurements, "laptop", "stage:1", "hidden_states"),
        aggregate_edge(edge_measurements, "stage:1", "laptop", "token_id"),
        {
            "src": "laptop",
            "dst": "client",
            "message": "http_response",
            "metrics": {"latency_ms": None, "bytes": 0},
        },
    ]

    total_gpu_ms = 0.0
    prefill_gpu_ms = 0.0
    decode_gpu_ms = 0.0
    runtime_totals = []
    for runtime in runtime_measurements.values():
        prefill_total = sum(runtime["prefill_gpu_ms"])
        decode_total = sum(runtime["decode_gpu_ms"])
        total = prefill_total + decode_total
        prefill_gpu_ms += prefill_total
        decode_gpu_ms += decode_total
        total_gpu_ms += total
        runtime_totals.append((total, runtime))

    total_network_ms = sum(
        sum(samples["latencies_ms"]) for samples in edge_measurements.values()
    )
    bottleneck = max(runtime_totals, key=lambda item: item[0])[1] if runtime_totals else None
    nonzero_runtime_totals = [total for total, _ in runtime_totals if total > 0]
    stage_balance_ratio = (
        max(nonzero_runtime_totals) / min(nonzero_runtime_totals)
        if len(nonzero_runtime_totals) >= 2
        else 1.0 if nonzero_runtime_totals else None
    )
    hidden_bytes_avg = average(hidden_payload_sizes)
    decode_steps = max(output_tokens - 1, 0)

    return {
        "request_id": request_id,
        "model": model,
        "created_at": created_at,
        "tokens": {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
        },
        "overall": {
            "request_e2e_ms": rounded(request_e2e_ms),
            "ttft_ms": rounded(ttft_ms),
            "ttlt_ms": rounded(request_e2e_ms),
            "tpot_ms": rounded(tpot_ms),
            "output_tokens_per_sec": rounded(output_tokens_per_sec),
        },
        "nodes": nodes,
        "edges": edges,
        "derived": {
            "total_gpu_ms": rounded(total_gpu_ms),
            "total_network_ms": rounded(total_network_ms),
            "network_percent_of_e2e": rounded(
                total_network_ms / request_e2e_ms * 100
                if request_e2e_ms > 0
                else 0.0
            ),
            "gpu_percent_of_e2e": rounded(
                total_gpu_ms / request_e2e_ms * 100
                if request_e2e_ms > 0
                else 0.0
            ),
            "prefill_ms_per_prompt_token": rounded(
                prefill_gpu_ms / prompt_tokens if prompt_tokens else None
            ),
            "decode_ms_per_output_token": rounded(
                decode_gpu_ms / decode_steps if decode_steps else None
            ),
            "bottleneck_node_id": bottleneck["node_id"] if bottleneck else None,
            "bottleneck_stage_id": bottleneck["stage_id"] if bottleneck else None,
            "stage_balance_ratio": rounded(stage_balance_ratio),
            "hidden_bytes_avg": rounded(hidden_bytes_avg),
            "hidden_mb_per_output_token": rounded(
                hidden_bytes_avg / (1024 * 1024)
                if hidden_bytes_avg is not None
                else None
            ),
        },
    }


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

    print(f"\n{REQUIRED_RUNTIMES} runtimes connected. Triggering downloads...")
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

    available_stages = set(range(REQUIRED_RUNTIMES)) - set(runtimes)
    if not available_stages:
        await websocket.close()
        return

    stage = min(available_stages)
    runtimes[stage] = websocket
    runtime_inboxes[stage] = asyncio.Queue()
    print(f"Runtime connected ({len(runtimes)}/{REQUIRED_RUNTIMES})")

    if download_started:
        await websocket.send_text(f"download {stage}")
    elif len(runtimes) == REQUIRED_RUNTIMES:
        asyncio.create_task(start_downloads())

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            text = message.get("text")
            if text == f"ready {stage}":
                ready_runtimes.add(stage)
                print(f"Runtime {stage} ready ({len(ready_runtimes)}/{REQUIRED_RUNTIMES})")
            elif text is not None:
                await runtime_inboxes[stage].put(text)
            else:
                await runtime_inboxes[stage].put(message["bytes"])
    finally:
        inbox = runtime_inboxes.pop(stage, None)
        if inbox is not None:
            await inbox.put(
                json.dumps({"type": "error", "message": f"runtime {stage} disconnected"})
            )
        runtimes.pop(stage, None)
        ready_runtimes.discard(stage)
        print(f"Runtime disconnected ({len(runtimes)}/{REQUIRED_RUNTIMES})")


if __name__ == "__main__":
    print(f"Server running on 0.0.0.0:8765. Waiting for {REQUIRED_RUNTIMES} runtimes...")
    uvicorn.run(app, host="0.0.0.0", port=8765, ws_max_size=64 * 1024 * 1024)
