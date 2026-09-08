import argparse
import asyncio
import json
import traceback
from time import perf_counter_ns

from safetensors.torch import load, save
from websockets.asyncio.client import connect

from model.download import download_stage
from worker.stage import StageRuntime, load_stage


def runtime_metrics(stage_runtime, node_id, gpu_ms, processing_ms):
    first_layer, last_layer = stage_runtime.stage["layers"]
    return {
        "node_id": node_id,
        "stage_id": stage_runtime.stage["id"],
        "device": str(stage_runtime.device),
        "layers": {"start": first_layer, "end": last_layer},
        "gpu_ms": gpu_ms,
        "processing_ms": processing_ms,
        "gpu_memory_reserved_bytes": stage_runtime.gpu_memory_reserved_bytes(),
    }


async def run(url, configured_node_id=None):
    stage_runtime: StageRuntime | None = None
    pending_hidden_command = None
    node_id = configured_node_id

    async with connect(url, max_size=None) as ws:
        print("Connected to server")
        async for msg in ws:
            try:
                if isinstance(msg, bytes):
                    if pending_hidden_command is None:
                        raise ValueError("Received hidden states without a command")
                    if stage_runtime is None:
                        raise RuntimeError("Stage is not loaded")

                    command = pending_hidden_command
                    pending_hidden_command = None
                    processing_start = perf_counter_ns()
                    hidden_states = load(msg)["hidden_states"]
                    token_id, gpu_ms = stage_runtime.timed_sample_token(hidden_states)
                    processing_ms = (perf_counter_ns() - processing_start) / 1_000_000
                    response = {
                        "type": "token",
                        "request_id": command["request_id"],
                        "step_id": command["step_id"],
                        "phase": command["phase"],
                        "token_id": token_id,
                        "runtime": runtime_metrics(
                            stage_runtime,
                            node_id,
                            gpu_ms,
                            processing_ms,
                        ),
                    }
                    await ws.send(json.dumps(response, separators=(",", ":")))
                    continue

                if msg.startswith("download "):
                    stage = int(msg.split()[1])
                    print(f"Downloading stage {stage} files...")
                    await asyncio.to_thread(download_stage, stage)
                    print(f"Loading stage {stage}...")
                    stage_runtime = await asyncio.to_thread(load_stage, stage)
                    node_id = configured_node_id or f"runtime-{stage}"
                    await ws.send(f"ready {stage}")
                    print(f"Stage {stage} ready. Staying connected.")
                    continue

                command = json.loads(msg)
                if command["type"] == "forward_tokens":
                    if stage_runtime is None:
                        raise RuntimeError("Stage is not loaded")
                    processing_start = perf_counter_ns()
                    hidden_states, gpu_ms = stage_runtime.timed_forward_tokens(
                        command["input_ids"]
                    )
                    payload = save({"hidden_states": hidden_states.cpu().contiguous()})
                    processing_ms = (perf_counter_ns() - processing_start) / 1_000_000
                    response = {
                        "type": "hidden_states",
                        "request_id": command["request_id"],
                        "step_id": command["step_id"],
                        "phase": command["phase"],
                        "runtime": runtime_metrics(
                            stage_runtime, node_id, gpu_ms, processing_ms
                        ),
                    }
                    await ws.send(json.dumps(response, separators=(",", ":")))
                    await ws.send(payload)
                elif command["type"] == "forward_hidden":
                    if stage_runtime is None:
                        raise RuntimeError("Stage is not loaded")
                    pending_hidden_command = command
            # Report request failures without terminating the long-lived runtime.
            except Exception as error:  # noqa: BLE001
                traceback.print_exc()
                await ws.send(json.dumps({"type": "error", "message": str(error)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--node-id")
    args = parser.parse_args()
    asyncio.run(run(args.url, args.node_id))


if __name__ == "__main__":
    main()
