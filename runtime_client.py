import argparse
import asyncio
import json

from safetensors.torch import load, save
from websockets.asyncio.client import connect

from download import download_stage
from stage_runtime import StageRuntime, load_stage


async def run(url):
    stage_runtime: StageRuntime | None = None
    waiting_for_hidden_states = False

    async with connect(url, max_size=None) as ws:
        print("Connected to server")
        async for msg in ws:
            try:
                if isinstance(msg, bytes):
                    if not waiting_for_hidden_states:
                        raise ValueError("Received hidden states without a command")
                    if stage_runtime is None:
                        raise RuntimeError("Stage is not loaded")

                    waiting_for_hidden_states = False
                    hidden_states = load(msg)["hidden_states"]
                    token_id = stage_runtime.sample_token(hidden_states)
                    await ws.send(json.dumps({"type": "token", "token_id": token_id}))
                    continue

                if msg.startswith("download "):
                    stage = int(msg.split()[1])
                    print(f"Downloading stage {stage} files...")
                    await asyncio.to_thread(download_stage, stage)
                    print(f"Loading stage {stage}...")
                    stage_runtime = await asyncio.to_thread(load_stage, stage)
                    await ws.send(f"ready {stage}")
                    print(f"Stage {stage} ready. Staying connected.")
                    continue

                command = json.loads(msg)
                if command["type"] == "forward_tokens":
                    if stage_runtime is None:
                        raise RuntimeError("Stage is not loaded")
                    hidden_states = stage_runtime.forward_tokens(command["input_ids"])
                    await ws.send(
                        save({"hidden_states": hidden_states.cpu().contiguous()})
                    )
                elif command["type"] == "forward_hidden":
                    if stage_runtime is None:
                        raise RuntimeError("Stage is not loaded")
                    waiting_for_hidden_states = True
            except Exception as error:
                await ws.send(json.dumps({"type": "error", "message": str(error)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    args = parser.parse_args()
    asyncio.run(run(args.url))
