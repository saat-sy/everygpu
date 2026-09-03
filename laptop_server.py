import asyncio
import json
from typing import NoReturn

from fastapi import FastAPI, HTTPException, WebSocket
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import uvicorn

from download import download_laptop

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.router.redirect_slashes = False

runtimes: dict[int, WebSocket] = {}
runtime_inboxes: dict[int, asyncio.Queue[str | bytes]] = {}
ready_runtimes: set[int] = set()
REQUIRED_RUNTIMES = 2
MAX_NEW_TOKENS = 20
download_started = False
tokenizer: PreTrainedTokenizerBase | None = None
pipeline_lock = asyncio.Lock()


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
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise HTTPException(400, "prompt must be a non-empty string")

    async with pipeline_lock:
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

        input_ids = list(
            active_tokenizer.encode(prompt, add_special_tokens=False)
        )
        generated_tokens = []

        for _ in range(MAX_NEW_TOKENS):
            await runtime_zero.send_text(
                json.dumps({"type": "forward_tokens", "input_ids": input_ids})
            )
            hidden_states = await receive_runtime_message(0)
            if not isinstance(hidden_states, bytes):
                raise_runtime_error(hidden_states)

            await runtime_one.send_text(json.dumps({"type": "forward_hidden"}))
            await runtime_one.send_bytes(hidden_states)

            response = await receive_runtime_message(1)
            if not isinstance(response, str):
                raise HTTPException(500, "runtime 1 returned an invalid response")

            try:
                response_data = json.loads(response)
            except json.JSONDecodeError:
                raise HTTPException(500, "runtime 1 returned invalid JSON") from None

            if not isinstance(response_data, dict):
                raise HTTPException(500, "runtime 1 returned an invalid response")
            if response_data.get("type") == "error":
                raise HTTPException(
                    500, response_data.get("message", "runtime 1 failed")
                )

            token_id = response_data.get("token_id")
            if not isinstance(token_id, int):
                raise HTTPException(500, "runtime 1 returned an invalid token")
            input_ids.append(token_id)
            generated_tokens.append(token_id)

            if token_id == active_tokenizer.eos_token_id:
                break

        text = active_tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {"text": text}


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
