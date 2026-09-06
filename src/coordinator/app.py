"""FastAPI entry point for the pipeline coordinator."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket
from opentelemetry.trace import SpanKind

import config
from coordinator.pipeline import Pipeline
from coordinator.runtimes import RuntimePool
from telemetry import otel

runtimes = RuntimePool(config.REQUIRED_RUNTIMES)
pipeline = Pipeline(runtimes)


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


@app.get("/health")
async def health():
    return pipeline.status()


@app.post("/v1/completions")
async def completions(request: dict):
    with otel.tracer.start_as_current_span(
        config.PIPELINE_TRACE_NAME,
        kind=SpanKind.SERVER,
    ):
        return await pipeline.complete(request)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stage = runtimes.connect(websocket)
    if stage is None:
        await websocket.close()
        return

    try:
        if pipeline.download_started:
            await websocket.send_text(f"download {stage}")
        elif runtimes.connected == runtimes.required:
            asyncio.create_task(pipeline.prepare())
        await runtimes.listen(stage)
    finally:
        await runtimes.disconnect(stage)


def main() -> None:
    print(
        "Server running on 0.0.0.0:8765. Waiting for "
        f"{config.REQUIRED_RUNTIMES} runtimes..."
    )
    uvicorn.run(app, host="0.0.0.0", port=8765, ws_max_size=64 * 1024 * 1024)


if __name__ == "__main__":
    main()
