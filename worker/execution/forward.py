"""FORWARD handler: validates, runs this worker's stage off the event loop, commits session caches."""

import dataclasses
import logging
import math
import time
import types

import torch
import trio

from monitoring.logging import logger as log_utils
from monitoring.telemetry import constants as telemetry_constants
from monitoring.telemetry import device as device_telemetry
from monitoring.telemetry import metrics
from monitoring.telemetry import phases
from monitoring.telemetry import slow
from monitoring.telemetry import timing
from monitoring.telemetry import tracing
from worker.execution import models
from worker.execution import relay
from worker.execution import stage as stage_lib

logger = logging.getLogger(__name__)
_DTYPES = types.MappingProxyType(
    {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
)
_DTYPE_NAMES = types.MappingProxyType({dtype: name for name, dtype in _DTYPES.items()})


class StageRunner:
    """Holds the executable stage once weights are resident; `forward` is the server's handler."""

    def __init__(self):
        self.stage: stage_lib.Stage | None = None
        self.unavailable = "no stage is resident"
        self._device = trio.CapacityLimiter(1)
        self._threads = trio.CapacityLimiter(1)

    def install(self, config: dict, weights: dict) -> None:
        """Build the stage; on failure forwards stay refused with the reason."""
        try:
            stage = stage_lib.Stage(config, weights)
            if stage.dtype not in _DTYPE_NAMES:
                raise models.StageError(f"unsupported weight dtype {stage.dtype}")
        except models.StageError as error:
            self.stage, self.unavailable = None, str(error)
            raise
        self.stage = stage

    def descriptor(self) -> models.StageDescriptor | None:
        """The stage this worker executes, or None while unavailable."""
        return self.stage and self.stage.descriptor

    async def forward(self, session: models.Session, meta: dict, payload: bytes):
        stage = self.stage
        if stage is None:
            raise models.RequestError("unavailable", self.unavailable)
        request = _parse(stage, meta, payload)
        _check_sender(session, meta)
        _check_live(session)
        session.scope = scope = trio.CancelScope()
        with scope:
            return await self._run(session, stage, request, payload)
        raise models.RequestError("unknown_session", "session released")

    async def _run(self, session, stage, request, payload: bytes):
        phase = timing.phase(request.step)
        async with phases.waited(session.lock, "worker.session_lock_wait", phase=phase):
            _check_live(session)
            if request.position != session.state.get("position", 0):
                raise models.RequestError(
                    "bad_position", "position does not continue the session"
                )
            if request.step != session.state.get("step", 0):
                raise models.RequestError(
                    "bad_step", "step does not continue the session"
                )
            caches = session.state.get("kv", {})
            started = time.monotonic()
            async with phases.waited(self._device, "worker.device_wait", phase=phase):
                extended, output = await trio.to_thread.run_sync(
                    _step,
                    stage,
                    request,
                    payload,
                    caches,
                    time.monotonic(),
                    limiter=self._threads,
                )
            logger.debug(
                "stage computed step=%d position=%d seq=%d layers=%d-%d elapsed_ms=%d",
                request.step,
                request.position,
                request.seq,
                stage.first,
                stage.stop,
                log_utils.elapsed_ms(started),
            )
            _check_live(session)
            session.state["kv"] = {**caches, **extended}
            session.state["position"] = request.position + request.seq
            session.state["step"] = request.step + 1
        if isinstance(output, int):
            return {
                "token_id": output,
                "step": request.step,
                "position": request.position,
            }, b""
        next_hop = {
            "session_id": session.session_id,
            "target": "layer",
            "layer_idx": stage.stop,
            "position": request.position,
            "temperature": request.temperature,
            "step": request.step,
            "sender": session.route[session.index].peer,
            "shape": [1, request.seq, stage.cfg.hidden],
            "dtype": _DTYPE_NAMES[stage.dtype],
        }
        reply = await relay.forward(session, next_hop, output, request.trace)
        return reply, b""


def _check_live(session: models.Session) -> None:
    if session.closed:
        raise models.RequestError("unknown_session", "session released")


def _check_sender(session: models.Session, meta: dict) -> None:
    """Activations are accepted only from the previous hop of the admitted route."""
    if meta.get("target") != "layer":
        return
    if (
        session.index == 0
        or meta.get("sender") != session.route[session.index - 1].peer
    ):
        raise models.RequestError(
            "bad_sender", "activations must come from the previous hop"
        )


def _bad(message: str) -> models.RequestError:
    return models.RequestError("bad_request", message)


def _parse(stage: stage_lib.Stage, meta: dict, payload: bytes) -> models.ForwardRequest:
    cfg = stage.cfg
    position = meta.get("position")
    if type(position) is not int or position < 0:
        raise _bad("position must be a non-negative integer")
    temperature = _temperature(meta.get("temperature", 0.0))
    step = meta.get("step")
    if type(step) is not int or step < 0:
        raise _bad("step must be a non-negative integer")
    target = meta.get("target")
    if target == "embedding":
        request = _parse_embedding(stage, meta, payload, position, temperature)
    elif target == "layer":
        request = _parse_layer(stage, meta, payload, position, temperature)
    else:
        raise _bad("target must be 'embedding' or 'layer'")
    if position + request.seq > cfg.max_positions:
        raise models.RequestError(
            "too_long", f"sequence exceeds {cfg.max_positions} positions"
        )
    trace = tracing.clean(meta.get(telemetry_constants.TRACE_KEY))
    return dataclasses.replace(request, step=step, trace=trace)


def _parse_embedding(stage, meta, payload, position, temperature):
    ids = meta.get("token_ids")
    if not stage.has_embedding:
        raise models.RequestError("wrong_stage", "this worker holds no embedding")
    if payload or "shape" in meta or "dtype" in meta:
        raise _bad("embedding requests carry no activations")
    if not (
        isinstance(ids, list)
        and 0 < len(ids) <= stage.cfg.max_positions
        and all(type(i) is int and 0 <= i < stage.cfg.vocab for i in ids)
    ):
        raise _bad("token_ids must be non-empty ids within the vocabulary")
    return models.ForwardRequest(position, temperature, len(ids), token_ids=ids)


def _parse_layer(stage, meta, payload, position, temperature):
    layer_idx, shape, dtype = (
        meta.get("layer_idx"),
        meta.get("shape"),
        meta.get("dtype"),
    )
    if type(layer_idx) is not int:
        raise _bad("layer_idx must be an integer")
    if stage.has_embedding or layer_idx != stage.first:
        raise models.RequestError(
            "wrong_stage", f"this stage starts at layer {stage.first}"
        )
    if not (isinstance(dtype, str) and dtype in _DTYPES):
        raise _bad("dtype must be float32, float16 or bfloat16")
    if not (
        isinstance(shape, list)
        and len(shape) == 3
        and all(type(d) is int for d in shape)
        and shape[0] == 1
        and 1 <= shape[1] <= stage.cfg.max_positions
        and shape[2] == stage.cfg.hidden
    ):
        raise _bad(f"activations must be [1, seq, {stage.cfg.hidden}]")
    if len(payload) != math.prod(shape) * _DTYPES[dtype].itemsize:
        raise _bad("payload length does not match shape and dtype")
    return models.ForwardRequest(
        position, temperature, shape[1], shape=tuple(shape), dtype=dtype
    )


def _temperature(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _bad("temperature must be a number")
    try:
        converted = float(value)
    except OverflowError:
        converted = math.inf
    if not math.isfinite(converted) or converted < 0:
        raise _bad("temperature must be finite and >= 0")
    return converted


def _step(
    stage, request: models.ForwardRequest, payload: bytes, caches: dict, queued: float
):
    """Blocking work: returns the extended caches and a sampled id or serialized activations."""
    phase = timing.phase(request.step)
    handoff = time.monotonic() - queued
    metrics.record("worker.thread_handoff", handoff, phase=phase)
    timer = device_telemetry.timer(torch, stage.device)
    attributes = {
        "everygpu.phase": phase,
        "everygpu.step": request.step,
        "everygpu.seq": request.seq,
        "everygpu.position": request.position,
        "everygpu.stage": f"{stage.first}-{stage.stop}",
        "everygpu.thread_handoff_seconds": handoff,
    }
    with tracing.span("worker.stage_execute", attributes) as span:
        with phases.timed("worker.stage_execute", phase=phase):
            extended, output = _compute(stage, request, payload, caches, timer, phase)
        _report_gpu(timer.read(), span, stage, request, phase)
    return extended, output


def _compute(stage, request, payload, caches, timer, phase):
    with torch.inference_mode():
        hidden = None
        if request.token_ids is None:
            with (
                tracing.span("worker.cpu_prepare"),
                phases.timed("worker.cpu_prepare", phase=phase),
            ):
                host = torch.frombuffer(
                    bytearray(payload), dtype=_DTYPES[request.dtype]
                ).reshape(request.shape)
            with timer.interval("h2d"):
                hidden = host.to(device=stage.device, dtype=stage.dtype)
        with timer.interval("compute"):
            if hidden is None:
                hidden = stage.embed(request.token_ids)
            hidden, extended = stage.run_layers(hidden, request.position, caches)
            if stage.completes:
                return extended, stage.sample(hidden, request.temperature)
        with timer.interval("d2h"):
            host = hidden.cpu().contiguous()
        with (
            tracing.span("worker.serialize"),
            phases.timed("worker.serialize", phase=phase),
        ):
            return extended, host.view(torch.uint8).numpy().tobytes()


def _report_gpu(readings: dict, span, stage, request, phase: str) -> None:
    for part, seconds in readings.items():
        metrics.record("worker.gpu", seconds, part=part, phase=phase)
        span.set_attribute(f"everygpu.gpu.{part}_seconds", seconds)
    if "compute" in readings:
        slow.check(
            telemetry_constants.KIND_GPU,
            (
                stage.first,
                stage.stop,
                phase,
                timing.bucket(request.seq),
                timing.bucket(request.position),
            ),
            readings["compute"],
            span,
            {
                "stage": f"{stage.first}-{stage.stop}",
                "step": request.step,
                "seq": request.seq,
                "position": request.position,
                "phase": phase,
            },
            phase=phase,
        )
