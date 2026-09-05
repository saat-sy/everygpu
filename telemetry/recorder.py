"""Record request metrics and trace details with OpenTelemetry."""

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from time import perf_counter_ns
from typing import Literal, Self, TypeGuard

from opentelemetry import trace

from . import constants, otel

Phase = Literal["prefill", "decode"]


def elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / constants.NANOSECONDS_PER_MILLISECOND


@dataclass(frozen=True, slots=True)
class RuntimeMeasurement:
    node_id: str
    stage_id: int
    device: str
    layer_start: int
    layer_end: int
    gpu_ms: float
    processing_ms: float
    gpu_memory_reserved_bytes: int

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> Self:
        layers = payload.get("layers")
        if not isinstance(layers, dict):
            raise TypeError("invalid layer metadata")

        node_id = payload.get("node_id")
        stage_id = payload.get("stage_id")
        device = payload.get("device")
        layer_start = layers.get("start")
        layer_end = layers.get("end")
        gpu_ms = payload.get("gpu_ms")
        processing_ms = payload.get("processing_ms")
        reserved_bytes = payload.get("gpu_memory_reserved_bytes")

        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("invalid node ID")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("invalid device")
        if not _nonnegative_int(stage_id):
            raise ValueError("invalid stage ID")
        if not _nonnegative_int(layer_start) or not _nonnegative_int(layer_end):
            raise ValueError("invalid layer range")
        if layer_end < layer_start:
            raise ValueError("invalid layer range")
        if not _nonnegative_number(gpu_ms):
            raise ValueError("invalid GPU duration")
        if not _nonnegative_number(processing_ms):
            raise ValueError("invalid processing duration")
        if not _nonnegative_int(reserved_bytes):
            raise ValueError("invalid reserved GPU memory")

        return cls(
            node_id=node_id,
            stage_id=stage_id,
            device=device,
            layer_start=layer_start,
            layer_end=layer_end,
            gpu_ms=float(gpu_ms),
            processing_ms=float(processing_ms),
            gpu_memory_reserved_bytes=reserved_bytes,
        )


@dataclass(slots=True)
class RequestTelemetry:
    model: str
    request_started_ns: int
    _token_ready_times_ns: list[int] = field(default_factory=list)
    _runtime_exchange_ms: float = 0.0

    @classmethod
    def start(
        cls,
        *,
        request_id: str,
        model: str,
        request_started_ns: int,
        request_bytes: int,
    ) -> Self:
        trace.get_current_span().set_attributes(
            {
                "request_id": request_id,
                "model": model,
                "created_at": datetime.now().astimezone().isoformat(),
            }
        )
        request = cls(model=model, request_started_ns=request_started_ns)
        request._record_edge(
            constants.CLIENT_ENDPOINT_ID,
            constants.SERVER_ENDPOINT_ID,
            "http_request",
            request_bytes,
        )
        return request

    def record_token_ready(self) -> None:
        self._token_ready_times_ns.append(perf_counter_ns())

    def record_runtime(
        self,
        measurement: RuntimeMeasurement,
        phase: Phase,
        *,
        request_message_type: str,
        request_latency_ms: float,
        request_bytes: int,
        response_message_type: str,
        response_bytes: int,
        exchange_ms: float,
    ) -> None:
        self._runtime_exchange_ms += exchange_ms
        attributes = {
            constants.MODEL_ATTRIBUTE: self.model,
            constants.NODE_ID_ATTRIBUTE: measurement.node_id,
            constants.STAGE_ID_ATTRIBUTE: measurement.stage_id,
            constants.DEVICE_ATTRIBUTE: measurement.device,
            constants.LAYER_START_ATTRIBUTE: measurement.layer_start,
            constants.LAYER_END_ATTRIBUTE: measurement.layer_end,
        }
        phased_attributes = {**attributes, constants.PHASE_ATTRIBUTE: phase}
        otel.stage_gpu_duration.record(measurement.gpu_ms, phased_attributes)
        otel.stage_processing_duration.record(
            measurement.processing_ms,
            phased_attributes,
        )
        otel.stage_gpu_memory_reserved.set(
            measurement.gpu_memory_reserved_bytes,
            attributes,
        )
        trace.get_current_span().add_event(
            "stage_runtime",
            {
                **phased_attributes,
                "gpu_ms": measurement.gpu_ms,
                "processing_ms": measurement.processing_ms,
                "gpu_memory_reserved_bytes": measurement.gpu_memory_reserved_bytes,
            },
        )
        self._record_edge(
            constants.SERVER_ENDPOINT_ID,
            measurement.node_id,
            request_message_type,
            request_bytes,
            request_latency_ms,
        )
        self._record_edge(
            measurement.node_id,
            constants.SERVER_ENDPOINT_ID,
            response_message_type,
            response_bytes,
            max(exchange_ms - request_latency_ms - measurement.processing_ms, 0.0),
        )

    def finish(
        self,
        *,
        prompt_tokens: int,
        request_e2e_ms: float,
        queue_wait_ms: float,
        tokenize_ms: float,
        detokenize_ms: float,
    ) -> None:
        output_tokens = len(self._token_ready_times_ns)
        first_token_ns = (
            self._token_ready_times_ns[0] if self._token_ready_times_ns else None
        )
        last_token_ns = (
            self._token_ready_times_ns[-1] if self._token_ready_times_ns else None
        )
        ttft_ms = self._since_start_ms(first_token_ns)
        ttlt_ms = self._since_start_ms(last_token_ns)
        tpot_ms = (
            (last_token_ns - first_token_ns)
            / constants.NANOSECONDS_PER_MILLISECOND
            / (output_tokens - 1)
            if first_token_ns is not None
            and last_token_ns is not None
            and output_tokens > 1
            else None
        )
        throughput = (
            output_tokens / (request_e2e_ms / constants.MILLISECONDS_PER_SECOND)
            if request_e2e_ms > 0
            else 0.0
        )
        orchestration_ms = max(
            request_e2e_ms
            - queue_wait_ms
            - tokenize_ms
            - detokenize_ms
            - self._runtime_exchange_ms,
            0.0,
        )
        model = {constants.MODEL_ATTRIBUTE: self.model}

        otel.request_duration.record(request_e2e_ms, model)
        otel.request_throughput.record(throughput, model)
        for token_type, value in (
            ("prompt", prompt_tokens),
            ("output", output_tokens),
            ("total", prompt_tokens + output_tokens),
        ):
            otel.request_tokens.record(
                value,
                {**model, constants.TOKEN_TYPE_ATTRIBUTE: token_type},
            )
        for timing, value in (("ttft", ttft_ms), ("ttlt", ttlt_ms), ("tpot", tpot_ms)):
            if value is not None:
                otel.request_token_latency.record(
                    value,
                    {**model, constants.TIMING_ATTRIBUTE: timing},
                )
        for phase, value in (
            ("queue", queue_wait_ms),
            ("orchestration", orchestration_ms),
            ("tokenize", tokenize_ms),
            ("detokenize", detokenize_ms),
        ):
            otel.server_duration.record(
                value,
                {**model, constants.SERVER_PHASE_ATTRIBUTE: phase},
            )

        span_attributes: dict[str, int | float] = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "request_e2e_ms": request_e2e_ms,
            "output_tokens_per_second": throughput,
            "queue_wait_ms": queue_wait_ms,
            "orchestration_overhead_ms": orchestration_ms,
            "tokenize_ms": tokenize_ms,
            "detokenize_ms": detokenize_ms,
        }
        for name, value in (
            ("time_to_first_token_ms", ttft_ms),
            ("time_to_last_token_ms", ttlt_ms),
            ("time_per_output_token_ms", tpot_ms),
        ):
            if value is not None:
                span_attributes[name] = value
        trace.get_current_span().set_attributes(span_attributes)

    def _since_start_ms(self, timestamp_ns: int | None) -> float | None:
        if timestamp_ns is None:
            return None
        return (
            timestamp_ns - self.request_started_ns
        ) / constants.NANOSECONDS_PER_MILLISECOND

    def _record_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        message_type: str,
        payload_bytes: int,
        latency_ms: float | None = None,
    ) -> None:
        attributes = {
            constants.MODEL_ATTRIBUTE: self.model,
            constants.SOURCE_NODE_ID_ATTRIBUTE: source_node_id,
            constants.TARGET_NODE_ID_ATTRIBUTE: target_node_id,
            constants.MESSAGE_TYPE_ATTRIBUTE: message_type,
        }
        otel.edge_payload_size.record(payload_bytes, attributes)
        event: dict[str, str | int | float] = {
            **attributes,
            "bytes": payload_bytes,
        }
        if latency_ms is not None:
            otel.edge_duration.record(latency_ms, attributes)
            event["latency_ms"] = latency_ms
        trace.get_current_span().add_event("pipeline_edge", event)


def _nonnegative_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _nonnegative_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
        and value >= 0
    )
