"""Turn an OpenTelemetry trace into a simple pipeline profile."""

from __future__ import annotations

from typing import Any

TRACE_NAME = "pipeline.request"


class InvalidProfileError(RuntimeError):
    """Raised when a trace does not contain a pipeline profile."""


def _otel_value(value: dict[str, Any]) -> Any:
    for key in ("stringValue", "intValue", "doubleValue", "boolValue", "bytesValue"):
        if key not in value:
            continue
        result = value[key]
        if key == "intValue":
            try:
                return int(result)
            except (TypeError, ValueError):
                return result
        return result
    if "arrayValue" in value:
        return [_otel_value(item) for item in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return _attributes(value["kvlistValue"].get("values", []))
    return None


def _attributes(items: list[dict[str, Any]] | None) -> dict[str, Any]:
    attributes = {}
    for item in items or []:
        key = item.get("key")
        value = item.get("value")
        if isinstance(key, str) and isinstance(value, dict):
            attributes[key] = _otel_value(value)
    return attributes


def _spans(trace: dict[str, Any]) -> list[dict[str, Any]]:
    body = trace.get("trace", trace)
    batches = body.get("batches") or body.get("resourceSpans") or []
    spans = []
    for batch in batches:
        scopes = (
            batch.get("scopeSpans") or batch.get("instrumentationLibrarySpans") or []
        )
        for scope in scopes:
            spans.extend(scope.get("spans", []))
    return spans


def number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def build_profile(trace: dict[str, Any], trace_id: str) -> dict[str, Any]:
    span = next(
        (
            candidate
            for candidate in _spans(trace)
            if candidate.get("name") == TRACE_NAME
        ),
        None,
    )
    if span is None:
        raise InvalidProfileError(
            f"Trace {trace_id} does not contain a {TRACE_NAME} span"
        )

    attributes = _attributes(span.get("attributes"))
    events = [
        {
            "name": event.get("name"),
            "time_unix_nano": event.get("timeUnixNano"),
            "attributes": _attributes(event.get("attributes")),
        }
        for event in span.get("events", [])
    ]

    stage_groups: dict[tuple[int, str], dict[str, Any]] = {}
    edge_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        event_attributes = event["attributes"]
        if event["name"] == "stage_runtime":
            _add_stage_event(stage_groups, event_attributes)
        elif event["name"] == "pipeline_edge":
            _add_edge_event(edge_groups, event_attributes)

    stages = _summarize_stages(stage_groups)
    edges = _summarize_edges(edge_groups)
    return {
        "trace_id": trace_id,
        "attributes": attributes,
        "events": events,
        "stages": stages,
        "edges": edges,
        "durations": _duration_rows(attributes, stages, edges),
    }


def _add_stage_event(
    groups: dict[tuple[int, str], dict[str, Any]], attributes: dict[str, Any]
) -> None:
    stage_id = attributes.get("stage_id")
    node_id = attributes.get("node_id")
    if not isinstance(stage_id, int) or not isinstance(node_id, str):
        return
    group = groups.setdefault(
        (stage_id, node_id),
        {
            "stage_id": stage_id,
            "node_id": node_id,
            "device": attributes.get("device", "unknown"),
            "layer_start": attributes.get("layer_start"),
            "layer_end": attributes.get("layer_end"),
            "memory_bytes": 0,
            "gpu_ms": [],
            "processing_ms": [],
            "prefill_gpu_ms": [],
            "decode_gpu_ms": [],
        },
    )
    gpu_ms = number(attributes.get("gpu_ms"))
    group["gpu_ms"].append(gpu_ms)
    group["processing_ms"].append(number(attributes.get("processing_ms")))
    group["memory_bytes"] = max(
        group["memory_bytes"],
        int(number(attributes.get("gpu_memory_reserved_bytes"))),
    )
    phase = attributes.get("phase")
    if phase == "prefill":
        group["prefill_gpu_ms"].append(gpu_ms)
    elif phase == "decode":
        group["decode_gpu_ms"].append(gpu_ms)


def _add_edge_event(
    groups: dict[tuple[str, str], dict[str, Any]], attributes: dict[str, Any]
) -> None:
    source = attributes.get("source_node_id")
    target = attributes.get("target_node_id")
    if not isinstance(source, str) or not isinstance(target, str):
        return
    group = groups.setdefault(
        (source, target),
        {
            "source": source,
            "target": target,
            "messages": set(),
            "latencies_ms": [],
            "payload_bytes": [],
        },
    )
    message = attributes.get("message_type")
    if isinstance(message, str):
        group["messages"].add(message)
    latency_ms = attributes.get("latency_ms")
    if isinstance(latency_ms, (int, float)):
        group["latencies_ms"].append(float(latency_ms))
    payload_bytes = attributes.get("bytes")
    if isinstance(payload_bytes, (int, float)):
        group["payload_bytes"].append(float(payload_bytes))


def _summarize_stages(groups: dict[tuple[int, str], dict[str, Any]]) -> list[dict]:
    stages = []
    for group in sorted(groups.values(), key=lambda item: item["stage_id"]):
        decode = group.pop("decode_gpu_ms")
        prefill = group.pop("prefill_gpu_ms")
        gpu = group.pop("gpu_ms")
        processing = group.pop("processing_ms")
        stages.append(
            {
                **group,
                "prefill_gpu_ms": sum(prefill),
                "decode_gpu_ms_avg": sum(decode) / len(decode) if decode else None,
                "total_gpu_ms": sum(gpu),
                "total_processing_ms": sum(processing),
                "calls": len(gpu),
            }
        )
    return stages


def _summarize_edges(groups: dict[tuple[str, str], dict[str, Any]]) -> list[dict]:
    edges = []
    for group in groups.values():
        latencies = group.pop("latencies_ms")
        payloads = group.pop("payload_bytes")
        messages = group.pop("messages")
        edges.append(
            {
                **group,
                "messages": ", ".join(sorted(messages)),
                "latency_ms_avg": sum(latencies) / len(latencies)
                if latencies
                else None,
                "latency_ms_total": sum(latencies),
                "payload_bytes_avg": sum(payloads) / len(payloads)
                if payloads
                else None,
                "calls": max(len(latencies), len(payloads)),
            }
        )
    return edges


def _duration_rows(
    attributes: dict[str, Any], stages: list[dict], edges: list[dict]
) -> list[dict]:
    rows = [
        {"component": "Queue wait", "duration_ms": number(attributes.get("queue_wait_ms"))},
        {"component": "Tokenization", "duration_ms": number(attributes.get("tokenize_ms"))},
    ]
    edge_by_pair = {(edge["source"], edge["target"]): edge for edge in edges}
    for stage in stages:
        node_id = stage["node_id"]
        request_edge = edge_by_pair.get(("server", node_id))
        response_edge = edge_by_pair.get((node_id, "server"))
        if request_edge:
            rows.append(
                {
                    "component": f"Server → {node_id}",
                    "duration_ms": request_edge["latency_ms_total"],
                }
            )
        rows.append(
            {"component": f"{node_id} GPU", "duration_ms": stage["total_gpu_ms"]}
        )
        if response_edge:
            rows.append(
                {
                    "component": f"{node_id} → server",
                    "duration_ms": response_edge["latency_ms_total"],
                }
            )
    rows.extend(
        [
            {
                "component": "Orchestration",
                "duration_ms": number(attributes.get("orchestration_overhead_ms")),
            },
            {
                "component": "Detokenization",
                "duration_ms": number(attributes.get("detokenize_ms")),
            },
        ]
    )
    return rows
