"""Render a pipeline profile with Streamlit."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from telemetry.profiler.trace import number


def _format_ms(value: Any) -> str:
    return f"{number(value):,.1f} ms"


def _format_rate(value: Any) -> str:
    return f"{number(value):,.2f} tok/s"


def _format_bytes(value: Any) -> str:
    byte_count = number(value)
    if byte_count >= 1024 * 1024:
        return f"{byte_count / (1024 * 1024):.2f} MiB"
    if byte_count >= 1024:
        return f"{byte_count / 1024:.1f} KiB"
    return f"{byte_count:.0f} B"


def _dot_string(value: Any) -> str:
    return json.dumps(str(value))


def pipeline_graph(profile: dict[str, Any]) -> str:
    attributes = profile["attributes"]
    lines = [
        "digraph pipeline {",
        '  graph [rankdir="LR", bgcolor="transparent", pad="0.2", nodesep="0.7"];',
        '  node [shape="box", style="rounded,filled", fontname="sans-serif", color="#64748b", fontcolor="#e2e8f0"];',
        '  edge [fontname="sans-serif", color="#94a3b8", fontcolor="#cbd5e1"];',
        '  client [label="Client", fillcolor="#332c40"];',
        (
            "  server [label="
            + _dot_string(
                "Server\n"
                f"Queue {_format_ms(attributes.get('queue_wait_ms'))}\n"
                f"Tokenize {_format_ms(attributes.get('tokenize_ms'))}\n"
                f"Orchestrate {_format_ms(attributes.get('orchestration_overhead_ms'))}"
            )
            + ', fillcolor="#163b3a"];'
        ),
    ]
    node_names = {"client": "client", "server": "server"}
    for stage in profile["stages"]:
        node_name = f"stage_{stage['stage_id']}"
        node_names[stage["node_id"]] = node_name
        label = (
            f"{stage['node_id']} · stage {stage['stage_id']}\n"
            f"Layers {stage['layer_start']}–{stage['layer_end']}\n"
            f"Prefill GPU {_format_ms(stage['prefill_gpu_ms'])}\n"
            f"Decode avg {_format_ms(stage['decode_gpu_ms_avg'])}\n"
            f"GPU reserved {_format_bytes(stage['memory_bytes'])}"
        )
        lines.append(
            f'  {node_name} [label={_dot_string(label)}, fillcolor="#272b4d"];'
        )

    for edge in profile["edges"]:
        source = node_names.get(edge["source"])
        target = node_names.get(edge["target"])
        if source is None or target is None:
            continue
        label_lines = [edge["messages"] or "request"]
        if edge["latency_ms_avg"] is not None:
            label_lines.append(f"avg {_format_ms(edge['latency_ms_avg'])}")
        if edge["payload_bytes_avg"] is not None:
            label_lines.append(f"avg {_format_bytes(edge['payload_bytes_avg'])}")
        lines.append(
            f"  {source} -> {target} "
            f"[label={_dot_string(chr(10).join(label_lines))}];"
        )
    lines.append("}")
    return "\n".join(lines)


def render_profile(profile: dict[str, Any]) -> None:
    attributes = profile["attributes"]
    request_id = str(attributes.get("request_id", profile["trace_id"]))
    model = attributes.get("model", "unknown model")
    prompt_tokens = int(number(attributes.get("prompt_tokens")))
    output_tokens = int(number(attributes.get("output_tokens")))
    st.caption(
        f"{model} · request {request_id[:12]} · "
        f"{prompt_tokens:,} prompt tokens · {output_tokens:,} output tokens"
    )

    columns = st.columns(4)
    columns[0].metric("E2E", _format_ms(attributes.get("request_e2e_ms")), border=True)
    columns[1].metric(
        "TTFT", _format_ms(attributes.get("time_to_first_token_ms")), border=True
    )
    columns[2].metric(
        "TPOT", _format_ms(attributes.get("time_per_output_token_ms")), border=True
    )
    columns[3].metric(
        "Throughput",
        _format_rate(attributes.get("output_tokens_per_second")),
        border=True,
    )

    st.subheader("What took how long")
    st.bar_chart(
        profile["durations"],
        x="component",
        y="duration_ms",
        horizontal=True,
        sort=False,
        color="#72a99a",
        x_label="Pipeline component",
        y_label="Milliseconds",
        height=max(380, len(profile["durations"]) * 38),
    )
    st.dataframe(
        profile["durations"],
        width="stretch",
        hide_index=True,
        column_config={
            "component": "Pipeline component",
            "duration_ms": st.column_config.NumberColumn("Duration", format="%.2f ms"),
        },
    )

    st.subheader("Request path")
    st.graphviz_chart(pipeline_graph(profile), width="stretch")

    with st.expander("Stage details"):
        st.dataframe(profile["stages"], width="stretch", hide_index=True)

    with st.expander("Trace details"):
        st.json(
            {
                "trace_id": profile["trace_id"],
                "attributes": attributes,
                "events": profile["events"],
            }
        )
