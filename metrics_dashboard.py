import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


METRICS_DIR = Path(__file__).resolve().parent / "metrics"


def metrics_file_signature() -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(METRICS_DIR.glob("requests-*.jsonl"))
    )


@st.cache_data(show_spinner=False)
def load_records(
    file_signature: tuple[tuple[str, int, int], ...],
) -> tuple[list[dict[str, Any]], list[str]]:
    records = []
    errors = []
    for filename, _, _ in file_signature:
        path = Path(filename)
        with path.open(encoding="utf-8") as metrics_file:
            for line_number, line in enumerate(metrics_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    errors.append(f"{path.name}:{line_number}: {error.msg}")
                    continue
                if isinstance(record, dict):
                    records.append(record)
                else:
                    errors.append(f"{path.name}:{line_number}: expected a JSON object")
    return records, errors


def request_rows(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        overall = record.get("overall", {})
        tokens = record.get("tokens", {})
        derived = record.get("derived", {})
        rows.append(
            {
                "created_at": record.get("created_at"),
                "day": str(record.get("created_at", ""))[:10],
                "request_id": record.get("request_id"),
                "model": record.get("model"),
                "prompt_tokens": tokens.get("prompt_tokens"),
                "output_tokens": tokens.get("output_tokens"),
                "total_tokens": tokens.get("total_tokens"),
                "request_e2e_ms": overall.get("request_e2e_ms"),
                "ttft_ms": overall.get("ttft_ms"),
                "ttlt_ms": overall.get("ttlt_ms"),
                "tpot_ms": overall.get("tpot_ms"),
                "output_tokens_per_sec": overall.get("output_tokens_per_sec"),
                "total_gpu_ms": derived.get("total_gpu_ms"),
                "total_network_ms": derived.get("total_network_ms"),
                "gpu_percent_of_e2e": derived.get("gpu_percent_of_e2e"),
                "network_percent_of_e2e": derived.get(
                    "network_percent_of_e2e"
                ),
                "bottleneck_node_id": derived.get("bottleneck_node_id"),
                "bottleneck_stage_id": derived.get("bottleneck_stage_id"),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["created_at"] = pd.to_datetime(
            frame["created_at"], errors="coerce", utc=True
        )
        frame = frame.sort_values("created_at")
    return frame


def average_values(mappings: list[dict[str, Any]]) -> dict[str, Any]:
    """Average numeric fields while preserving representative text fields."""
    keys = dict.fromkeys(key for mapping in mappings for key in mapping)
    averaged: dict[str, Any] = {}
    for key in keys:
        values = [mapping.get(key) for mapping in mappings]
        numbers = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if numbers:
            averaged[key] = round(sum(numbers) / len(numbers), 3)
            continue
        present = [value for value in values if value is not None]
        if present:
            try:
                averaged[key] = Counter(present).most_common(1)[0][0]
            except TypeError:
                averaged[key] = present[0]
    return averaged


def average_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Create one run-shaped record containing per-field arithmetic means."""
    node_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    edge_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        for node in record.get("nodes", []):
            key = (
                node.get("node_id"),
                node.get("node_type"),
                node.get("stage_id"),
            )
            node_groups.setdefault(key, []).append(node)
        for edge in record.get("edges", []):
            key = (edge.get("src"), edge.get("dst"), edge.get("message"))
            edge_groups.setdefault(key, []).append(edge)

    nodes = []
    for group in node_groups.values():
        representative = {
            key: value
            for key, value in group[0].items()
            if key != "metrics"
        }
        representative["metrics"] = average_values(
            [node.get("metrics", {}) for node in group]
        )
        nodes.append(representative)

    edges = []
    for group in edge_groups.values():
        representative = {
            key: value
            for key, value in group[0].items()
            if key != "metrics"
        }
        representative["metrics"] = average_values(
            [edge.get("metrics", {}) for edge in group]
        )
        edges.append(representative)

    models = {record.get("model") for record in records if record.get("model")}
    return {
        "request_id": f"average-of-{len(records)}-runs",
        "model": next(iter(models)) if len(models) == 1 else "multiple models",
        "created_at": records[0].get("created_at"),
        "tokens": average_values([record.get("tokens", {}) for record in records]),
        "overall": average_values(
            [record.get("overall", {}) for record in records]
        ),
        "nodes": nodes,
        "edges": edges,
        "derived": average_values(
            [record.get("derived", {}) for record in records]
        ),
    }


def format_ms(value: float | None) -> str:
    return "—" if value is None else f"{value:,.1f} ms"


def format_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:,.2f} tok/s"


def format_bytes(value: float | int | None) -> str:
    if value is None:
        return "—"
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.2f} MiB"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value:.0f} B"


def duration_breakdown(record: dict[str, Any]) -> pd.DataFrame:
    output_tokens = float(record.get("tokens", {}).get("output_tokens") or 0)
    decode_steps = max(output_tokens - 1, 0)
    nodes = record.get("nodes", [])
    edges = record.get("edges", [])
    orchestrator = next(
        (node for node in nodes if node.get("node_type") == "orchestrator"), {}
    ).get("metrics", {})
    tokenizer = next(
        (node for node in nodes if node.get("node_type") == "tokenizer"), {}
    ).get("metrics", {})
    model_nodes = sorted(
        (node for node in nodes if node.get("node_type") == "model_stage"),
        key=lambda node: node.get("stage_id", -1),
    )

    def edge_duration(src: str, dst: str) -> float | None:
        edge = next(
            (
                candidate
                for candidate in edges
                if candidate.get("src") == src and candidate.get("dst") == dst
            ),
            None,
        )
        if edge is None:
            return None
        metrics = edge.get("metrics", {})
        average_latency = metrics.get("latency_ms_avg")
        if isinstance(average_latency, (int, float)):
            return float(average_latency) * output_tokens
        latency = metrics.get("latency_ms")
        return float(latency) if isinstance(latency, (int, float)) else None

    rows: list[tuple[str, float | None]] = [
        ("Client → laptop", edge_duration("client", "laptop")),
        ("Laptop queue", orchestrator.get("queue_wait_ms")),
        ("Laptop tokenization", tokenizer.get("tokenize_ms")),
    ]
    for node in model_nodes:
        node_id = node.get("node_id", "unknown")
        stage_id = node.get("stage_id")
        stage_key = f"stage:{stage_id}"
        metrics = node.get("metrics", {})
        prefill_ms = metrics.get("prefill_gpu_ms") or 0.0
        decode_average = metrics.get("decode_gpu_ms_avg") or 0.0
        gpu_total = float(prefill_ms) + float(decode_average) * decode_steps
        rows.extend(
            [
                (f"Laptop → {node_id}", edge_duration("laptop", stage_key)),
                (f"{node_id} GPU", gpu_total),
            ]
        )
        sample_average = metrics.get("sample_ms_avg")
        if isinstance(sample_average, (int, float)):
            rows.append((f"{node_id} sampling", sample_average * output_tokens))
        rows.append((f"{node_id} → laptop", edge_duration(stage_key, "laptop")))

    rows.extend(
        [
            (
                "Laptop orchestration",
                orchestrator.get("orchestration_overhead_ms"),
            ),
            ("Laptop detokenization", tokenizer.get("detokenize_ms")),
            ("Laptop → client", edge_duration("laptop", "client")),
        ]
    )

    clean_rows = [
        {
            "step": index,
            "component": name,
            "duration_ms": float(value) if isinstance(value, (int, float)) else None,
        }
        for index, (name, value) in enumerate(rows, start=1)
    ]
    known_duration = sum(
        row["duration_ms"] or 0.0 for row in clean_rows
    )
    e2e_ms = record.get("overall", {}).get("request_e2e_ms")
    if isinstance(e2e_ms, (int, float)) and e2e_ms > known_duration:
        insertion_index = max(len(clean_rows) - 2, 0)
        clean_rows.insert(
            insertion_index,
            {
                "step": insertion_index + 1,
                "component": "Runtime serialization / other",
                "duration_ms": e2e_ms - known_duration,
            },
        )
        for index, row in enumerate(clean_rows, start=1):
            row["step"] = index

    return pd.DataFrame(clean_rows)


def mermaid_label(value: Any) -> str:
    return html.escape(str(value), quote=True).replace("\n", "<br/>")


def mermaid_node_id(value: Any) -> str:
    cleaned = "".join(
        character if character.isalnum() else "_" for character in str(value)
    )
    return f"node_{cleaned}"


def mermaid_diagram(record: dict[str, Any]) -> str:
    nodes = record.get("nodes", [])
    edges = record.get("edges", [])
    orchestrator = next(
        (node for node in nodes if node.get("node_type") == "orchestrator"), {}
    ).get("metrics", {})
    tokenizer = next(
        (node for node in nodes if node.get("node_type") == "tokenizer"), {}
    ).get("metrics", {})
    model_nodes = sorted(
        (node for node in nodes if node.get("node_type") == "model_stage"),
        key=lambda node: node.get("stage_id", -1),
    )

    laptop_label = "\n".join(
        [
            "Laptop",
            f"Queue {format_ms(orchestrator.get('queue_wait_ms'))}",
            f"Tokenize {format_ms(tokenizer.get('tokenize_ms'))}",
            f"Orchestration {format_ms(orchestrator.get('orchestration_overhead_ms'))}",
            f"Detokenize {format_ms(tokenizer.get('detokenize_ms'))}",
        ]
    )
    declarations = {
        "client": 'client["Client"]',
        "laptop": f'laptop["{mermaid_label(laptop_label)}"]',
    }
    runtime_ids = []
    for node in model_nodes:
        stage_id = node.get("stage_id")
        metrics = node.get("metrics", {})
        stage_key = f"stage:{stage_id}"
        runtime_id = mermaid_node_id(stage_key)
        runtime_ids.append(runtime_id)
        metric_lines = [
            f"{node.get('node_id')} · stage {stage_id}",
            f"Prefill GPU {format_ms(metrics.get('prefill_gpu_ms'))}",
            f"Decode avg {format_ms(metrics.get('decode_gpu_ms_avg'))}",
            f"Decode p50 {format_ms(metrics.get('decode_gpu_ms_p50'))}",
            f"Decode p95 {format_ms(metrics.get('decode_gpu_ms_p95'))}",
        ]
        if metrics.get("sample_ms_avg") is not None:
            metric_lines.append(
                f"Sample avg {format_ms(metrics.get('sample_ms_avg'))}"
            )
        metric_lines.extend(
            [
                f"GPU allocated {format_bytes(metrics.get('gpu_memory_allocated_bytes'))}",
                f"GPU reserved {format_bytes(metrics.get('gpu_memory_reserved_bytes'))}",
            ]
        )
        declarations[stage_key] = (
            f'{runtime_id}["{mermaid_label(chr(10).join(metric_lines))}"]'
        )

    lines = ["flowchart LR", *[f"    {value}" for value in declarations.values()]]
    for edge in edges:
        src = edge.get("src")
        dst = edge.get("dst")
        if src not in declarations or dst not in declarations:
            continue
        src_id = (
            "client"
            if src == "client"
            else "laptop" if src == "laptop" else mermaid_node_id(src)
        )
        dst_id = (
            "client"
            if dst == "client"
            else "laptop" if dst == "laptop" else mermaid_node_id(dst)
        )
        metrics = edge.get("metrics", {})
        latency = metrics.get("latency_ms_avg", metrics.get("latency_ms"))
        byte_count = metrics.get("bytes_avg", metrics.get("bytes"))
        edge_label = "\n".join(
            [
                str(edge.get("message", "network")),
                format_ms(latency),
                format_bytes(byte_count),
            ]
        )
        lines.append(
            f'    {src_id} -->|"{mermaid_label(edge_label)}"| {dst_id}'
        )

    lines.extend(
        [
            "    classDef clientNode fill:#332C40,color:#F4EEFF,stroke:#C4A7E7,stroke-width:2px",
            "    classDef laptopNode fill:#163B3A,color:#E6FFFA,stroke:#72C7B8,stroke-width:2px",
            "    classDef runtimeNode fill:#272B4D,color:#EEF2FF,stroke:#8EA1E1,stroke-width:2px",
            "    class client clientNode",
            "    class laptop laptopNode",
        ]
    )
    if runtime_ids:
        lines.append(f"    class {','.join(runtime_ids)} runtimeNode")
    return "\n".join(lines)


def render_mermaid(diagram: str) -> None:
    st.iframe(
        f"""
        <style>
          body {{ margin: 0; background: #0F172A; }}
          .mermaid {{ display: flex; justify-content: center; padding: 28px; }}
          .mermaid svg {{ width: 100%; min-width: 1100px; max-width: 1350px !important; }}
        </style>
        <pre class="mermaid">{html.escape(diagram)}</pre>
        <script type="module">
          import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
          mermaid.initialize({{
            startOnLoad: true,
            theme: "base",
            flowchart: {{
              curve: "basis",
              nodeSpacing: 110,
              rankSpacing: 145,
              padding: 10
            }},
            themeVariables: {{
              background: "#0F172A",
              primaryTextColor: "#E5E7EB",
              lineColor: "#94A3B8",
              edgeLabelBackground: "#1E293B",
              fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
              fontSize: "14px"
            }},
            themeCSS: ".edgeLabel p {{ color: #E5E7EB !important; background: #1E293B !important; border: 1px solid #64748B; border-radius: 8px; padding: 6px 8px; }} .node rect {{ rx: 10px; ry: 10px; }}"
          }});
        </script>
        """,
        height=720,
    )


def filtered_dashboard_data():
    st.caption(f"Reading request records from {METRICS_DIR}")
    if st.sidebar.button("Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    signature = metrics_file_signature()
    records, load_errors = load_records(signature)
    if not records:
        st.info("No request metrics found yet. Run a successful completion first.")
        if load_errors:
            st.error("\n".join(load_errors))
        st.stop()

    requests = request_rows(records)
    models = sorted(value for value in requests["model"].dropna().unique())
    selected_models = st.sidebar.multiselect("Models", models, default=models)
    if not selected_models:
        st.warning("Select at least one model.")
        st.stop()
    selected_ids = set(
        requests.loc[requests["model"].isin(selected_models), "request_id"]
    )
    requests = requests[requests["request_id"].isin(selected_ids)]

    days = sorted(requests["day"].dropna().unique(), reverse=True)
    if not days:
        st.warning("No dated requests match the current filters.")
        st.stop()
    selected_day = st.sidebar.selectbox("Day", days)
    selected_ids = set(requests.loc[requests["day"] == selected_day, "request_id"])
    requests = requests[requests["request_id"].isin(selected_ids)]
    visible_records = [
        record for record in records if record.get("request_id") in selected_ids
    ]

    if load_errors:
        with st.expander(f"Skipped {len(load_errors)} malformed JSONL line(s)"):
            st.code("\n".join(load_errors))
    if requests.empty:
        st.warning("No requests match the current filters.")
        st.stop()
    return visible_records, requests, selected_day


def render_record_metrics(
    record: dict[str, Any],
    *,
    inspect_label: str,
) -> None:
    overall = record.get("overall", {})
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "E2E", format_ms(overall.get("request_e2e_ms")), border=True
    )
    metric_columns[1].metric(
        "TTFT", format_ms(overall.get("ttft_ms")), border=True
    )
    metric_columns[2].metric(
        "TPOT", format_ms(overall.get("tpot_ms")), border=True
    )
    metric_columns[3].metric(
        "Throughput",
        format_rate(overall.get("output_tokens_per_sec")),
        border=True,
    )

    st.subheader("What took how long")
    with st.container(border=True):
        st.caption(
            "Ordered by pipeline position. Repeated decode and network work is "
            "summed across the run. Client HTTP latency is unavailable from the "
            "server."
        )
        breakdown = duration_breakdown(record)
        left, right = st.columns([2, 1])
        with left:
            if breakdown.empty:
                st.info("No duration measurements are available.")
            else:
                chart_data = breakdown[["component", "duration_ms"]].copy()
                chart_data["duration_ms"] = chart_data["duration_ms"].fillna(0.0)
                st.bar_chart(
                    chart_data,
                    x="component",
                    y="duration_ms",
                    horizontal=True,
                    sort=False,
                    color="#72A99A",
                    x_label="Pipeline step",
                    y_label="Milliseconds",
                    height=max(440, len(chart_data) * 38),
                )
        with right:
            st.dataframe(
                breakdown,
                width="stretch",
                hide_index=True,
                column_config={
                    "step": st.column_config.NumberColumn("Step", format="%d"),
                    "component": "Pipeline component",
                    "duration_ms": st.column_config.NumberColumn(
                        "Duration", format="%.2f ms"
                    ),
                },
            )

    st.subheader("Request path")
    with st.container(border=True):
        render_mermaid(mermaid_diagram(record))

    with st.expander(inspect_label, expanded=False):
        st.json(record)


def render_run_metrics_page() -> None:
    st.title("Run Metrics")
    records, requests, selected_day = filtered_dashboard_data()
    record_by_id = {record.get("request_id"): record for record in records}
    visible_ids = requests.sort_values("created_at", ascending=False)[
        "request_id"
    ].tolist()
    selected_request_id = st.sidebar.selectbox(
        "Run",
        visible_ids,
        format_func=lambda request_id: (
            f"{record_by_id[request_id].get('created_at', '')[11:19]} · "
            f"{request_id[:8]}"
        ),
    )
    selected_record = record_by_id[selected_request_id]

    st.caption(f"{selected_day} · request {selected_request_id}")
    render_record_metrics(selected_record, inspect_label="Inspect one request")


def render_overall_metrics_page() -> None:
    st.title("Overall Metrics")
    records, requests, selected_day = filtered_dashboard_data()
    averaged_record = average_records(records)
    st.caption(f"{selected_day} · averaged across {len(requests):,} runs")
    render_record_metrics(
        averaged_record,
        inspect_label="Inspect averaged metrics",
    )


def main() -> None:
    st.set_page_config(page_title="Pipeline Metrics", layout="wide")
    navigation = st.navigation(
        [
            st.Page(
                render_run_metrics_page,
                title="Run Metrics",
                icon=":material/timeline:",
                default=True,
            ),
            st.Page(
                render_overall_metrics_page,
                title="Overall Metrics",
                icon=":material/monitoring:",
            ),
        ],
        position="sidebar",
        expanded=True,
    )
    navigation.run()


if __name__ == "__main__":
    main()
