"""Read completed pipeline traces from Tempo."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import streamlit as st

from telemetry.profiler.trace import build_profile

TEMPO_URL = os.getenv("TEMPO_URL", "http://127.0.0.1:3200").rstrip("/")
SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "server")
LOOKBACK_HOURS = int(os.getenv("TEMPO_LOOKBACK_HOURS", "168"))
TRACE_NAME = "pipeline.request"


class TempoError(RuntimeError):
    """Raised when Tempo cannot serve profiler data."""


@dataclass(frozen=True, slots=True)
class TraceSummary:
    trace_id: str
    started_ns: int
    duration_ms: float

    @property
    def label(self) -> str:
        started_at = datetime.fromtimestamp(
            self.started_ns / 1_000_000_000,
            tz=timezone.utc,
        ).astimezone()
        return (
            f"{started_at:%Y-%m-%d %H:%M:%S} · "
            f"{self.duration_ms:,.1f} ms · {self.trace_id[:8]}"
        )


def _tempo_json(path: str, parameters: dict[str, str | int] | None = None) -> Any:
    url = f"{TEMPO_URL}{path}"
    if parameters:
        url = f"{url}?{urlencode(parameters)}"
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=10) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read(500).decode("utf-8", errors="replace")
        raise TempoError(f"Tempo returned HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError) as error:
        raise TempoError(f"Cannot reach Tempo at {TEMPO_URL}: {error}") from error


@st.cache_data(show_spinner=False, ttl=30)
def list_profiles() -> list[TraceSummary]:
    now = int(time.time())
    response = _tempo_json(
        "/api/search",
        {
            "q": (
                f'{{ resource.service.name = "{SERVICE_NAME}" '
                f'&& name = "{TRACE_NAME}" }}'
            ),
            "start": now - LOOKBACK_HOURS * 60 * 60,
            "end": now,
            "limit": 100,
        },
    )
    summaries = []
    for trace in response.get("traces", []):
        trace_id = trace.get("traceID")
        if not isinstance(trace_id, str):
            continue
        if trace.get("rootTraceName") not in (None, TRACE_NAME):
            continue
        try:
            summaries.append(
                TraceSummary(
                    trace_id=trace_id,
                    started_ns=int(trace.get("startTimeUnixNano", 0)),
                    duration_ms=float(trace.get("durationMs", 0)),
                )
            )
        except (TypeError, ValueError):
            continue
    return sorted(summaries, key=lambda item: item.started_ns, reverse=True)


@st.cache_data(show_spinner=False, ttl=30)
def load_profile(trace_id: str) -> dict[str, Any]:
    return build_profile(_tempo_json(f"/api/traces/{trace_id}"), trace_id)
