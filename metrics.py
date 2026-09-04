import json
import math
from datetime import datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Any


METRICS_DIR = Path(__file__).resolve().parent / "metrics"


def elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def rounded(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def append_metric_record(record: dict[str, Any]) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    date = datetime.fromisoformat(record["created_at"]).date().isoformat()
    path = METRICS_DIR / f"requests-{date}.jsonl"
    with path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(record, separators=(",", ":")))
        metrics_file.write("\n")
    return path
