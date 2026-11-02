"""Constants for telemetry: configuration, bounds, slow-operation tuning and the metric catalogue."""

import types

ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
METRIC_PREFIX = "everygpu."
SCOPE_NAME = "everygpu"
SERVICE_CLIENT = "everygpu-client"
SERVICE_WORKER = "everygpu-worker"

TRACE_KEY = "trace"
TELEMETRY_KEY = "telemetry"
MAX_TRACEPARENT_CHARS = 128
MAX_TRACESTATE_CHARS = 256
MAX_ROUTE_EDGES = 64

SPAN_QUEUE_SIZE = 4096
SPAN_BATCH_SIZE = 512
SPAN_DELAY_MILLIS = 2000
EXPORT_TIMEOUT_SECONDS = 5
METRIC_INTERVAL_MILLIS = 10000
SHUTDOWN_SECONDS = 7
FAILURE_LOG_SECONDS = 60
OTEL_LOGGER = "opentelemetry"

TOLERANCE_SECONDS = 0.001
CANCELLATION_NAMES = frozenset(
    {"Cancelled", "CancelledError", "GeneratorExit", "KeyboardInterrupt"}
)

BASELINE_WINDOW = 64
BASELINE_WARMUP = 16
BASELINE_FACTOR = 3.0
BASELINE_MIN_SECONDS = 0.001
BASELINE_MAX_GROUPS = 256
KIND_COMMUNICATION = "communication"
KIND_GPU = "gpu"

PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"
PHASE_ADMIT = "admit"
CONNECTION_COLD = "cold"
CONNECTION_REUSED = "reused"
EDGE_CLIENT = "client_worker"
EDGE_RELAY = "worker_worker"
CLIENT_PEER = "client"
KNOWN_CODES = frozenset(
    {
        "unavailable",
        "unknown_session",
        "bad_request",
        "bad_position",
        "bad_step",
        "bad_sender",
        "bad_route",
        "wrong_stage",
        "too_long",
        "duplicate_session",
        "at_capacity",
        "stale_stage",
        "malformed_frame",
        "internal",
        "downstream_timeout",
        "downstream_unreachable",
        "downstream_failed",
    }
)
OTHER = "other"

COUNTER, UPDOWN, HISTOGRAM = "counter", "updown", "histogram"
SECONDS_BUCKETS = (
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)
BYTES_BUCKETS = tuple(256 * 4**i for i in range(10))
RATE_BUCKETS = tuple(1024 * 4**i for i in range(12))
SMALL_BUCKETS = (1, 2, 4, 8, 16, 32, 64)


def _seconds(description: str) -> tuple:
    return (HISTOGRAM, "s", description, SECONDS_BUCKETS)


# key: (kind, unit, description, buckets)
METRICS = types.MappingProxyType(
    {
        "request.duration": _seconds("HTTP request lifetime through body completion"),
        "request.ttft": _seconds("Request start to first sampled token at the client"),
        "request.first_content_delivery": _seconds(
            "Request start to first nonempty streamed content accepted by ASGI send"
        ),
        "request.generation_duration": _seconds("Generation loop duration"),
        "request.tpot": _seconds("Mean time per output token after the first"),
        "token.interval": _seconds("Interval between sampled-token arrivals"),
        "setup.total": _seconds("Route discovery, admission and preparation"),
        "setup.discovery_lock_wait": _seconds("Wait for the discovery lock"),
        "setup.checkpoint_lookup": _seconds("Network pin and manifest lookup"),
        "setup.provider_lookup": _seconds("Worker provider lookup"),
        "setup.route_selection": _seconds("Route selection"),
        "setup.admission": _seconds("Admission on every selected worker"),
        "setup.tokenizer_wait": _seconds("Wait for the tokenizer lock"),
        "setup.tokenizer_load": _seconds("Tokenizer fetch and load"),
        "setup.prompt_encoding": _seconds("Chat template and tokenization"),
        "hop.dial": _seconds("TCP connection establishment"),
        "hop.exchange": _seconds("Sender-observed exchange from send to full reply"),
        "hop.handling": _seconds("Receiver-reported handling of one request"),
        "hop.communication": _seconds(
            "Estimated communication residual: exchange minus handling"
        ),
        "hop.request_bytes": (HISTOGRAM, "By", "Request frame bytes", BYTES_BUCKETS),
        "hop.reply_bytes": (HISTOGRAM, "By", "Reply frame bytes", BYTES_BUCKETS),
        "hop.throughput": (
            HISTOGRAM,
            "By/s",
            "Frame bytes divided by exchange time",
            RATE_BUCKETS,
        ),
        "route.step_exchange": _seconds("Client exchange with the first worker"),
        "route.step_communication": _seconds("Summed edge residuals of one step"),
        "worker.session_lock_wait": _seconds("Wait for the session lock"),
        "worker.device_wait": _seconds("Wait for the device limiter"),
        "worker.thread_handoff": _seconds("Handoff to the compute thread"),
        "worker.cpu_prepare": _seconds("Host-side input preparation"),
        "worker.serialize": _seconds("Output serialization to bytes"),
        "worker.stage_execute": _seconds("Wall time of the compute thread body"),
        "worker.gpu": _seconds("CUDA-event interval on the device stream"),
        "network.join": _seconds("DHT join duration"),
        "requests": (
            COUNTER,
            "{request}",
            "Finished chat requests by outcome and phase",
            None,
        ),
        "timeouts": (COUNTER, "{timeout}", "Timeouts by phase", None),
        "worker.rejections": (COUNTER, "{request}", "Refused worker requests", None),
        "tokens.prompt": (COUNTER, "{token}", "Prompt tokens", None),
        "tokens.sampled": (COUNTER, "{token}", "Sampled tokens, including EOS", None),
        "chunks.text": (COUNTER, "{chunk}", "Text chunks produced", None),
        "slow_operations": (COUNTER, "{event}", "Heuristic slow outliers", None),
        "telemetry.incomplete": (COUNTER, "{event}", "Missing measurements", None),
        "network.reconnects": (COUNTER, "{event}", "DHT reconnects", None),
        "requests.active": (UPDOWN, "{request}", "Active chat requests", None),
        "sessions.active": (UPDOWN, "{session}", "Active sessions", None),
        "network.connected": (UPDOWN, "1", "1 while joined to the DHT", None),
    }
)
