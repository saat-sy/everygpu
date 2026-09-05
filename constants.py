"""Project-wide constants and environment-backed configuration."""

import os


# Model artifacts
MODEL_REPOSITORY = "saatwiksy/olmoe-pipeline-fp16"
MODEL_NAME = "olmoe-pipeline-fp16"
LAPTOP_FILES = (
    "manifest.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

# Pipeline configuration
REQUIRED_RUNTIMES = 2
MAX_NEW_TOKENS = 20
PIPELINE_TRACE_NAME = "pipeline.request"

# Profiler configuration
TEMPO_URL = os.getenv("TEMPO_URL", "http://127.0.0.1:3200").rstrip("/")
TEMPO_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "server")
TEMPO_LOOKBACK_HOURS = int(os.getenv("TEMPO_LOOKBACK_HOURS", "168"))

# Telemetry endpoints and units
CLIENT_ENDPOINT_ID = "client"
SERVER_ENDPOINT_ID = "server"
NANOSECONDS_PER_MILLISECOND = 1_000_000
MILLISECONDS_PER_SECOND = 1_000
DEFAULT_OTEL_SERVICE_NAME = "server"

# Telemetry attribute names
MODEL_ATTRIBUTE = "model"
NODE_ID_ATTRIBUTE = "node_id"
STAGE_ID_ATTRIBUTE = "stage_id"
DEVICE_ATTRIBUTE = "device"
LAYER_START_ATTRIBUTE = "layer_start"
LAYER_END_ATTRIBUTE = "layer_end"
PHASE_ATTRIBUTE = "phase"
SOURCE_NODE_ID_ATTRIBUTE = "source_node_id"
TARGET_NODE_ID_ATTRIBUTE = "target_node_id"
MESSAGE_TYPE_ATTRIBUTE = "message_type"
