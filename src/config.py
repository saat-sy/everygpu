"""Application configuration."""

import os


# Model artifacts
MODEL_REPOSITORY = "saatwiksy/olmoe-pipeline-fp16"
MODEL_NAME = "olmoe-pipeline-fp16"
COORDINATOR_FILES = (
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
